"""Native conversion, refusal, provenance and worker journeys for bounded local domain imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pycocotools.coco import COCO
from pycocotools import mask as M
from scipy.io import wavfile
from seqeval.metrics.sequence_labeling import get_entities

from artifact_store import ArtifactStore
from connectors.domain_import import import_dataset, provenance, check_compressed_rle
from control.app import create_app
from domain import samples
from graph_core.schema import Graph
from speech.audio import load_npz as audio_load
from vision.contract import load_npz as vision_load, validate_sample
from vision.transforms import apply_step, Step, BoxPolicy
from tabular.core import ExecutionError
from test_tabular_api import submit, wait_for


@pytest.fixture
def source(tmp_path):
    spec=importlib.util.spec_from_file_location("import_fixtures",Path(__file__).parents[1]/"examples/make_import_fixtures.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.generate(tmp_path/"source")


def request(source, kind, **kw):
    return {"kind":kind,"path":str(source/{"coco":"coco.json","conll":"ner.conll","wav":"audio.json"}[kind]),
        "root":str(source),"synthetic":True,"license":"CC0-1.0; generated SYNTHETIC fixture", "flip_pairs":[[0,1]] if kind=="coco" else [],**kw}


@pytest.mark.parametrize("kind",["coco","conll","wav"])
def test_snapshot_determinism_source_independence_and_hash_integrity(source,tmp_path,kind):
    store=ArtifactStore(tmp_path/"wb");req=request(source,kind)
    a=import_dataset(store,req);b=import_dataset(store,req);assert a==b
    assert a["synthetic"] and a["license"]["authority"]=="user supplied; not independently verified"
    for original in a["sources"]:
        raw=Path(original["path"]).read_bytes(); assert hashlib.sha256(raw).hexdigest()==original["sha256"]
        assert store.read_artifact(original["sha256"])==raw
        Path(original["path"]).unlink()
    path=Path(a["path"]);p=provenance(path,None,"")
    assert p["datasetId"]==a["id"] and p["sha256"]==a["payloadSha256"] and p["license"]==a["license"]
    original=path.read_bytes();path.write_bytes(original+b"tamper")
    with pytest.raises(ExecutionError,match="hash differs") as e:provenance(path,None,"")
    assert e.value.code=="E_DATASET_INTEGRITY"
    path.write_bytes(original);path.with_name("manifest.json").write_text("garbled")
    with pytest.raises(ExecutionError) as e:provenance(path,None,"")
    assert e.value.code=="E_DATASET_INTEGRITY"


def test_coco_matches_native_masks_boxes_keypoints_and_flip(source,tmp_path):
    native=COCO(str(source/"coco.json"));a=import_dataset(ArtifactStore(tmp_path/"wb"),request(source,"coco"))
    data,spec,raw=vision_load(a["path"],None)
    assert raw["categoryIds"]==[7,42] and raw["size"]==[32,40] and raw["cocoLicenses"][0]["id"]==1
    for i,s in enumerate(data):
        validate_sample(s,spec);anns=native.imgToAnns[i]
        for j,ann in enumerate(anns):
            assert np.array_equal(s.masks[j].numpy(),native.annToMask(ann))
            x,y,w,h=ann["bbox"]; assert s.boxes[j].tolist()==[x,y,x+w,y+h]
            assert s.keypoints[j].tolist()==np.array(ann["keypoints"]).reshape(-1,3)[:,:2].tolist()
        assert (s.masks[0]&s.masks[1]).any()  # overlap survives conversion
        flipped,_=apply_step(s,spec,Step(op="hflip"),BoxPolicy())
        assert torch.equal(flipped.masks,s.masks.flip(-1))
        expected=s.keypoints[:,[1,0],:].clone();expected[:,:,0]=40-expected[:,:,0]
        assert torch.equal(flipped.keypoints,expected) and torch.equal(flipped.visibility,s.visibility[:,[1,0]])
        assert s.meta["imported"]["annotationIds"]==[10*i,10*i+1]
    # Native uncompressed RLE has the same pixels as native compressed/polygon decoding.
    raw=json.loads((source/"coco.json").read_text());mask=native.annToMask(raw["annotations"][0]).flatten(order="F")
    counts=[];last=0;n=0
    for pixel in mask:
        if pixel==last:n+=1
        else:counts.append(n);n=1;last=int(pixel)
    counts.append(n);raw["annotations"][0]["segmentation"]={"size":[32,40],"counts":counts}
    (source/"coco.json").write_text(json.dumps(raw))
    b=import_dataset(ArtifactStore(tmp_path/"wb2"),request(source,"coco"))
    assert torch.equal(vision_load(b["path"],None)[0][0].masks,data[0].masks)


def test_rle_validation_matches_native_signed_delta_encoding():
    rng=np.random.default_rng(3)
    for h,w in [(1,1),(32,40),(512,512)]:
        for image in [np.zeros((h,w),np.uint8),np.ones((h,w),np.uint8),(rng.random((h,w))>.7).astype(np.uint8)]:
            encoded=M.encode(np.asfortranarray(image));check_compressed_rle(encoded["counts"].decode(),h*w)
            assert np.array_equal(image,M.decode(encoded))
    for text in ["", "!", "P", "1"]:
        with pytest.raises(ExecutionError) as e:check_compressed_rle(text,100)
        assert e.value.code=="E_DATASET_MASK"


def test_conll_unicode_entities_match_seqeval_and_record_reconstruction(source,tmp_path):
    a=import_dataset(ArtifactStore(tmp_path/"wb"),request(source,"conll"))
    records=[json.loads(x) for x in Path(a["path"]).read_text().splitlines()]
    assert records[0]["text"]=="Zoë visits New York ." and records[0]["sourceLines"]==[3,4,5,6,7]
    assert "reconstructed" in a["contract"]["textPolicy"]
    for r in records:
        starts=[];offset=0
        for t in r["originalTokens"]:starts.append(offset);offset+=len(t)+1
        reference=[{"label":ty,"start":starts[start],"end":starts[end]+len(r["originalTokens"][end])} for ty,start,end in get_entities(r["originalTags"])]
        assert r["spans"]==reference


@pytest.mark.parametrize("dtype",[np.uint8,np.int16,np.int32,np.float32,np.float64])
def test_wav_native_normalization_channels_lengths_and_missing_timestamps(source,tmp_path,dtype):
    entries=json.loads((source/"audio.json").read_text())
    raw=[]
    for i,e in enumerate(entries):
        if dtype==np.uint8:x=np.tile(np.array([[0,255],[128,64]],dtype=dtype),(256+i,1))
        elif np.issubdtype(dtype,np.signedinteger):x=np.tile(np.array([[np.iinfo(dtype).min,np.iinfo(dtype).max],[1,0]],dtype=dtype),(256+i,1))
        else:x=np.tile(np.array([[-.75,.9],[.1,0]],dtype=dtype),(256+i,1))
        wavfile.write(source/e["file"],16000,x);raw.append(x)
    a=import_dataset(ArtifactStore(tmp_path/"wb"),request(source,"wav"));clips,spec=audio_load(a["path"],None)
    assert spec["channels"]==2
    for c,x in zip(clips,raw):
        ref=(x.astype(np.float32)-128)/128 if dtype==np.uint8 else x.astype(np.float32)/float(2**(x.dtype.itemsize*8-1)) if np.issubdtype(dtype,np.signedinteger) else x.astype(np.float32)
        assert np.array_equal(c.wave.numpy(),ref.T) and c.samples_per_channel==len(x) and c.channels==2
        assert c.segments is None and c.duration==len(x)/16000


def test_signed_24_bit_wav_matches_native_left_justified_pcm(source,tmp_path):
    import wave
    values=np.tile(np.array([-8388608,8388607,1,-1],np.int64),128)
    encoded=b"".join((int(v)&0xffffff).to_bytes(3,"little") for v in values)
    for e in json.loads((source/"audio.json").read_text()):
        with wave.open(str(source/e["file"]),"wb") as f:
            f.setnchannels(1);f.setsampwidth(3);f.setframerate(16000);f.writeframes(encoded)
    a=import_dataset(ArtifactStore(tmp_path/"wb"),request(source,"wav"))
    clips,_=audio_load(a["path"],None)
    assert np.array_equal(clips[0].wave.numpy()[0],(values/8388608).astype(np.float32))
    _,native=wavfile.read(source/"tone_0.wav")
    assert np.array_equal(clips[0].wave.numpy()[0],native.astype(np.float32)/2147483648)


def test_bounded_sources_records_and_annotation_allocation(source,tmp_path,monkeypatch):
    import connectors.domain_import as D
    store=ArtifactStore(tmp_path/"wb")
    entries=json.loads((source/"audio.json").read_text());(source/"audio.json").write_text(json.dumps(entries*33))
    with pytest.raises(ExecutionError) as e:import_dataset(store,request(source,"wav"))
    assert e.value.code=="E_DATASET_BOUNDS"
    monkeypatch.setattr(D,"MAX_INPUT",32)
    with pytest.raises(ExecutionError) as e:import_dataset(store,request(source,"conll"))
    assert e.value.code=="E_DATASET_BOUNDS" and not list(store.root.glob("domain-datasets/*/manifest.json"))


@pytest.mark.parametrize("change,code",[("crowd","E_DATASET_UNSUPPORTED"),("boxonly","E_DATASET_UNSUPPORTED"),("box","E_DATASET_BOX"),("mask","E_DATASET_MASK"),("kp","E_DATASET_KEYPOINT"),("geometry","E_DATASET_BOUNDS"),("path","E_DATASET_PATH"),("pairs","E_DATASET_FORMAT")])
def test_coco_refusals_leave_no_ready_import(source,tmp_path,change,code):
    r=json.loads((source/"coco.json").read_text());req=request(source,"coco")
    if change=="crowd":r["annotations"][0]["iscrowd"]=1
    if change=="boxonly":del r["annotations"][0]["segmentation"]
    if change=="box":r["annotations"][0]["bbox"][2]=100
    if change=="mask":r["annotations"][1]["segmentation"]["counts"]="!"
    if change=="kp":r["annotations"][0]["keypoints"][2]=3
    if change=="geometry":r["images"][0]["width"]=513
    if change=="path":r["images"][0]["file_name"]="../secret.png"
    if change=="pairs":req["flip_pairs"]=[[0,0]]
    (source/"coco.json").write_text(json.dumps(r));store=ArtifactStore(tmp_path/"wb")
    with pytest.raises(ExecutionError) as e:import_dataset(store,req)
    assert e.value.code==code and not list(store.root.glob("domain-datasets/*/manifest.json"))


@pytest.mark.parametrize("change,code",[("rate","E_DATASET_AUDIO_RATE"),("channels","E_DATASET_AUDIO_CHANNELS"),("nan","E_DATASET_AUDIO"),("too_long","E_DATASET_BOUNDS"),("symlink","E_DATASET_PATH"),("iob","E_DATASET_IOB2"),("columns","E_DATASET_FORMAT")])
def test_audio_text_and_symlink_refusals(source,tmp_path,change,code):
    kind="conll" if change in ("iob","columns") else "wav";req=request(source,kind)
    if change in ("rate","channels","nan","too_long"):
        wavfile.write(source/"tone_0.wav",8000 if change=="rate" else 16000,
            np.full((32001 if change=="too_long" else 512,1 if change=="channels" else 2),np.nan if change=="nan" else 0,np.float32))
    if change=="symlink":
        (tmp_path/"outside.wav").write_bytes((source/"tone_0.wav").read_bytes());(source/"tone_0.wav").unlink();(source/"tone_0.wav").symlink_to(tmp_path/"outside.wav")
    if change=="iob":(source/"ner.conll").write_text("bad I-PER\n\nother O\n")
    if change=="columns":req["label_column"]=10
    with pytest.raises(ExecutionError) as e:import_dataset(ArtifactStore(tmp_path/"wb"),req)
    assert e.value.code==code


def test_api_requires_explicit_declarations_and_lists_integrity(source,tmp_path):
    with TestClient(create_app(tmp_path/"wb",api_token="dataset-import-test-token")) as c:
        assert c.post("/api/domain/datasets",json=request(source,"conll")).status_code==401
        c.headers["Authorization"]="Bearer dataset-import-test-token"
        for key in ("synthetic","license"):
            req=request(source,"conll");del req[key];assert c.post("/api/domain/datasets",json=req).status_code==422
        for patch in ({"synthetic":"false"},{"license":" "},{"surprise":1},{"kind":"wav","root":None}):
            assert c.post("/api/domain/datasets",json={**request(source,"conll"),**patch}).status_code==422
        a=c.post("/api/domain/datasets",json=request(source,"conll")).json()
        assert c.get("/api/domain/datasets").json()["datasets"][0]["id"]==a["id"]
        Path(a["path"]).write_text("changed")
        assert c.get("/api/domain/datasets").json()["datasets"][0]["error"]=="E_DATASET_INTEGRITY"


@pytest.mark.parametrize("kind,builder",[("coco",samples.vision_graph),("conll",samples.nlp_graph),("wav",samples.speech_graph)])
def test_imported_worker_train_checkpoint_inference_and_declared_flag_transport(source,tmp_path,kind,builder):
    # Synthetic generated input remains labelled in the fixture. This false declaration tests transport,
    # not the declaration's truth: the importer explicitly reports that it is user-supplied.
    with TestClient(create_app(tmp_path/"wb")) as c:
        response=c.post("/api/domain/datasets",json=request(source,kind,synthetic=False));assert response.status_code==201,response.text
        a=response.json();g=Graph.model_validate(builder(a["path"],epochs=2));assert g.nodes[0].type=={"coco":"domain.vision_source","conll":"domain.nlp_source","wav":"domain.audio_source"}[kind];g.nodes[-1].config.update({"width":4} if kind=="coco" else {"hidden":8,**({"embedding":8} if kind=="conll" else {})})
        r=submit(c,g);assert r.status_code==201,r.text
        rid=r.json()["runId"];final=wait_for(c,rid);assert final["status"]=="completed",final
        m=c.get("/api/domain/models",params={"runId":rid}).json()["models"][0];assert m["available"]
        assert m["source"]["synthetic"] is False and m["source"]["datasetId"]==a["id"]
        for node in (g.nodes[0],g.nodes[-1]):
            s=c.post(f"/api/runs/{rid}/inspect",json={"kind":"summary","node":node.id}).json()["data"]
            assert s["synthetic"] is False and "user-declared" in s["note"]
        ex=c.get(f'/api/domain/models/{m["modelId"]}/example').json()
        assert ex["synthetic"] is False and ex["source"]["license"]==a["license"]
        if kind=="wav":assert len(ex["records"][0]["samples"])==2 and m["inference"]["channels"]==2
        Path(a["path"]).unlink()  # pinned model inference is independent of the imported canonical file
        prediction=c.post(f'/api/domain/models/{m["modelId"]}/predict',json={"records":ex["records"]})
        assert prediction.status_code==200,prediction.text
        assert prediction.json()["provenance"]["source"]==m["source"]
        v=c.post("/api/production/versions",json={"runId":rid,"node":m["node"],"name":"import-test","owner":"tests","intendedUse":"declaration transport test","limitations":"SYNTHETIC input; declaration truth not tested"})
        assert v.status_code==201,v.text
        ref=c.get(f'/api/production/versions/{v.json()["id"]}/reference-input').json()
        assert "SYNTHETIC held-out" not in ref["labelNote"] and ref["provenance"]["source"]==m["source"]
