"""Native checkpoint persistence, exact CPU epoch continuation and pinned local inference."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from control.app import create_app
from domain import samples, checkpoints as CP
from domain.core import png_b64
from domain.inference import make_model, predict
from graph_core.schema import Graph
from tabular.core import ExecutionError
from test_tabular_api import submit, wait_for

BUILDERS = [(samples.vision_graph, "vision"), (samples.nlp_graph, "nlp"), (samples.speech_graph, "speech")]

@pytest.fixture(params=BUILDERS, ids=["vision", "nlp", "speech"])
def trained(request, tmp_path, domain_fixtures):
    builder, family = request.param
    g = Graph.model_validate(builder(epochs=2)); g.nodes[0].config["n"] = 12
    if family == "vision": g.nodes[-1].config["width"] = 4
    if family == "nlp": g.nodes[-1].config.update(hidden=8, embedding=8)
    if family == "speech": g.nodes[-1].config["hidden"] = 8
    with TestClient(create_app(tmp_path / "wb")) as c:
        r = submit(c, g); assert r.status_code == 201, r.text
        rid = r.json()["runId"]; final = wait_for(c, rid); assert final["status"] == "completed", final
        models = c.get("/api/domain/models", params={"runId": rid}).json()["models"]
        assert len(models) == 1
        yield c, g, models[0], family


def request_record(c, model, family):
    summary = c.post(f'/api/runs/{model["runId"]}/inspect', json={"kind":"summary", "node":model["node"]}).json()["data"]
    if family == "vision":
        record = {"imagePng": summary["samples"][0]["image"]}
    elif family == "nlp":
        record = {"text": summary["samples"][0]["text"]}
    else:
        from speech.audio import load_npz
        from speech.features import pad_features, FeatureConfig, features_for
        source = model["source"]["path"]; clips, _ = load_npz(Path(source), None)
        idx = summary["split"]["valIndices"][0]
        record = {"samples": clips[idx].wave.tolist(), "sampleRate": model["inference"]["sampleRate"]}
    return record, summary


def test_persist_reload_native_predictions_source_independence_and_export(trained):
    c, g, m, family = trained; store = c.app.state.services.store
    raw = c.get(f'/api/domain/models/{m["modelId"]}/checkpoint'); assert raw.status_code == 200
    assert hashlib.sha256(raw.content).hexdigest() == m["checkpointSha256"]
    state = torch.load(io.BytesIO(raw.content), weights_only=True, map_location="cpu")
    recorded = c.get(f'/api/domain/models/{m["modelId"]}/example')
    assert recorded.status_code == 200 and recorded.json()["synthetic"] and recorded.json()["partition"] == "validation"
    assert state["epochs"] == 2 and state["optimizer"]["state"] and state["generator"].dtype == torch.uint8
    record, summary = request_record(c, m, family)
    before = store.max_seq(m["runId"]), store.artifacts(m["runId"])
    result = c.post(f'/api/domain/models/{m["modelId"]}/predict', json={"records":[record]})
    assert result.status_code == 200, result.text
    p = result.json()["predictions"][0]
    batch = c.post(f'/api/domain/models/{m["modelId"]}/predict', json={"records": [record, record]}).json()
    assert batch["predictions"] == [p, p]
    if family == "vision": assert p["maskPng"] == summary["samples"][0]["predMask"]
    elif family == "nlp": assert p["predictions"] == summary["samples"][0]["predTokens"]
    else: assert p["text"] == summary["samples"][0]["hyp"]
    # Native loaded model's forward pass is the reference, including exact preprocessing.
    native = make_model(m); native.load_state_dict(state["weights"]); native.eval()
    if family == "vision":
        from vision.segment import prep
        im = np.array(Image.open(io.BytesIO(base64.b64decode(record["imagePng"]))))
        with torch.no_grad(): mask = native(prep(torch.from_numpy(im).permute(2,0,1)[None]))[0].argmax(0)
        assert np.array_equal(np.array(Image.open(io.BytesIO(base64.b64decode(p["maskPng"])))), mask.numpy())
    elif family == "nlp":
        from tokenizers import Tokenizer
        from nlp.subword import encode_example, batch_tensors
        cfg = m["inference"]; tok = Tokenizer.from_str(cfg["tokenizerJson"])
        ex = encode_example(tok, {"id":"request", "text":record["text"], "spans":[]}, {t:i for i,t in enumerate(cfg["labels"])},cfg["maxLength"],cfg["labelPolicy"],cfg["ignoreIndex"])
        ids, att, _ = batch_tensors([ex],cfg["architecture"]["pad_id"],cfg["ignoreIndex"])
        with torch.no_grad(): prob = native(ids,att)[0].softmax(-1)
        torch.testing.assert_close(torch.tensor(p["probabilities"]),prob,atol=1e-6,rtol=1e-6)
    else:
        from speech.features import FeatureConfig, features_for, pad_features
        from speech.ctc import greedy_decode
        cfg=m["inference"]; f=features_for(torch.tensor(record["samples"]),FeatureConfig(**cfg["features"]),cfg["sampleRate"])
        x,l,_=pad_features([(f-state["mean"])/state["std"]])
        with torch.no_grad(): lp,ol=native(x,l)
        tok,at=greedy_decode(lp[0],int(ol[0]))
        assert tok==p["tokenIds"] and at==p["tokenFrames"]
    assert (store.max_seq(m["runId"]), store.artifacts(m["runId"])) == before
    # New control process can infer from artifacts even when sources are absent.
    src=Path(m["source"]["path"]); backup=src.with_name(src.name+'.checkpoint-test')
    src.rename(backup)
    try:
        with TestClient(create_app(store.root)) as reopened:
            again=reopened.post(f'/api/domain/models/{m["modelId"]}/predict',json={"records":[record]})
            assert again.status_code==200 and again.json()==result.json()
    finally: backup.rename(src)


def test_resume_matches_uninterrupted_weights_optimizer_rng_and_curves(trained):
    c,g,m,family=trained; store=c.app.state.services.store
    resumed=g.model_copy(deep=True);resumed.nodes[-1].config.update(epochs=4,resume_model_id=m["modelId"])
    reference=g.model_copy(deep=True);reference.nodes[-1].config["epochs"]=4
    if family=="nlp":
        resumed.nodes[1].config["fitted_model_id"]=m["modelId"]
        reference.nodes[1].config["fitted_model_id"]=m["modelId"]
    states=[]
    for graph in [resumed,reference]:
        response=submit(c,graph); assert response.status_code==201,response.text
        rid=response.json()["runId"];final=wait_for(c,rid); assert final["status"]=="completed",final
        entry=c.get('/api/domain/models',params={"runId":rid}).json()["models"][0]
        states.append(CP.read_state(store,entry))
        if graph is resumed: assert entry["parentModelId"]==m["modelId"] and entry["epochs"]==4
    a,b=states
    for key in a["weights"]: assert torch.equal(a["weights"][key],b["weights"][key]),key
    assert torch.equal(a["generator"],b["generator"]) and torch.equal(a["torchRng"],b["torchRng"])
    assert a["curve"]==b["curve"]
    for param,values in a["optimizer"]["state"].items():
        for key,value in values.items():
            assert torch.equal(value,b["optimizer"]["state"][param][key]) if isinstance(value,torch.Tensor) else value==b["optimizer"]["state"][param][key]
    assert CP.read_state(store,m)["epochs"]==2  # parent remains immutable


def test_checkpoint_hash_trust_environment_and_input_refusals(trained):
    c,g,m,family=trained;store=c.app.state.services.store
    record,_=request_record(c,m,family)
    assert c.post(f'/api/domain/models/{m["modelId"]}/predict',json={"records":[{}]}).status_code==422
    assert c.post(f'/api/domain/models/{m["modelId"]}/predict',json={"records":[record]*5}).status_code==422
    if family=="speech":
        bad={**record,"sampleRate":8000}
        assert c.post(f'/api/domain/models/{m["modelId"]}/predict',json={"records":[bad]}).json()["detail"]["code"]=="E_AUDIO_SAMPLE_RATE"
    path=store.path_of(m["checkpointSha256"]); original=path.read_bytes();path.write_bytes(b'changed')
    try:
        assert c.post(f'/api/domain/models/{m["modelId"]}/predict',json={"records":[record]}).json()["detail"]["code"]=="E_DOMAIN_CHECKPOINT_INTEGRITY"
    finally:path.write_bytes(original)
    untrusted=store.put_bytes(b'{}')
    assert c.get(f'/api/domain/models/{untrusted}').json()["detail"]["code"]=="E_DOMAIN_CHECKPOINT_TRUST"
    changed=copy.deepcopy(m);changed.pop('modelId');changed['environment']['torch']='wrong'
    artifact=store.add_artifact(m['runId'],'domain_model',json.dumps(changed).encode(),'complete',2,{'internalDomainModel':True,'node':m['node']})
    assert c.get(f'/api/domain/models/{artifact["sha256"]}').json()["detail"]["code"]=="E_DOMAIN_CHECKPOINT_ENVIRONMENT"
    overview=c.get("/api/domain/models").json()["models"]
    assert any(row["modelId"]==m["modelId"] and row["available"] for row in overview)
    assert any(row["modelId"]==artifact["sha256"] and not row["available"] for row in overview)


def test_resume_refuses_changed_training_data_configuration_and_epoch_target(trained):
    c,g,m,family=trained
    for patch,code in [({'lr':0.01,'epochs':4},'E_DOMAIN_RESUME_INCOMPATIBLE'),({'epochs':2},'E_DOMAIN_RESUME_EPOCHS')]:
        draft=g.model_copy(deep=True);draft.nodes[-1].config.update(resume_model_id=m['modelId'],**patch)
        if family=='nlp':draft.nodes[1].config['fitted_model_id']=m['modelId']
        rid=submit(c,draft).json()['runId'];final=wait_for(c,rid)
        assert final['status']=='failed' and code in final['error'],final
    draft=g.model_copy(deep=True);draft.nodes[0].config['n']=10;draft.nodes[-1].config.update(resume_model_id=m['modelId'],epochs=4)
    if family=='nlp':draft.nodes[1].config['fitted_model_id']=m['modelId']
    rid=submit(c,draft).json()['runId'];final=wait_for(c,rid)
    assert final['status']=='failed' and 'E_DOMAIN_RESUME_INCOMPATIBLE' in final['error'],final
