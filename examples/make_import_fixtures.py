"""Deterministic labelled SYNTHETIC files for the bounded COCO/CoNLL/WAV import journey."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import wavfile
from pycocotools import mask as M


def generate(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(8):
        image = np.full((32, 40, 3), 20, np.uint8)
        for j, (x, y, w, h, category) in enumerate([(3+i%3, 4, 16, 15, 7), (12, 12, 18, 14, 42)]):
            image[y:y+h, x:x+w] = [(220, 40, 60), (40, 180, 220)][j]
            mask = np.zeros((32, 40), np.uint8); mask[y:y+h, x:x+w] = 1
            rle = M.encode(np.asfortranarray(mask)); rle["counts"] = rle["counts"].decode("ascii")
            segmentation = [[x,y,x+w,y,x+w,y+h,x,y+h]] if j == 0 else rle
            annotations.append({"id": 10*i+j, "image_id": i, "category_id": category, "bbox": [x,y,w,h],
                "segmentation": segmentation, "iscrowd": 0, "area": w*h, "keypoints": [x+2,y+2,2,x+w-2,y+2,1], "num_keypoints": 2})
        name = f"image_{i}.png"; Image.fromarray(image).save(root/name)
        images.append({"id": i, "file_name": name, "width": 40, "height": 32, "license": 1})
    coco = {"info": {"description": "SYNTHETIC overlapping rectangles; no real-image quality claim"}, "images": images,
        "categories": [{"id": i, "name": n, "keypoints": ["left", "right"]} for i,n in [(7,"red"),(42,"blue")]],
        "annotations": annotations, "licenses": [{"id":1,"name":"Generated SYNTHETIC fixture; CC0-1.0"}]}
    (root/"coco.json").write_text(json.dumps(coco, sort_keys=True), encoding="utf-8")
    sentences = ["Zoë B-PER\nvisits O\nNew B-LOC\nYork I-LOC\n. O", "Ali B-PER\nworks O\nin O\nParis B-LOC\n. O"] * 4
    (root/"ner.conll").write_text("-DOCSTART- O\n\n"+"\n\n".join(sentences)+"\n", encoding="utf-8")
    manifest = []
    for i in range(8):
        text = "ab" if i%2 == 0 else "ba"; n = 1600+i*64; t = np.arange(n)/16000
        hz = np.where(np.arange(n)<n//2, 440 if text[0]=="a" else 660, 440 if text[1]=="a" else 660)
        wave = np.stack([.4*np.sin(2*np.pi*hz*t), .2*np.sin(2*np.pi*hz*t)], 1)
        name=f"tone_{i}.wav"; wavfile.write(root/name,16000,(wave*32767).astype(np.int16))
        manifest.append({"id": f"synthetic_tone_{i}", "file": name, "text": text})
    (root/"audio.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (root/"PROVENANCE.txt").write_text("All files are generated SYNTHETIC teaching fixtures. CC0-1.0. No real-world quality evidence.\n")
    return root


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out", required=True); a=p.parse_args()
    print(generate(a.out))
