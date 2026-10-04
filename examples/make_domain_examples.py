"""Write the Milestone 6b example projects (graph JSON + UI JSON) into examples/. Deterministic.

    python examples/make_domain_fixtures.py     # the SYNTHETIC fixtures first
    python examples/make_domain_examples.py

- vision_segmentation_synthetic   annotated shapes -> annotation-preserving transforms -> tiny FCN, IoU / Dice / mAP
- nlp_token_classification        span corpus -> WordPiece subwords with label alignment -> BiGRU tagger, seqeval-convention F1
- speech_ctc_tones                tone-sequence audio -> mel features -> CTC model, greedy decoding, CER / WER alignment"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "python"), str(HERE)]

from domain import samples as ds  # noqa: E402
from graph_core.hashing import semantic_hash  # noqa: E402
from graph_core.schema import Graph  # noqa: E402
from graph_core.validate import validate  # noqa: E402
from make_m5_examples import layout  # noqa: E402

EXAMPLES = {
    "vision_segmentation_synthetic": (ds.vision_graph(), "SYNTHETIC drawn shapes (rectangles and disks) with boxes, instance masks and keypoints. Transforms keep every annotation consistent with the image; a tiny FCN is trained for a few seconds. "
                                      "Run the graph, then open the nodes: sample overlays, transform before/after with the applied parameters, prediction vs ground truth, IoU/Dice and mAP (torchmetrics)."),
    "nlp_token_classification": (ds.nlp_graph(), "SYNTHETIC template sentences with made-up names and CHARACTER-offset entity spans (PER, ORG, LOC). A WordPiece tokenizer is trained offline on the training sentences; labels follow a declared alignment policy. "
                                 "Open the tokenizer and tagger nodes: text, subwords, labels and predictions aligned; span metrics follow seqeval conventions."),
    "speech_ctc_tones": (ds.speech_graph(), "SYNTHETIC tone sequences (not speech): each symbol is a pure tone, 16 kHz mono, clips of 1.0/1.5/2.0 s (a 2 s clip has exactly 32,000 samples). Mel features with declared window/hop/padding, "
                         "a tiny CTC model and greedy decoding; open the nodes for waveform, spectrogram, frame-to-token alignment and the S/D/I error alignment."),
}


def main(out: Path = HERE) -> None:
    for name, (gj, desc) in EXAMPLES.items():
        g = Graph.model_validate(gj)
        rep = validate(g)
        assert rep.ok, (name, [(d.code, d.message) for d in rep.errors])
        (out / f"{name}.project.json").write_text(json.dumps(g.to_json(), indent=2, ensure_ascii=False) + "\n")
        ui = layout(g)
        ui["description"], ui["synthetic"] = desc, True
        (out / f"{name}.ui.json").write_text(json.dumps(ui, indent=2, ensure_ascii=False) + "\n")
        print("wrote", name, semantic_hash(g)[:12])


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
