"""Vision workflow (A56): annotation contracts, annotation-preserving transforms, segmentation / detection metrics against native references."""
from __future__ import annotations

import copy

import pytest
import torch
from torchvision import tv_tensors
from torchvision.ops import box_convert, box_iou

from conftest import load_domain_generator
from vision import contract as C
from vision import segment as S
from vision import transforms as T


@pytest.fixture(scope="module")
def data(domain_fixtures):
    samples, spec, raw = C.load_npz(domain_fixtures.VISION_NPZ, 40)
    return samples, spec, raw


# ------------------------------------------------------------------------------------------------ contract
def test_fixture_is_deterministic_and_labelled_synthetic(tmp_path):
    gen = load_domain_generator()
    a, b = gen.write_vision(tmp_path / "a.npz", n=12), gen.write_vision(tmp_path / "b.npz", n=12)
    assert a.read_bytes() == b.read_bytes()
    assert gen.write_vision(tmp_path / "c.npz", n=12, seed=1).read_bytes() != a.read_bytes()
    _, _, raw = C.load_npz(a, None)
    assert raw["synthetic"] is True and raw["boxFormat"] == "xyxy"


def test_loaded_samples_satisfy_the_contract_and_are_internally_consistent(data):
    samples, spec, _ = data
    for s in samples:
        C.validate_sample(s, spec)
        c = C.consistency(s, spec, 0.0)
        assert c["ok"], c  # boxes = tight bounds of the visible mask; visible keypoints are on their mask
    assert any(int((s.visibility == 1).sum()) for s in samples)  # the fixture really contains occluded keypoints


@pytest.mark.parametrize("fmt", C.BOX_FORMATS)
@pytest.mark.parametrize("coords", C.COORDS)
def test_box_conversion_roundtrip_and_native_reference(fmt, coords):
    xyxy = torch.tensor([[2.0, 3.0, 12.0, 9.0], [0.0, 0.0, 48.0, 24.0]])
    size = (24, 48)
    b = C.from_xyxy_pixel(xyxy, fmt, coords, size)
    assert torch.allclose(C.to_xyxy_pixel(b, fmt, coords, size), xyxy)
    if coords == "pixel":
        assert torch.allclose(b, box_convert(xyxy, "xyxy", fmt))  # torchvision's own conversion
    else:
        assert b.max() <= 1.0 and b.min() >= 0.0


def test_wrong_declared_format_is_caught_by_the_contract(data):
    samples, spec, _ = data
    s = samples[0].clone()
    s.boxes = C.convert_boxes(s.boxes, ("xyxy", "pixel"), ("xywh", "pixel"), s.size)
    wrong = C.VisionSpec(spec.classes, spec.keypoint_names, spec.flip_pairs, "xyxy", "pixel")  # claims xyxy, values are xywh
    # x2 <= x1 happens for these boxes only when the width is smaller than x1: a mislabelled format is NOT always detectable by shape alone, so
    # the check is the consistency against the mask, which always is
    assert not C.consistency(s, wrong, 0.0)["ok"]
    bad = s.clone()
    bad.labels = bad.labels[:-1] if len(bad.labels) else bad.labels
    if len(s.labels):
        with pytest.raises(C.ContractError) as e:
            C.validate_sample(bad, spec)
        assert e.value.code == "E_VISION_LABEL_COUNT"


# ------------------------------------------------------------------------------------------------ transforms: boxes / masks / keypoints match the image
EXACT = {
    "hflip": [T.Step(op="hflip")], "vflip": [T.Step(op="vflip")], "rot90_1": [T.Step(op="rot90", k=1)], "rot90_2": [T.Step(op="rot90", k=2)], "rot90_3": [T.Step(op="rot90", k=3)],
    "pad": [T.Step(op="pad", pad=[3, 5, 7, 2])], "resize2x": [T.Step(op="resize", size=[96, 96])], "chain": [T.Step(op="vflip"), T.Step(op="rot90", k=1), T.Step(op="hflip"), T.Step(op="pad", pad=[1, 2, 3, 4])],
    "affine_translate": [T.Step(op="affine", translate=[3, 2])],
}


@pytest.mark.parametrize("name", list(EXACT))
def test_boxes_masks_keypoints_match_the_transformed_image_exactly(data, name):
    samples, spec, _ = data
    for i, s in enumerate(samples):
        o, _ = T.apply_pipeline(s, spec, EXACT[name], T.BoxPolicy(), i)
        C.validate_sample(o, spec)
        c = C.consistency(o, spec, 0.0)
        assert c["ok"], (name, s.meta["id"], c)
        assert len(o.boxes) == len(s.boxes)


def test_image_and_masks_equal_native_torch_operations(data):
    samples, spec, _ = data
    s = samples[3]
    o, _ = T.apply_step(s, spec, T.Step(op="rot90", k=1), T.BoxPolicy())
    assert torch.equal(o.image, torch.rot90(s.image, 1, (1, 2))) and torch.equal(o.masks, torch.rot90(s.masks, 1, (1, 2)))
    o, _ = T.apply_step(s, spec, T.Step(op="hflip"), T.BoxPolicy())
    assert torch.equal(o.image, s.image.flip(-1)) and torch.equal(o.masks, s.masks.flip(-1))
    o, _ = T.apply_step(s, spec, T.Step(op="vflip"), T.BoxPolicy())
    assert torch.equal(o.image, s.image.flip(-2))


def test_boxes_equal_torchvision_bounding_box_transforms(data):
    from torchvision.transforms.v2 import functional as F

    samples, spec, _ = data
    s = samples[5]
    tb = tv_tensors.BoundingBoxes(s.boxes, format="XYXY", canvas_size=s.size, clamping_mode=None)
    for step, fn in [(T.Step(op="hflip"), F.horizontal_flip), (T.Step(op="rot90", k=1), lambda x: F.rotate(x, 90.0, expand=True)), (T.Step(op="pad", pad=[3, 5, 7, 2]), lambda x: F.pad(x, [3, 5, 7, 2]))]:
        o, _ = T.apply_step(s, spec, step, T.BoxPolicy(clip=False, min_size=0))
        assert torch.allclose(o.boxes, torch.as_tensor(fn(tb)))


@pytest.mark.parametrize("name", ["hflip", "vflip", "rot90_1", "rot90_2", "rot90_3", "pad", "chain"])  # resize interpolates the image; it is covered by the mask-based exactness test
def test_a_keypoint_marker_painted_on_the_image_follows_the_transformed_keypoint(name):
    """Independent of the masks: paint the pixel under each keypoint with a unique colour, transform, and find it again."""
    H, W = 20, 30
    img = torch.zeros(3, H, W, dtype=torch.uint8)
    kps = torch.tensor([[[4.5, 3.5], [20.5, 11.5], [10.5, 17.5]]])
    for k, (x, y) in enumerate(kps[0].tolist()):
        img[:, int(y), int(x)] = torch.tensor([255, 10 * (k + 1), 7])
    s = C.VisionSample(img, torch.tensor([[2.0, 2.0, 25.0, 19.0]]), torch.tensor([0]), None, kps.clone(), torch.full((1, 3), 2, dtype=torch.int8))
    spec = C.VisionSpec(["x"], ["a", "b", "c"], [])
    o, _ = T.apply_pipeline(s, spec, EXACT[name], T.BoxPolicy(clip=False, min_size=0))
    for k in range(3):
        ys, xs = torch.nonzero((o.image[0] == 255) & (o.image[1] == 10 * (k + 1)), as_tuple=True)
        assert (int(xs[0]), int(ys[0])) == (int(o.keypoints[0, k, 0]), int(o.keypoints[0, k, 1])), (name, k)


def test_keypoint_flip_pairs_are_swapped_only_when_declared(data):
    samples, spec, _ = data
    s = next(x for x in samples if len(x.boxes) and (x.labels == 0).any())
    i = int((s.labels == 0).nonzero()[0])
    assert s.keypoints[i, 0, 0] < s.keypoints[i, 1, 0]  # left end is left of right end
    o, lg = T.apply_step(s, spec, T.Step(op="hflip"), T.BoxPolicy())
    assert o.keypoints[i, 0, 0] < o.keypoints[i, 1, 0] and lg["keypointPairsSwapped"] == [[0, 1]]
    undeclared = C.VisionSpec(spec.classes, spec.keypoint_names, [])
    o2, lg2 = T.apply_step(s, undeclared, T.Step(op="hflip"), T.BoxPolicy())
    assert o2.keypoints[i, 0, 0] > o2.keypoints[i, 1, 0] and "keypointPairsSwapped" not in lg2  # not swapped: 'left' now sits on the right
    assert torch.equal(o.keypoints[i, 0], o2.keypoints[i, 1])


def test_flip_twice_is_identity(data):
    samples, spec, _ = data
    s = samples[7]
    o, _ = T.apply_pipeline(s, spec, [T.Step(op="hflip"), T.Step(op="hflip")], T.BoxPolicy())
    assert torch.equal(o.image, s.image) and torch.allclose(o.boxes, s.boxes) and torch.allclose(o.keypoints, s.keypoints) and torch.equal(o.visibility, s.visibility)


def test_declared_box_format_survives_transforms(data):
    samples, spec, _ = data
    s = samples[2]
    for fmt, coords in [("xywh", "pixel"), ("cxcywh", "normalized"), ("xyxy", "normalized")]:
        sp = C.VisionSpec(spec.classes, spec.keypoint_names, spec.flip_pairs, fmt, coords)
        c = s.clone()
        c.boxes = C.convert_boxes(s.boxes, ("xyxy", "pixel"), (fmt, coords), s.size)
        c.keypoints = C.keypoints_from_pixel(s.keypoints, coords, s.size)
        o, _ = T.apply_pipeline(c, sp, [T.Step(op="rot90", k=1), T.Step(op="pad", pad=[4, 0, 0, 4])], T.BoxPolicy())
        ref, _ = T.apply_pipeline(s, spec, [T.Step(op="rot90", k=1), T.Step(op="pad", pad=[4, 0, 0, 4])], T.BoxPolicy())
        assert torch.allclose(C.to_xyxy_pixel(o.boxes, fmt, coords, o.size), ref.boxes, atol=1e-4)
        assert C.consistency(o, sp, 1e-3)["ok"]


# ------------------------------------------------------------------------------------------------ box policy
def test_crop_clips_and_removes_by_the_declared_policy(data):
    samples, spec, _ = data
    steps = [T.Step(op="crop", top=0, left=0, height=24, width=24)]
    removed_any = False
    for strict_frac in (0.0, 0.5):
        for i, s in enumerate(samples):
            o, lg = T.apply_pipeline(s, spec, steps, T.BoxPolicy(clip=True, min_visible_fraction=strict_frac, refit_to_mask=True), i)
            n = len(o.boxes)
            assert len(o.labels) == n and o.masks.shape[0] == n and o.keypoints.shape[0] == n and o.visibility.shape[0] == n  # rows removed together
            assert len(lg[0]["removed"]) == len(s.boxes) - n
            removed_any |= bool(lg[0]["removed"])
            xyxy = o.boxes
            assert (xyxy[:, [0, 2]] >= 0).all() and (xyxy[:, [0, 2]] <= 24).all() and (xyxy[:, [1, 3]] >= 0).all() and (xyxy[:, [1, 3]] <= 24).all()
            assert C.consistency(o, spec, 0.0)["ok"]
    assert removed_any
    # stricter policy never keeps more instances
    s = samples[0]
    a, _ = T.apply_pipeline(s, spec, [T.Step(op="crop", top=10, left=10, height=20, width=20)], T.BoxPolicy(min_visible_fraction=0.0))
    b, _ = T.apply_pipeline(s, spec, [T.Step(op="crop", top=10, left=10, height=20, width=20)], T.BoxPolicy(min_visible_fraction=0.9))
    assert len(b.boxes) <= len(a.boxes)


def test_not_clipping_keeps_boxes_outside_the_canvas_and_keypoints_outside_become_invisible(data):
    samples, spec, _ = data
    s = next(x for x in samples if len(x.boxes) and x.boxes[:, 0].min() < 10)
    o, lg = T.apply_step(s, spec, T.Step(op="crop", top=0, left=12, height=48, width=36), T.BoxPolicy(clip=False, min_visible_fraction=0.0))
    assert (o.boxes[:, 0] < 0).any() or lg["removed"]
    assert lg["keypointsMarkedInvisible"] >= 0 and (o.visibility[(o.keypoints[..., 0] < 0)] == 0).all()


def test_random_flip_records_the_actual_decision_and_is_seeded(data):
    samples, spec, _ = data
    st = T.Step(op="random_hflip", p=0.5, seed=3)
    runs = [[T.apply_step(s, spec, st, T.BoxPolicy(), i)[1]["applied"] for i, s in enumerate(samples[:20])] for _ in range(2)]
    assert runs[0] == runs[1] and 0 < sum(runs[0]) < 20


def test_static_size_and_crop_error_codes():
    assert T.static_size([T.Step(op="rot90", k=1), T.Step(op="pad", pad=[1, 1, 1, 1])], (20, 30)) == (32, 22)
    with pytest.raises(C.ContractError) as e:
        T.static_size([T.Step(op="crop", top=10, left=0, height=40, width=10)], (48, 48))
    assert e.value.code == "E_VISION_TRANSFORM_CROP"


# ------------------------------------------------------------------------------------------------ metrics vs native references
def test_box_iou_against_torchvision_and_hand_calculation():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)
    assert torch.allclose(iou, torch.tensor([[50.0 / 150.0, 1.0, 0.0]]))


def test_segmentation_metrics_hand_calculation_and_torchmetrics():
    from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

    target = torch.tensor([[0, 0, 1, 1], [0, 2, 2, 1]])
    pred = torch.tensor([[0, 1, 1, 1], [0, 2, 0, 1]])
    m = S.seg_metrics(S.confusion(pred, target, 3))
    # class 1: TP 3, FP 1, FN 0 -> 3/4;  class 2: TP 1, FP 0, FN 1 -> 1/2;  class 0: TP 2, FP 1, FN 1 -> 2/4
    assert m["iou"] == pytest.approx([0.5, 0.75, 0.5]) and m["dice"] == pytest.approx([2 * 2 / 6, 6 / 7, 2 / 3])
    ref_iou = MulticlassJaccardIndex(3, average=None)(pred, target)
    ref_f1 = MulticlassF1Score(3, average=None)(pred, target)  # Dice == F1 per class
    assert torch.allclose(torch.tensor(m["iou"], dtype=torch.float32), ref_iou) and torch.allclose(torch.tensor(m["dice"], dtype=torch.float32), ref_f1)
    assert m["pixelAccuracy"] == pytest.approx(6 / 8)


def test_map_matches_hand_calculation():
    gt = [{"boxes": torch.tensor([[0.0, 0, 10, 10]]), "labels": torch.tensor([0])}]
    perfect = [{"boxes": torch.tensor([[0.0, 0, 10, 10]]), "labels": torch.tensor([0]), "scores": torch.tensor([0.9])}]
    assert S.map_metrics(perfect, gt)["map"] == pytest.approx(1.0)
    shifted = [{"boxes": torch.tensor([[0.0, 0, 10, 15]]), "labels": torch.tensor([0]), "scores": torch.tensor([0.9])}]  # IoU 2/3: a hit at 0.50, a miss at 0.75
    m = S.map_metrics(shifted, gt)
    assert m["map50"] == pytest.approx(1.0) and m["map75"] == pytest.approx(0.0)
    wrong_class = [{"boxes": torch.tensor([[0.0, 0, 10, 10]]), "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}]
    assert S.map_metrics(wrong_class, gt)["map"] == pytest.approx(0.0)


def test_components_turn_a_predicted_mask_into_scored_boxes():
    pred = torch.zeros(10, 10, dtype=torch.int64)
    pred[2:5, 3:7] = 1  # rows 2-4, cols 3-6
    pred[7:9, 0:2] = 2
    prob = torch.zeros(3, 10, 10)
    prob[0] = 0.2
    prob[1][pred == 1] = 0.9
    prob[2][pred == 2] = 0.6
    b, l, s = S.components(prob, pred)
    assert b.tolist() == [[3.0, 2.0, 7.0, 5.0], [0.0, 7.0, 2.0, 9.0]] and l.tolist() == [0, 1] and s.tolist() == pytest.approx([0.9, 0.6])


def test_tiny_fcn_trains_on_the_fixture_and_beats_background(data):
    samples, spec, _ = data
    res = S.train_and_eval(samples, spec, epochs=30, lr=5e-3, width=8, batch_size=6, seed=0, val_fraction=0.25, n_inspect=3)
    c = res["curve"]
    assert c[-1]["trainLoss"] < c[0]["trainLoss"]
    assert res["seg"]["iou"][1] > 0.1 or res["seg"]["iou"][2] > 0.1
    assert len(res["records"]) == 3 and res["records"][0]["image"].shape == (48, 48, 3)
    again = S.train_and_eval(samples, spec, epochs=2, lr=3e-3, width=8, batch_size=10, seed=0, val_fraction=0.25, n_inspect=1)
    again2 = S.train_and_eval(samples, spec, epochs=2, lr=3e-3, width=8, batch_size=10, seed=0, val_fraction=0.25, n_inspect=1)
    assert again["curve"] == again2["curve"]  # seeded: repeatable
