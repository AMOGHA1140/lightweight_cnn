# Secondary pipelines

These exist for comparison and are not the project's research focus.

## `faster_rcnn/` — two-stage axis-aligned detector (reference only)

An experimental two-stage Faster R-CNN that operates on **horizontal** boxes
`[x1, y1, x2, y2]`. It is incomplete and kept only for reference:

- it is axis-aligned, not oriented;
- the ROI head trains on random features rather than RoIs pooled from the feature
  maps, so the second stage does not learn meaningfully;
- `forward_test` returns raw top-k RPN outputs without decoding or NMS;
- its "multi-scale" FPN input is faked by pooling a single feature map at three
  resolutions.

It also expects `backbone.forward_features` to return a single tensor, whereas the
current backbone returns three feature maps — so it needs adapting (e.g. use the
deepest map) before it can run.

Modules: `model.py` (FPN / RPNHead / ROIHead / AnchorGenerator / FasterRCNN),
`dataset.py` (OpenCV load + letterbox to 800², HBB labels), `metrics.py`
(`torchvision.ops.box_iou`, 11-point VOC AP). Run: `python -m faster_rcnn.train`.

## `yolo_compare/` — Ultralytics YOLO benchmark

Fine-tunes off-the-shelf Ultralytics YOLO models on DOTA converted to YOLO format
(enclosing horizontal boxes). It is independent of the custom backbone and produces
comparison numbers only.

- `dataset_processor.py` (`DOTADatasetProcessor`): converts DOTA 8-point labels →
  normalised YOLO `<cls> <xc> <yc> <w> <h>`, copies images into
  `images/` + `labels/`, and writes `dataset.yaml`.
- `benchmark.py` (`YOLOModelTrainer`): trains/evaluates the models listed in
  `self.models` (default `yolov8s`) and writes a CSV plus comparison plots.

Run: `python -m yolo_compare.benchmark`.
