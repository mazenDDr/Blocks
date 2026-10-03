"""Display names and one-line purposes for the registry listing (the library). Facts only."""
from __future__ import annotations

META: dict[str, tuple[str, str, str]] = {
    # type: (display name, category, purpose)
    "core.tensor_input": ("Tensor input", "Core", "Entry point of the graph: a tensor of the declared shape and dtype (N is the batch size)."),
    "pytorch.nn.conv2d": ("Conv2D", "Layers", "Slides learned filters over the input and sums over input channels: one feature map per filter."),
    "pytorch.nn.relu": ("ReLU", "Layers", "Elementwise max(0, x)."),
    "pytorch.nn.max_pool2d": ("MaxPool2D", "Layers", "Keeps the maximum of each window; shrinks the spatial size."),
    "pytorch.nn.adaptive_avg_pool2d": ("AdaptiveAvgPool2D", "Layers", "Averages each channel down to a fixed output size (1x1 = global average pooling)."),
    "pytorch.nn.flatten": ("Flatten", "Layers", "Merges dimensions into one, keeping the batch axis."),
    "pytorch.nn.linear": ("Linear", "Layers", "Fully connected layer: y = xW^T + b."),
    "pytorch.loss.cross_entropy": ("CrossEntropy", "Loss", "Classification loss on logits and integer class targets."),
    "core.sub": ("Subtract", "Tensor ops", "Elementwise a - b (equal shapes)."),
    "core.add": ("Add", "Tensor ops", "Elementwise a + b (equal shapes)."),
    "core.square": ("Square", "Tensor ops", "Elementwise x^2."),
    "core.sum": ("Sum", "Tensor ops", "Sum over chosen dims."),
    "core.mean": ("Mean", "Tensor ops", "Mean over chosen dims with an explicit divisor rule."),
    "core.scalar_mul": ("Scalar multiply", "Tensor ops", "Multiply every element by a constant."),
}
