# Observer resolution and measured cost

The reconstruction observer uses a fixed **96×96 input with sixteen 24×24
masks**. This retains the complete image at the default rendering resolution;
other sizes are bilinearly resized to 96×96 without cropping. The measurements
below describe compute and memory costs, not selection quality.

## Which model is affected?

| Component | Current input handling | Consequence of 96×96 |
| --- | --- | --- |
| Frozen ImageNet classifier | Applies its checkpoint's image transform. Default ResNet18 resizes to 256, then centre-crops to 224. | Already accepts a 96×96 source image. It does not see the complete source frame after cropping. |
| Scratch reconstruction observer | Resizes each image to 96×96; predicts hidden RGB pixels using a modified ResNet18. | Preprocessing, masks and output head use the same resolution. No checkpoint constraint. |
| ImageNet-initialized reconstruction observer | Same architecture and reconstruction objective; loads only pretrained backbone weights, with a new head and fourth mask channel. All parameters train. | Backbone weights have compatible shapes at 96×96. The new reconstruction head remains randomly initialized. Compatibility does not establish good reconstruction or useful selection. |
| Offspring scalar predictor | Separate model taking full-resolution parent/reference pixels and scoring context. | No need to resize its inputs to match the reconstruction observer. Its inputs already retain source resolution. |

The classifier transform is documented in the
[official ResNet18 checkpoint documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
The [ResNet implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html)
uses adaptive average pooling before its final linear layer: a 32×32 or 96×96
input still produces a 512-dimensional backbone vector. Our local installed
implementation has the same structure. Input spatial size does not determine
checkpoint convolution-weight shapes. Pretrained classification accuracy under
the official transform is not evidence of reconstruction accuracy under either
of our resolutions.

In [image_predictability.py](../src/automated_picbreeder/image_predictability.py),
the fixed resolution determines preprocessing, visibility tensors, tile coordinates,
head width, output reshaping and reconstruction assembly. The reconstruction head
is `512 → 27648`, producing three channels per pixel.

## Mask size changes the question and the cost

| Configuration | Hidden region per prediction | Fraction hidden | Predictions per image | Predictions for nine images |
| --- | --- | --- | --- | --- |
| Benchmark: 32×32, 8×8 tiles | 64 pixels | 1/16 | 16 | 144 |
| Implemented: 96×96, 24×24 tiles | 576 pixels | 1/16 | 16 | 144 |
| Benchmark: 96×96, 8×8 tiles | 64 pixels | 1/144 | 144 | 1,296 |

Sixteen 24×24 masks preserve the hidden fraction and the number of inference
passes. The observer sees the detail previously removed by downsampling, but
must reconstruct a larger, more detailed region. Keeping 8×8 masks changes the
task to filling a much smaller fraction of the image. It also needs nine times
as many masked examples for exhaustive scoring. Equal numbers of training
updates do not give equal per-tile coverage between those two designs.

Full-resolution targets remove the specific loss of detail caused by resizing
96×96 to 32×32. They do not remove the backbone's own downsampling or pooling,
the preference for easily predicted images, or the advantage of previously
trained-on parents. Better resolution is not evidence of UFR or creativity.

## Measured cost

Measured on 30 September 2026 using macOS 26.6.2, arm64, PyTorch 2.14.0 and
Torchvision 0.29.0. CPU only, one Torch thread, batch size 16, five timed repeats
after one warm-up per operation, fresh process per configuration. The exact CPU
model was unavailable in this environment. MPS was built into Torch but unavailable
to the process, so there are no measured GPU results.

The one-off benchmark used the modified ResNet18, synthetic random RGB images,
masking, hidden-pixel loss and real Adam updates. It downloaded no checkpoint.
The one-off benchmark script is not included; these are recorded measurements,
not a benchmark suite shipped with the project. Both initialization arms
have the same architecture; their selection quality is not tested here.

| Measurement | 32×32 / 8×8 | 96×96 / 24×24 | 96×96 / 8×8 |
| --- | ---: | ---: | ---: |
| Median scoring time, nine images | 1.347 s | 1.961 s | 18.030 s |
| Median time, one training update | 0.207 s | 0.394 s | 0.400 s |
| Total trainable parameters | 12,755,584 | 25,363,072 | 25,363,072 |
| Reconstruction-head parameters | 1,575,936 | 14,183,424 | 14,183,424 |
| Weights + gradients + two Adam moments, estimate | 194.6 MiB | 387.0 MiB | 387.0 MiB |
| Whole-process peak resident memory, measured | 580.6 MiB | 1,003.1 MiB | 778.7 MiB |
| Prepared float32 replay image | 12 KiB | 108 KiB | 108 KiB |

The 16-mask 96×96 variant took **1.46× the scoring time and 1.90× the update
time** in this probe. These are measurements on this environment, not scaling
guarantees. The head grows ninefold; the whole model approximately doubles.
Adam's memory estimate is four float32 arrays per parameter. It excludes model
buffers, activations, temporary tensors and allocator overhead, which contribute
to the larger measured process memory. These are single-process peak observations,
not repeated memory estimates: the lower 96×96/8×8 RSS does not establish a memory
advantage over 96×96/24×24. Their model sizes and bounded batch dimensions match.

The 144-mask variant took **13.39× the baseline scoring time**, and **9.19× the
16-mask 96×96 scoring time**. Training-update time stays similar between the two
96×96 designs because both sample sixteen masked examples per update. Exhaustive
scoring uses batches of sixteen in the benchmark; evaluating all 144 masks at
once would have a different memory profile.

At the default twenty observer updates per decision, adding the measured scoring
median to twenty update medians gives approximately **5.49 s at 32×32 versus
9.84 s at 96×96 with sixteen masks**; the 144-mask version gives approximately
26.02 s. This is arithmetic from separate microbenchmarks, not measured
end-to-end decision latency. Patch-based offspring selection also scores eight
children under a saved observer snapshot; that work is additional.

The probe excludes image preparation, reconstruction mosaics, model hashing,
snapshot copies, scalar-predictor work, breeding, rendering and persistence.
Snapshots copy model weights/buffers, and the unbounded distinct-image observer
replay uses nine times as much image storage at 96×96. Ten thousand prepared
images alone occupy approximately 117 MiB versus 1,055 MiB, before Python/container
overhead. Whole-run memory must therefore be measured separately.

## Scope

Only the 96×96, sixteen-mask configuration is implemented. Observer provenance
records size, tile size, hidden fraction and mask count, including in frozen
snapshots. Tests cover mask coverage, hidden-pixel loss, matching-resolution pixel
preservation, scoring and training, matched head initialization, seeded replay
and independent snapshots.

The objective is hidden-pixel reconstruction error. Full-resolution targets do
not resolve its preference for simplicity and familiarity. These microbenchmarks
exclude complete-run costs and do not establish improved selection quality.
