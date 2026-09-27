# Picbreeder skull reference

Source: https://github.com/akarshkumar0101/fer

Pinned commit: `67c270c197d8243435d6a9ad036e26c9c5e3712d`.
Copyright and Apache-2.0 license: see the included `LICENSE`.
The source authors are Akarsh Kumar, Jeff Clune, Joel Lehman and Kenneth O. Stanley.

This is the **evolved Picbreeder CPPN**, not `data/sgd_skull`.
The original ZIP contains 23 nodes, including four inputs and one `ink` output.
As in the authors' `process_pb.py`, ink becomes brightness and two zero-valued
hue/saturation nodes are added to make an HSB output. No training is performed.

## Files and derivation

- `original.zip`: unchanged `picbreeder_genomes/skull.zip`.
- `published.png`: unchanged `data/picbreeder_skull/img.png`.
- `model.json`: original graph, source hashes, and the four published controls.
  XML node identities combine branch and ID. Link values are retained from XML;
  evaluation casts weights and coordinates to float32, as in the JAX model.
- `layered_weights.npz`: numerical arrays extracted from the published
  `data/picbreeder_skull/params.pkl`. Runtime and tests read NPZ with
  `allow_pickle=False`; they do not load or execute pickle files.

The source vector has 5,478 float32 parameters. It was extracted using a restricted
unpickler permitting only NumPy array reconstruction and a local replacement for
JAX's array reconstruction. Its matrix order is lexicographic by layer name
(`Dense_0`, `Dense_1`, `Dense_10`, ...). Matrix shapes are 4×22 for layer 0,
22×22 for layers 1–11, and 22×3 for layer 12. The 22 hidden positions contain
15 identity copies, four signed Gaussians, two identities, and one sine.

The original graph's evaluation order and the authors' `layerize_nn` ordering
identify these correspondences. All four are original nonzero connections,
not newly introduced identity-copy weights or empty dense-matrix positions:

| Control | Flat parameter ID | Matrix / position | Original source → target |
| --- | --- | --- | --- |
| Mouth opening | 4371 | Dense_7 [15, 15] | 540_106 → 576_223 |
| Eye winking | 5009 | Dense_9 [0, 15] | 534_2 → 534_5 |
| Eye width | 5097 | Dense_9 [4, 15] | 542_96 → 534_5 |
| Jaw width | 37 | Dense_0 [1, 15] | 534_3 → 576_238 |

The original labels and weights are tested against the archived XML and published
parameter vector. The notebook displays the original graph using compact local
node IDs; these are not the flattened parameter IDs above.

## Verification and limits

An independent NumPy evaluation of the published layer matrices agrees with our
direct graph evaluation for the baseline and all four controls at offsets
−1, −0.5, 0, +0.5, +1. Tests require raw HSB agreement within 3e-5 and rendered
sweep images within one byte level at the tested resolution. Different accumulation
orders prevent assuming bitwise equality.

The supplied repository PNG is a separate comparison: at 256×256 our baseline
differs by about 0.632 byte levels on average, maximum 12/255. The cause has not
been established. Do not describe it as an exact reproduction of that PNG, or
silently change the model to fit it. Both the original source data and this
discrepancy are retained for inspection.

This adapter preserves the reference's signed Gaussian, ordinary sine, radius
scale 1.4 and explicit constant input. It does not alter local breeding, activation
defaults or HSB rendering. The shared inspector's additional bias/clamp/disable
controls are exploratory interventions beyond the four reproduced weight sweeps.
