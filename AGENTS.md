# Automated Picbreeder: agent guide

## Purpose and current scope

This is a small research and learning project for evolving compositional pattern-producing networks (CPPNs). The user experiments in stages, inspecting human-selected and automatically selected images before extending the system.

The longer-term question is whether automatically evolved CPPNs develop unified factored representations (UFR): meaningful properties that can be varied independently, and repeated structure that depends on shared computation. The working breeding tools are implemented; the quantitative representation assessment and frozen experimental protocol are not yet complete. High classifier confidence is not evidence of naturalness, creativity or UFR.

Read [docs/experiment-brief.md](docs/experiment-brief.md) for the scientific objective and [README.md](README.md) for usage and implementation details. Some notebook prose still describes the earlier MAP-Elites proposal. The current experiment is the single-parent loop below; do not restore an archive-based method based on those historical notes.

## Current experiment

- Begin with nine random CPPNs. Select one parent, retain it unchanged in the first position, and generate eight independently mutated offspring. Repeat.
- Human selection happens in the notebook. Automated selection chooses the largest score across every candidate and every ImageNet class: `argmax_i max_class score[i, class]`. Classes may change; exact ties favour the first candidate, retaining the parent after initialization.
- Both paths use the same breeding and rendering code. When comparing selectors, match seeds, image sizes, mutation settings and decision budgets. Human backtracking, resets or parameter adjustments introduce additional differences.
- Use NEAT-Python's genomes and mutation operators, not its full population/speciation/crossover algorithm. There is currently no crossover or MAP-Elites archive.
- The automated runner rescores all nine candidates, including the retained parent. With `S` decisions it evaluates `9*S` images and generates `9 + 8*(S-1)` genomes. Keep those counts distinct.
- The frozen classifier is Torchvision ResNet18 with explicit `IMAGENET1K_V1` weights, CPU by default. Its softmax scores are relative across 1,000 classes. Official preprocessing includes a centre crop, so the classifier sees a cropped version of the displayed image.

## Code structure

Paths below are relative to the repository root. Core modules live in `src/automated_picbreeder/`.

| Path | Responsibility |
| --- | --- |
| `cppn.py`, `cppn.cfg` | NEAT configuration, genome initialization/mutation, coordinate inputs, rendering and genome serialization |
| `breeding.py` | `BreedingSession`: shared candidates, selection, ancestry, mutation, backtracking and reset history |
| `notebook.py` | `CPPNPlayground`: ipywidgets interface around the shared session |
| `evaluation.py` | Generic `ImageEvaluator` protocol and image-by-measurement `Evaluation` |
| `imagenet.py` | Optional frozen Torchvision classifier adapter and preprocessing provenance |
| `selection.py` | Replaceable selection rules; currently `MaximumClassConfidence` |
| `experiment.py` | Automated loop, progress reporting, contact sheets and checkpoint scheduling |
| `persistence.py` | `SessionWriter`: common assembly and saving of run data for both interfaces |
| `experiments/imagenet_selection.py` | Runnable automated experiment CLI; experiment entry points belong in `experiments/` |
| `notebooks/01_cppn_selection.ipynb` | Human breeding, genome inspection and weight sweeps |
| `notebooks/02_image_evaluation.ipynb` | Classifier inspection and evaluator substitution |
| `tests/` | Rendering, mutation, UI callbacks, evaluator contracts, exports and human/automated parity |

## Representation and colour

Inputs are `(x, y, radius)`; outputs are mutable hue, saturation and brightness (HSB/HSV), not RGB directly. The only supported mapping is `[CPPNRendering] output_mapping = picbreeder_hsb_v1` in `cppn.cfg`:

```text
H = (h + 1) % 1
S = clip(s, 0, 1)
B = clip(abs(b), 0, 1)
RGB = colorsys.hsv_to_rgb(H, S, B)
byte = int(255 * clip(channel, 0, 1) + 0.5)
```

This matches Picbreeder-VLM's output-to-pixel mapping. Our activation-function definitions and reproduction rules differ from that repository. Output nodes are not forced to sigmoid. Do not silently change the mapping or add legacy grayscale/sigmoid-HSB compatibility; the user explicitly prefers one simple current implementation. `CPPNConfig.save()` preserves the rendering setting that plain NEAT configuration saving would omit.

## Shared interfaces and records

Keep candidate generation, evaluation, selection and persistence separate. New notions of novelty or interestingness should use the evaluator/selector interfaces rather than introduce classifier-specific breeding logic. Evaluation rows follow candidate order; columns have stable positional identities because display names may repeat.

Both interfaces export the same version-2 `session.json` through `SessionWriter`, with all genomes, ancestry, events, configuration, rendering metadata and every generated PNG in `images/`. Include rejected alternatives and discarded branches. Each selection event has optional evaluation values/names/metadata; human choices use `evaluation: null`. Scores belong to decisions, since repeated evaluations of a parent may differ.

Automated runs additionally save grids and checkpoints. These are audit snapshots, not resumable evolution sessions. Use new output directories. `runs/`, `.cache/` and `.venv/` are local ignored artifacts; do not commit generated runs or model weights.

## Development and verification

Python 3.13+, managed with `uv`; dependencies are locked in `uv.lock`, with NEAT-Python pinned to 2.0.0. From the repository root:

```sh
uv sync --locked --extra imagenet
uv run jupyter lab notebooks/01_cppn_selection.ipynb
uv run --extra imagenet python experiments/imagenet_selection.py --steps 3 --seed 7
uv run --extra imagenet pytest
```

The ImageNet extra is optional for human-only work. Classifier unit tests use controlled models and do not download weights; a real classifier smoke run may download the checkpoint into `.cache/imagenet`.

For changes affecting breeding, rendering or saving, check deterministic replay between human and automated paths, unchanged parents, output mapping, complete records and saved-image reproduction. Preserve the user's notebook seeds and exploratory settings unless the requested change requires otherwise. Check actual notebook/CLI settings rather than assuming they match.

The same seed and action sequence reproduce candidates in a fixed environment. Back restores a grid but does not rewind the RNG; reset draws new roots from the continuing stream. RNG isolation is designed for single-threaded use. Restart notebook kernels after changing imported modules; rerunning imports alone does not refresh existing playground instances.

## Working with the user

Prefer direct explanations, inspectable changes and small experiments. Prioritize accuracy over reassuring claims. Keep implementation simple; avoid speculative frameworks and compatibility layers. Explain scientific consequences when changing representation or mutation. Distinguish implemented behaviour, proposed experiments and empirical findings. Preserve useful rough edges and user experiments rather than tidying unrelated work.
