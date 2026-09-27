# Automated Picbreeder

An exploratory notebook for learning how CPPNs generate images and how NEAT-Python
mutates their weights and topology. This first prototype uses human selection.

The second notebook adds a frozen ImageNet classifier through a replaceable image
evaluation interface. The automated experiment now replaces the human choice in
the same breeding loop with maximum class confidence. MAP-Elites was the original
proposal; it is deferred as a possible comparison.

## Start

```sh
uv sync --locked
uv run jupyter lab notebooks/01_cppn_selection.ipynb
```

Run the notebook cells in order. Select an image with the button underneath it,
then click **Mutate selected**. The selected parent stays in position 1 and eight
offspring appear alongside it. **Back** returns to an earlier grid so you can
branch; **New random images** creates new roots without discarding session history.
The kernel must remain running for the buttons to work.

If using VS Code, choose this project's `.venv/bin/python` notebook kernel and
enable the Python/Jupyter extensions. If widgets display only as text, try the
JupyterLab command above; the required widget extensions are included in the lockfile.

**Save session** writes all candidate genomes (including unselected alternatives),
parent IDs, ordered displays and decisions, the configuration, package versions,
and PNGs of every generated image to `runs/`, using the shared session format
described below. The notebook shows how to render an
exported genome again. Exports are inspectable records, not resumable evolution
checkpoints or a finished human-choice data collection protocol. Sessions remain
in memory until saved; save before restarting the kernel.

## Small experiments

1. Choose a pattern and generate several rounds. Which image properties persist?
2. Turn off **Allow structure / activation changes**. Compare mutation σ = 0.05
   with σ = 0.5. This changes the standard deviation of weight and bias perturbations;
   it does not directly set image difference.
3. Turn structure changes back on and inspect hidden-node and enabled-connection
   counts. A structural mutation is probabilistic, so counts need not change each time.
4. Use the notebook's single-weight sweep to separate one intervention from the
   many simultaneous changes in an offspring.

These are observations about rendering and mutation, not evidence of semantic
factorisation or shared computation.

## Image evaluation: step 2

```sh
uv sync --locked --extra imagenet
uv run --extra imagenet jupyter lab notebooks/02_image_evaluation.ipynb
```

The `imagenet` extra keeps PyTorch optional for the original playground and for
non-classifier experiments. Include `--extra imagenet` in `uv run` commands that
need it, so uv retains these optional dependencies in the environment.

The notebook renders nine CPPNs, shows the classifier's top labels, inspects its
actual input crop, and substitutes a simple contrast evaluator on the same images.
Contrast is an interface example, not a proposed measure of interestingness.

```python
from automated_picbreeder.evaluation import ImageEvaluator
from automated_picbreeder.imagenet import ImageNetEvaluator

evaluator: ImageEvaluator = ImageNetEvaluator(cache_dir=".cache/imagenet")
result = evaluator.evaluate(images)  # list of uint8 grayscale or RGB arrays
result.values                      # shape (number of images, number of measurements)
result.names                       # names in column order
result.metadata                    # checkpoint, preprocessing, device and versions
```

**Evaluation and selection are separate.** An evaluator supplies measurements;
it does not mutate genomes, choose parents or update an archive. An alternative
only needs an `evaluate(images) -> Evaluation` method. Measurements can be
negative, scalar or multi-column; they need not be class probabilities. Row order
matches input order, including duplicates. Column indices identify measurements;
names are display labels and need not be unique. Policies decide whether higher
or lower values are preferable and how to combine or retain measurements.

For future novelty experiments, an evaluator can receive an explicit reference
set, with its identity recorded in result metadata. Updating that reference set
belongs to the search procedure. For descriptor-based quality diversity, the
policy will also need to specify which measurements define niches and which
measure quality. The current interface does not assume every search algorithm
works like the classifier archive, or force everything into one fitness score.

The first adapter uses **ResNet-18 / IMAGENET1K_V1**, with all parameters frozen,
evaluation mode and gradient-free inference. It returns all 1,000 softmax class
scores, not just a top label. The first load downloads ~45 MB of official weights;
later loads use the cache. Package versions are locked in `uv.lock`; result
metadata records the concrete weight enum, checkpoint SHA-256 and preprocessing.
`DEFAULT` weight aliases are rejected to avoid silently changing checkpoints.
CPU is the default; `device="mps"` or `device="cuda"` explicitly opts into an
available accelerator. Changing model, device, library version or batching can
change numerical results; fix these before main runs.

CPPNs render as RGB from HSB outputs. The classifier adapter also accepts
grayscale arrays and replicates them across RGB channels. The checkpoint's official transform
resizes the shorter side to 256, centre-crops to 224, rescales pixels and applies
ImageNet channel normalization. **This crops the image edges**; the notebook shows
what is retained. No per-image min/max normalization is applied. Inputs must be
uint8 H×W or H×W×3 arrays; float ranges and other channel layouts are rejected.

ImageNet is the training dataset/category set; ResNet-18 is the classifier. Its
scores are not validated naturalness, novelty or interestingness measures, and
abstract CPPNs are outside its ordinary natural-image setting. This is an
exploratory choice, not an exact Innovation Engine replication.

## Automated selection experiment

```sh
uv run --extra imagenet python experiments/imagenet_selection.py --steps 100 --seed 7
```

This is the human notebook's local loop with the choice automated:

1. Generate nine random CPPNs and select the image with the highest confidence
   for **any** ImageNet class.
2. Keep that parent in position 1 and generate eight independently mutated children.
3. Score all nine, select one, and repeat. The winning class can change freely.

The notebook and runner both use `BreedingSession`; they share initialization,
mutation, parent retention and candidate ordering. The runner does not automatically
reset or backtrack. Exact ties pick the first position, retaining the parent after
initialization. A human can accept a lower-confidence candidate; this selector
cannot under a fixed deterministic scoring function. We match the available local
choices, not human decision-making or the full historical Picbreeder website.

`--steps` includes the initial choice. A 100-step run evaluates **900 images** and
creates **801 genomes**. Parents are rescored in each grid; generated duplicates
are not filtered. Other flags include `--size`, `--mutation-strength`,
`--no-topology`, `--device`, `--checkpoint-every` and `--output`.

The default output is a new `runs/imagenet-selection-<timestamp>-seed<seed>/`:

Both human saves and automated runs use `SessionWriter` and the same version-2
format, based on the original human session export:

- `session.json`: seed, image size, output encoding (`HSB`), output mapping (`picbreeder_hsb_v1`),
  image mode (`RGB`), rendering
  convention, versions, embedded CPPN
  configuration, all genomes with parent IDs and image paths/hashes, the event
  history, current display/selection, run metadata and an optional summary.
- `images/<genome-id>.png`: every generated image, including rejected candidates
  and discarded branches. IDs are padded to eight digits.
- `cppn.cfg`, `source/`: mutation configuration and a hashed snapshot of the
  implementation and lockfile.
- `selected.png`: a convenience copy of the current selection, if there is one.

Every `select` event has `position`, `genome`, ordered `displayed` IDs and an
`evaluation` field. Human choices have `evaluation: null`. Automated choices
store `evaluation.values` (rows follow `displayed`, columns follow `names`),
`evaluation.names` and `evaluation.metadata`. Scores belong to each decision,
so repeated evaluations of the same parent remain distinct. Measurement names
may repeat; column position identifies the measurement. Other event types record
mutation settings, resets and backtracking as before.

Human exports have null `metadata.settings` and `summary`; their actual mutation
settings are recorded in the evolve events. Automated runs store their configured
settings and progress summary in those fields. The metadata structure is assembled
by the same writer for both modes.

Automated runs additionally write `grids/` for inspection and `checkpoints/` for
periodic snapshots of the **same session schema**. Paths within both session and
checkpoint documents are relative to the run directory. `session.json` is updated
before evaluation and after each selection; a failed evaluation leaves the current
candidate set and earlier decisions available, with a null summary.

Older saved runs are left untouched. Human version-1 exports lack the new image,
evaluation and metadata fields; earlier automated runs used a different layout.
New exports identify the shared format with `format_version: 2`. Colour exports
add `output_encoding` and `image_mode`; earlier exports may omit these fields.
The renderer supports the current three-output mapping only; old experimental
rendering conventions are not supported.

Existing output directories are never overwritten. Checkpoints support inspection;
there is currently no resume command. Repeating the seed, settings and selector in
the same environment reproduces the trajectory, subject to the classifier's
numerical determinism. The generated source snapshots allow implementation changes
to be distinguished from changes to the selection rule.

For a different experiment, supply another `ImageEvaluator` and `Selector` to
`run_experiment`; the selector returns a zero-based position in the current grid.
`MaximumClassConfidence` is the first concrete rule. This experiment does not
maintain one champion per class or implement MAP-Elites.

The approach resembles the **inner** interaction loop in
[Sakana's AI Picbreeder](https://pub.sakana.ai/picbreeder-vlm/). Their full system also
uses VLMs, shared publication archives, branching and critic agents. Those mechanisms
are not part of this local single-parent experiment. Class confidence is a proposed
proxy for recognisability/human preference, not an established naturalness metric.

## Implementation choices

`cppn.cfg` explicitly declares `[CPPNRendering] output_mapping = picbreeder_hsb_v1`.
This is the only supported mapping. Configuration loading and rendering reject
unknown mappings; exported configurations and `session.json` preserve the setting.


- Three inputs per pixel: x and y in [-1, 1], plus Euclidean radius in [0, √2].
  y increases down the image; neural biases supply constant offsets.
- Three mutable outputs, ordered hue, saturation and brightness (HSB/HSV).
  Following Picbreeder-VLM, H = (h + 1) % 1, S = clip(s, 0, 1), and
  B = clip(abs(b), 0, 1). Python colorsys converts HSV to RGB; channels are
  rounded to bytes with int(255 * channel + 0.5). Output and hidden activations
  can be sine, Gaussian, tanh, sigmoid, identity or absolute value. NEAT-Python's built-in
  scaling applies (e.g. its sine is sin(clip(5z, -60, 60))).
- Two initial hidden nodes; initially fully connected with direct input-output links.
- Images are evaluated at 96 × 96 by the native `neat.nn.FeedForwardNetwork`.
  No per-image normalization or image filtering. Flat images and duplicate outputs
  are retained. The UI displays the images at 160 × 160.
- Each step clones one parent and applies NEAT-Python mutation. No crossover,
  speciation, or fitness-based population selection: this is **not full NEAT**.
- Disabling structure/activation changes freezes connections' enabled states,
  graph structure and activation functions, leaving weight/bias perturbations.
- Seeds isolate NEAT's global random state for a single-threaded notebook session.
  The same seed and action sequence reproduce the candidates with locked dependencies.
  Back restores the earlier choices, but does not rewind randomness: the next click
  produces a fresh branch. Mutation σ = 0 only guarantees unchanged images when
  structure/activation changes are also disabled.

The existing `neat-python==2.0.0` dependency supplies the genome and mutation
implementation. This is a small learning prototype, not a replication of Picbreeder
or Innovation Engines. Those systems' rendering, initialization and mutation
choices differ from our implementation. We match Picbreeder-VLM’s mapping from
HSB network outputs to RGB pixels, while retaining our activation-function
definitions and mutation/reproduction rules. We do not adopt its extra output
transforms, custom mutation operators or colour/brightness subnetworks.

## Experimental sequence

| Step | Build and learn | What to establish before proceeding |
| --- | --- | --- |
| 1. CPPN playground (current) | Render, select, mutate, inspect genomes and sweep a weight | Understand coordinate inputs, activations, weights and topology |
| 2. Automated search (current) | Frozen ImageNet CNN; same nine-image choice loop as the human notebook; select maximum confidence over any class | Check shared breeding behavior, full choice-set records, costs and selection trajectories |
| 3. Pilot assessment | Graphs, activation maps, selective attribute interventions and tests of shared computation | Define measurable attributes, preservation criteria, thresholds and held-out intervention settings |
| 4. Frozen experiment | Independent seeds and fixed evaluated-image budgets, checkpoints and network-sampling rules | Measure both selective semantic control and shared computation; include duplicates, failures and uncertainty |
| 5. Human-choice follow-up | Same CPPN implementation and matched candidate sets; record position, alternatives, ancestry, branching and resets | Specify a classifier choice rule; predictive agreement does not establish the same mechanism |

The primary hypothesis concerns whether the complete automated system can produce
UFR evidence. Class confidence is not a naturalness measure. Changes in the selected
image's top label are classifier-label transitions. The prototype's weight sweep
is exploratory and is not a validated UFR metric or standard DCI.

## Files and checks

- `notebooks/01_cppn_selection.ipynb`: guided, runnable first experiment.
- `notebooks/02_image_evaluation.ipynb`: classifier scores and evaluator replacement.
- `src/automated_picbreeder/cppn.py`: native NEAT genomes, mutation, rendering and JSON conversion.
- `src/automated_picbreeder/cppn.cfg`: explicit starting configuration.
- `src/automated_picbreeder/notebook.py`: human selection interface.
- `src/automated_picbreeder/persistence.py`: shared session metadata, schema and file writer.
- `src/automated_picbreeder/evaluation.py`: generic batch result and evaluator protocol.
- `src/automated_picbreeder/imagenet.py`: optional frozen Torchvision adapter.
- `src/automated_picbreeder/breeding.py`: shared human/automated breeding state.
- `src/automated_picbreeder/selection.py`: replaceable choice rules.
- `experiments/imagenet_selection.py`: runnable automated experiment.

```sh
uv run pytest
uv run --extra imagenet pytest  # also run offline classifier adapter checks
```

Classifier unit tests use controlled logits and the real preprocessing transform;
they never download weights. To check the downloaded model itself, run notebook 02.

## References

- [Experiment brief](docs/experiment-brief.md), supplied by the project owner.
- [NEAT-Python configuration](https://neat-python.readthedocs.io/en/latest/config_file.html).
  The installed 2.0.0 source is authoritative for this prototype; online docs may
  describe newer options.
- [ipywidgets button and callback API](https://ipywidgets.readthedocs.io/en/latest/reference/ipywidgets.html).
- [Torchvision ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Torchvision model and checkpoint API](https://docs.pytorch.org/vision/stable/models.html).
- [Earle et al. Picbreeder implementation](https://github.com/smearle/picbreeder-vlm),
  a reference to review before choosing the automated experiment's implementation.
