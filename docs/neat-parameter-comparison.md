# NEAT parameter comparison

Checked 2026-09-27. This is a source audit, not an experiment measuring the effects of these differences. No evolution settings were changed.

## Assessment

The current implementation belongs to the same CPPN/interactive-evolution family, but is not a parameter-matched reproduction of Picbreeder or Picbreeder-VLM. Several differences concern the mutation algorithm and representation, so editing probabilities alone would not establish equivalence.

The FER/UFR paper analyzes archived human-evolved Picbreeder networks and SGD reconstructions. It does not provide an independent set of NEAT mutation hyperparameters to reproduce. Its Appendix A and released networks are useful representation references.

## Sources and scope

- [Original Picbreeder, Secretan et al. (2011)](https://www.campbellssite.com/papers/secretan_ecj11.pdf): 15-image interface, multiple-parent selection, user-controlled mutation and colour/structure modes.
- [FER/UFR paper, Appendix A](https://arxiv.org/html/2505.11581v1#A1): representation and historical selection description. [Released CPPN implementation](https://github.com/akarshkumar0101/fer/blob/67c270c197d8243435d6a9ad036e26c9c5e3712d/src/cppn.py).
- [Sakana paper, section 3](https://arxiv.org/html/2605.23908v1#S3): reproduction and interface description.
- Sakana repository pinned at `76c2563787c1789f2b810be2f49970ef819d20af`: [runtime configuration, mutation and initialization](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/picbreeder_vlm/core/neat_components.py), [reproduction](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/picbreeder_vlm/core/picbreeder_reproduction.py), [rendering](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/picbreeder_vlm/core/picture2d.py).
- Legacy Java Picbreeder code bundled in that repository: [configuration](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/third-party/webneat/config/default.xml), [mutation-category chooser](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/third-party/webneat/client/evolution/generators/SuperSweetGenerator.java), [weight mutation](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/third-party/webneat/client/evolution/generators/AbstractLinkWeightMutator.java), [node addition](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/third-party/webneat/client/evolution/generators/AddNodes.java).

The recovered Java snapshot is evidence about an original implementation, not proof of the settings used throughout the historical website or along any particular published lineage. Users could change settings. Likewise, current Sakana source is not a verified frozen commit for every published run.

## Effective mutation behaviour

Local values come from `src/automated_picbreeder/cppn.cfg`, `cppn.py`, and installed NEAT-Python 2.0.0. Sakana values below come from its runtime overrides and custom genome, **not** the unmodified `interactive_config_color` file. That file contains many values that do not govern actual mutation.

| Behaviour | Local implementation | Recovered legacy / Sakana implementation |
| --- | --- | --- |
| Mutation categories | At most one structural operation, followed by connection and node attribute mutation on every child | Choose one category: weights 10/21, add connection 6/21, add node 4/21, activation 1/21 |
| Existing-weight perturbation | Each connection gets a 0.70 perturbation probability on every mutation call | Conditional on choosing weight mutation, each eligible connection gets probability 0.20 |
| Effective per-link perturbation probability | 70%, excluding structural changes to that link | About 9.52% in all-channel mode: `(10/21) * 0.20` |
| Gaussian perturbation standard deviation | 0.20 by default | Sakana: `0.01 + 1.99*s`, with default slider `s=0.5`, giving 1.005. Legacy configured range is 0–2; midpoint maps to 1 if applied |
| Weight initialization | Normal, mean 0, standard deviation 1 | Uniform between -3 and 3, with some initial colour links deliberately zero |
| Weight bounds | -10 to 10 | -3 to 3 |
| Add-node category probability | 0.20 | 4/21, approximately 0.1905 |
| Add-connection category probability | 0.20 | 6/21, approximately 0.2857; all-channel mode then makes separate colour, structure and structure-to-colour attempts |
| Delete node / connection | 0.02 / 0.05 per child | No deletion category in the inspected mutation scheme |
| Connection enable mutation | Configured rate 0.01 per connection; NEAT draws a new Boolean, so actual state changes are not guaranteed | No ordinary enable/disable mutation |
| Activation resampling | 0.05 per non-input node on every child | Category probability 1/21, then 0.05 per eligible node; approximately 0.00238 overall |
| Add-node semantics | Disable old edge; insert edges initially weighted 1 and the old weight, then perform attribute mutation | Retain old edge; add a new two-edge path with fresh random weights |

These are attempt/resampling probabilities, not guarantees of a changed image. Structural proposals can fail, replacement activations can equal the previous choice, and clipping can eliminate a numerical change. Channel-restricted mutation changes eligibility. Sakana's add-connection category can attempt more than one edge, unlike the local single structural operation.

For a hypothetical eligible 20-edge parent with topology fixed, the expected number of existing-weight perturbations is 14 locally versus about 1.90 under the category-based scheme. This does **not** imply that local image changes are larger: Sakana's perturbations are larger when selected, and rendering sensitivity differs.

Local mutation strength is supplied by `mutate_genome(..., strength=...)`, which overwrites both weight and bias mutation power. Changing those two config-file powers alone will not change notebook/CLI behaviour. Both interfaces default to 0.2. Disabling topology locally also disables activation and enable-state mutation, while retaining weight and bias mutation.

## Representation and initialization

| Feature | Local implementation | Reference |
| --- | --- | --- |
| Inputs | x, y, Euclidean radius | Legacy/Sakana add constant 1 and multiply radius by sqrt(2); FER reconstruction uses 1.4 |
| Node bias | Every non-input node starts with normal bias, standard deviation 0.5; 30% mutation probability | Sakana node biases fixed at zero; offsets are represented through connections from the constant input |
| Response gain | Fixed at 1 | Matches Sakana |
| Local initial graph | Two hidden nodes; inputs-to-hidden, hidden-to-output and direct input-to-output edges; verified 21 edges | Legacy/Sakana build a one-hidden-node brightness network, then append two colour hidden nodes and brightness-to-hue/saturation connections |
| Colour initialization | All three output activations randomly selected; no dedicated colour/structure organization | Current Sakana code sets hue to sine and saturation to signed sigmoid; colour hidden-to-output weights start at zero |
| HSB conversion | Hue wraps, saturation clips, absolute brightness clips | Matches FER/Sakana conversion |
| Feed-forward / aggregation | Acyclic, sum | Same broad design |

An explicit constant input and unrestricted node biases can express overlapping functions, but their connectivity and mutation opportunities differ. They are not the same evolutionary representation.

Activation names also conceal substantive differences. Ignoring numerical clipping at extreme arguments:

| Name | Local NEAT-Python function | FER/Sakana function |
| --- | --- | --- |
| sine | `sin(5*z)` | `sin(z)` |
| Gaussian | `exp(-5*z*z)` | `2*exp(-z*z)-1` |
| sigmoid | `1/(1+exp(-5*z))` | `2/(1+exp(-z))-1` |
| tanh | `tanh(2.5*z)` | FER: `tanh(z)`; absent from current Sakana default set |

Local choices are sine, Gaussian, tanh, sigmoid, identity and absolute value. Current Sakana runtime choices are sine, signed sigmoid, signed Gaussian, cosine and identity. The recovered Java default XML lists sigmoid, Gaussian and sine; it should not be treated as proof of the function set used for all historical networks. FER Appendix A lists identity, sine, cosine, tanh, signed sigmoid and signed Gaussian.

**Paper/code discrepancy:** Sakana section 3.1 describes brightness initialized with sigmoid, hue/saturation with identity, and a four-function hidden-node set. Its currently published seeding code instead randomizes brightness, sets hue to sine and saturation to signed sigmoid, and includes Gaussian among five activation choices. A reproduction must specify which version it follows.

## Selection protocol and inactive settings

The local loop uses nine candidates and exactly one parent, retained unchanged, plus eight independently mutated children. Original Picbreeder and Sakana show 15 candidates and permit multiple parents; Sakana retains selected parents and supports crossover when enabled. Both references additionally support publication and branching. These are deliberate experimental differences, not necessarily defects.

Speciation thresholds, stagnation, survival fraction and NEAT reproduction elitism in the local config do not control this loop: it uses `DefaultGenome` initialization/mutation directly, bypassing population evolution. Parent retention is implemented explicitly in `BreedingSession`. Candidate count is also explicitly implemented there, so changing `pop_size` alone would not resize the grid.

## Implications

For a close comparison aimed at representational properties, prioritize activation semantics, constant-input/bias representation, initialization, and the mutation-category and add-node rules. These change the structures available and how existing computation is modified. Keeping the chosen single-parent selector is compatible with studying those mechanisms, provided the resulting system is described as a controlled variant rather than a full replication.

The direction of the effect on UFR remains an empirical question. The audit establishes differences; it does not show that any particular setting causes or prevents UFR.
