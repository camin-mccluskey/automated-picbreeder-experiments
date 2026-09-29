# Experiment brief: UFR in automatically evolved CPPNs

## Objective and hypothesis

All automated selection strategies run through `experiments/run_selection.py`;
see the [README](../README.md#automated-selection-experiments) for configuration.
Pixel novelty, online image predictability and offspring-value prediction use the
same breeding loop. Their observers and predictors are separate from the CPPNs
being evolved; observer learning and pixel diversity are not UFR evidence.
The representation study below retains its scientific objective.

Test whether classifier-guided selection in a Picbreeder-style breeding loop can produce compositional pattern-producing networks (CPPNs) with evidence of **unified factored representations (UFR)**, as described by Kumar et al. [1].

- **Factored:** meaningful image properties can be varied independently, with limited unintended changes.
- **Unified:** repeated structure depends on shared computation rather than separate implementations of the same structure.

The hypothesis is that automated selection can produce these properties without human selection during evolution. This is a sufficiency claim about the complete system: CPPN representation, evolutionary operators and selection strategy. The initial classifier-guided condition replaces the choice in our human notebook with maximum ImageNet class confidence. The implementation also supports uniform random selection and ImageNet selection with occasional random exploration. These are available experimental conditions, not comparison results or a frozen protocol. All use the same local breeding loop; this is not a claim that the classifier reproduces human preferences.

**Method decision (2026-09-27):** replace the originally proposed Innovation Engines MAP-Elites archive [2] with the single-parent choice loop below. This is closer to the local interaction loop in Earle et al. [3], but does not implement their shared publication archive, critic agents, or full Picbreeder interaction system. MAP-Elites remains a possible later comparison.

A positive result would not establish that the automated and human selection mechanisms are identical, that familiarity causes UFR, or that the process is indefinitely open-ended. The classifier itself incorporates prior learning from human-labelled data.

## Phase 1: Automated evolution and representation assessment

### 1. Build the evolution system

The implemented runner uses the same candidate-generation and selection opportunities as our human notebook. Its greedy ImageNet condition is:

1. Encode images with CPPNs whose weights and topology can evolve.
2. Evaluate images with one frozen ImageNet-trained convolutional neural network.
3. Start with nine random candidate CPPNs. Evaluate each against all ImageNet classes.
4. Score each candidate by its highest class confidence, regardless of class. Select the highest-scoring candidate; exact ties select the first displayed candidate.
5. Retain the selected parent unchanged in position 1, generate eight independent mutated offspring, and repeat selection from these nine images. Classes can change between decisions; no class-specific archive is maintained.

`RandomSelectionStrategy()` instead selects uniformly from all nine candidates without inference. `ImageNetSelectionStrategy(epsilon=p)` evaluates and scores the same way as the greedy condition, then chooses uniformly with probability p and greedily otherwise. Random draws include the retained parent and may coincide with the greedy choice. Epsilon 0 is greedy; epsilon 1 is always random with ImageNet measurements still recorded. Selection strategies own evaluation, scoring and choice behind one `choose(images, *, rng)` interface. The runner has no separate evaluator or scorer arguments. See the [README](../README.md#automated-selection-experiments) for Python and CLI instructions.

Use the same shared breeding implementation for the notebook and automated runner, with matched initialization, rendering and mutation settings. The runner makes selection decisions only: it does not exercise the notebook's optional backtracking or reset controls. In the greedy condition, exact score ties retain the parent, and a fixed deterministic scoring function cannot select a lower-scoring image. Exploratory choices can. A finite sequence of selections is one run; its length includes the initial random-grid choice. The runner uses a separate, recorded selection RNG seed so exploration draws do not alter mutation draws.

ImageNet selection uses class confidence as a candidate proxy for recognisability/human preference, not a validated measure of naturalness or interestingness. The random condition has no such preference. These implemented conditions do not use a vision-language model (VLM) or an image-reconstruction objective.

Both human and automated experiments now use three mutable CPPN outputs ordered hue, saturation and brightness (HSB/HSV), converted by the shared renderer to RGB using Picbreeder-VLM’s mapping: hue wraps modulo one, saturation clips to [0, 1], and brightness takes its absolute value then clips to [0, 1]. RGB bytes use half-up rounding. Coordinates, mutation rates, mutation strength and the parent-plus-eight-offspring loop are unchanged. This adopts HSB colour semantics without Picbreeder-VLM’s custom reproduction or colour/brightness subnetworks. Earlier experimental rendering conventions are not supported. Activation-function definitions remain those of our NEAT-Python configuration, without Picbreeder-VLM’s extra output transforms.

Reuse existing code where practical. Earle et al. provide a recent Picbreeder implementation [3, §3.1]. Pin the code version, classifier checkpoint, preprocessing, rendering and mutation configuration. Document deviations from the original Innovation Engine rather than describing an adapted implementation as an exact replication.

Save every candidate genome, parent ID, available classifier-score vector, ordered choice set and selected position, including rejected alternatives and repeated parent selections. Save decision checkpoints and sufficient configuration and source code to reproduce images and runs. With S decisions, every condition presents 9S candidates and generates 9 + 8(S - 1) genomes. ImageNet conditions perform 9S image evaluations, including exploratory turns; pure random selection performs none. Human and automated runs use the same version-2 `session.json` format and shared persistence code, including PNGs of every generated candidate. Selection events contain optional evaluation values, names and metadata, plus decision mode and preference scores. Human choices have null evaluation and decision; the pure random selection strategy has null evaluation and scores. Current checkpoints are audit snapshots, not resumable sessions. Existing version-2 HSB sessions remain readable without rewriting them.

### 2. Pilot and freeze the protocol

Use a small pilot to establish computational cost and develop the representation assessment. Before the main runs, fix:

- Selection strategies and epsilon values to compare, number of independent seeds, and decision budget per run; record image-evaluation cost separately.
- Checkpoint intervals.
- Checkpoints and sampling rule for selecting networks for assessment across independent runs.
- Intervention ranges, annotation instructions, thresholds and analysis procedure.

These numerical choices remain unresolved. Match candidate-selection opportunities using decision budgets (or equivalently candidate presentations for this fixed nine-image loop). Report image evaluations and computational cost separately: pure random selection cannot be compared using a positive classifier-evaluation budget. Equal seeds do not yield matched candidate sets after different choices. Separate pilot findings from the main evaluation.

### 3. Assess evolved networks directly

Assess selected networks at predetermined checkpoints and at the end of independent runs. Record repeated selections, duplicate images and unrecognisable images; do not silently exclude them or select only attractive examples. Use published human-evolved CPPNs as reference cases, not as matched experimental controls.

For each assessed network:

- Visualise its graph and neuron activation maps.
- Sweep individual weights to identify and evaluate selective changes to meaningful image attributes.
- Inspect candidate shared computations underlying repeated structure and intervene on them to test their contribution.

Follow Kumar et al.’s analysis format for comparability [1, §3 and Appendix D]. Assess the evolved networks themselves; training another network to reconstruct their images is unnecessary for the primary question.

**The quantitative assessment is not yet fully specified.** During the pilot, define measurable attributes, minimum target changes, tolerated unintended changes and image-preservation criteria. Separate discovery of candidate controls from evaluation on held-out intervention settings. Use blinded annotations or independently validated attribute measurements.

Established disentanglement criteria provide guidance [4, §2; 5, §4 and Appendix C], but adapted weight-intervention tests must not be labelled standard DCI. Selective semantic control alone does not establish shared computation. Do not claim a universal UFR score or treat failure on chosen attributes as proof of FER.

### 4. Record secondary outcomes

At fixed checkpoints, assess recognisability, visual diversity and novelty relative to earlier outputs using a fixed sampling and rating protocol. This addresses whether meaningful variation continues or stagnates within the allotted budget.

For ImageNet conditions, log changes in the selected image's highest-scoring class and inspect lineages. Pure random runs have no classifier measurements unless evaluated separately after the run. This differs from the Innovation Engine's archive-based operational definition of goal switching [2, §5.2]. A changed top class is a classifier-label transition, not evidence of autonomous invention of a new objective.

Classifier confidence, class-label changes and visually different noise are insufficient evidence of creativity or sustained meaningful novelty.

## Phase 2: Human-choice dataset

Treat this as a separate follow-up, not a prerequisite for Phase 1.

Build or adapt a Picbreeder interface using the same CPPN implementation. Record every displayed candidate, its position, the selected candidates, ancestry, branching and resets. Preserve rejected alternatives and decision context, not just successful trajectories.

Compare human decisions with the maximum-class-confidence choice rule applied to the same candidate sets, including the retained parent. Free-running human and automated sessions will diverge after different choices; using the same breeding loop does not by itself provide matched candidate sets for a decision-level comparison. Agreement on matched sets would demonstrate predictive similarity, not identity of mechanism.

## Interpretation and deliverables

The desired positive finding is reproducible evidence of both selective semantic control and shared computation in automatically evolved networks. Report their occurrence across assessed networks and independent runs, including failures and uncertainty.

A negative finding means the tested configuration and budget did not demonstrate these properties. Controlled comparisons between these selection strategies can test the contribution of selection under the shared representation and mutation settings. Their availability alone does not establish that contribution; those comparisons and the representation assessment have not yet been completed.

Deliver reproducible code, frozen configuration, complete logs, assessment materials and a short report. Position the contribution as representation analysis of classifier-guided CPPN evolution. Acknowledge existing automated-creativity results [2] and the recent VLM replication’s preliminary representation analysis [3, Appendix B.1].

## References

1. **Kumar et al. (2025), _Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis_.** §3 and Appendix D: representation analysis; §6.5: open-ended search. [Paper](https://arxiv.org/html/2505.11581v1) · [Code](https://github.com/akarshkumar0101/fer).
2. **Nguyen, Yosinski and Clune (2016), _Understanding Innovation Engines: Automated Creativity and Improved Stochastic Optimization via Deep Learning_.** §2.2: implemented approach; §§4.1–4.2: classifier and evolution; §§5.1–5.2: outputs, multiple objectives and goal switching. [Paper](https://yosinski.com/media/papers/Nguyen__2016__Understanding_Innovation_Engines_Automated_Creativity.pdf).
3. **Earle et al. (2026), _In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models_.** §3.1: implementation; §5: experimental results; Appendix B.1: internal CPPN representations. [Paper](https://arxiv.org/html/2605.23908v1) · [Code](https://github.com/smearle/picbreeder-vlm).
4. **Eastwood and Williams (2018), _A Framework for the Quantitative Evaluation of Disentangled Representations_.** §2: disentanglement, completeness and informativeness (DCI). [Paper](https://www.pure.ed.ac.uk/ws/files/57345826/iclr_final.pdf).
5. **Locatello et al. (2019), _Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations_.** §4 and Appendix C: evaluation metrics and their requirements. [Paper](https://proceedings.mlr.press/v97/locatello19a/locatello19a.pdf) · [Supplement](https://proceedings.mlr.press/v97/locatello19a/locatello19a-supp.pdf).
