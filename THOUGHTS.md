# Selection strategy experiments

Working notes from the discussion on 28 September 2026. These are proposed
experiments and open design choices, not a frozen protocol. Implementation follows
[PLAN.md](PLAN.md) and [TODO.md](TODO.md). Pixel novelty and the two predictability
observer initializations and the comparison workflow are implemented. The first
nine-run MPS pilot is complete; see [WRAP_UP.md](WRAP_UP.md) for findings and
[README.md](README.md#run-the-experiment-1-comparison) for commands. The
offspring-value experiment is planned in
[PLAN_EXPERIMENT_2.md](PLAN_EXPERIMENT_2.md), now accepted. Phase 1 review passed. Phase 2 implements online scalar prediction and
selection through the existing strategy interface. The matched six-run MPS
comparison and reporting are now complete; see the Experiment 2 findings in
[README.md](README.md#experiment-2-pilot-findings). See [TODO_EXPERIMENT_2.md](TODO_EXPERIMENT_2.md) for progress.

## Intent

Explore selection strategies inspired by research on interestingness, learning
progress and evolvability, while testing our own ideas in the existing CPPN image
generation system. Start with small experiments that isolate a mechanism before
combining mechanisms or scaling runs.

The two experiments are:

1. Novelty plus learnability: discover unfamiliar images with understandable or
   learnable structure.
2. Combine that criterion with evolvability: prefer parents whose descendants
   offer valuable, meaningfully different possibilities.

These experiments exclude VLM-based selection and discovery-driven goal adaptation.

These experiments complement the representation question in
[the experiment brief](docs/experiment-brief.md). Attractive outputs, improved
observer predictions and diverse offspring do not by themselves establish UFR
inside the evolved CPPNs.

## Experiment 1: novelty plus learnability

**Question:** Can we discover unfamiliar forms while preserving learnable structure,
without favouring noise or settling on simple repetition?

The starting idea is to combine novelty with comprehension:

```text
value_t(image) = alpha * novelty_t(image) + beta * comprehension_t(image)
```

Here, `t` matters: novelty depends on what has been encountered, and comprehension
depends on the observer's current knowledge.

- **Novelty:** the first implementation measures mean squared pixel difference
  from the mean image of the previous generation's nine displayed candidates.
  This measures local change and permits revisiting older images. Learned
  representations and wider history remain possible future experiments.
- **Comprehension:** use an observer trained on previous experience to predict
  visual structure or mutation consequences. The motivating idea is to anticipate
  what selecting a parent could produce. Before those offspring exist, prediction
  reliability must be estimated from prior evidence; their actual prediction error
  becomes available only after breeding.

There are two distinct versions worth comparing:

| Version | What receives a high score? | Interpretation |
| --- | --- | --- |
| Predictability | Accurate predictions before additional training | Unfamiliar overall, but already understandable |
| Learning progress | Improvement on withheld examples after a fixed learning effort | Investigation yields new understanding |

The first is the simpler starting point. It could favour new combinations of
familiar structure. The second is closer to learning-progress and
compression-progress accounts of interestingness. High prediction error alone
does not show that anything can be learned.

Predicting mutation consequences and predicting withheld image regions are
different tasks. The former motivated the original proposal; during planning we
chose withheld-region prediction for the first experiment. Compare identical
ResNet18 observers initialized randomly or from ImageNet, trained online on all
distinct displayed images after scoring each grid. Both start fresh per run;
neither uses historical CPPN training data. Predicting offspring value remains
the separate second experiment.

### Small comparison

Compare novelty alone and novelty plus predictability with each observer
initialization on common
grids first, then in short independent breeding runs. Add a learning-progress
variant once we understand the simpler scores. ImageNet selection is an available
reference condition, not a required component of the new criterion. Existing random
selection is excluded from this batch's comparison.

Inspect whether the comprehension term preserves structure while allowing novelty,
or simply favours flat images and smooth gradients. Novelty and comprehension must
not be opposite transformations of the same prediction-error measurement.

For learning-progress comparisons, candidate-specific training trials should start
from the same observer state and receive equal training budgets. Evaluate progress
on withheld data rather than the examples just fitted. The choice of withheld
examples determines whether we test local interpolation or broader generalisation.

Use the same reporting standard as the existing strategies, rather than saving a
separate collection of notable examples. Run for a fixed number of selection steps
and report the final selected image as the run's most "interesting" output. The
working assumption is that selection improves the score on average from generation
to generation, not that every generation necessarily improves it.

### Choices to resolve

The accepted plan resolves the initial target, novelty reference and training
history above, and specifies pilot defaults for the questions below. Keep these
questions for later refinement rather than treating them as implementation blockers.

- What visual representation makes novelty useful rather than merely sensitive to
  colour changes or noise?
- What does the observer predict, and how is prediction quality measured?
- Does observer training use a shared initial dataset, earlier runs, the current
  run, or a combination? How do we handle the initial lack of experience?
- Does history include every displayed candidate or only selected images? How are
  repeats handled? Compare candidates against a common history before adding the
  current grid.
- How are the two scores scaled? Is a weighted sum appropriate, or should we seek
  novelty only among candidates meeting a comprehension threshold?
- How do we update the persistent observer after scoring, while avoiding forgetting
  and relearning the same patterns as a source of apparent progress?

## Experiment 2: predict offspring value

**Question:** Can a model learn which parents are likely to produce valuable
descendants, and use those predictions to improve selection?

Use the value measure from Experiment 1, but predict offspring value before
breeding. The candidate offspring are unknown at selection time. We do not generate
trial offspring from every candidate to decide which parent to select.

```text
predicted_offspring_value_t(parent) = predictor_t(parent, mutation_settings, context_t)

score_t(parent) = value_t(parent)
                 + gamma * predicted_offspring_value_t(parent)
```

The model directly predicts the expected mean value of the offspring a parent
would produce under the mutation process. It learns this scalar prediction from
the values of realised offspring.

### Online prediction and selection loop

1. Initialize a fresh offspring-value predictor from scratch in each run. This was
   confirmed during Experiment 2 planning; historical pretraining is deferred.
2. Predict offspring value for each of the nine current candidates using the same
   model state. Combine the prediction with current-image value and select a parent.
3. Generate the usual eight offspring from that selected parent, retaining the
   parent unchanged. These are the actual next generation, not selection probes.
4. Evaluate the realised offspring and compare their value with the prediction
   recorded before breeding. Exclude the retained parent from the offspring target.
5. Train the predictor on this new experience, then use the updated model for the
   next selection.

This couples two processes: learning to forecast offspring value and selecting
parents to maximise the resulting selection score. Improving prediction is not
itself proof that selection improves; measure both separately.

Novelty and comprehension change with history and observer learning. Define the
realised training target using the selection-time value function and history,
before incorporating the new offspring into either. During planning we chose to
rank each offspring against the original nine-image comparison, keeping the
selection-time observer, novelty mean image and rank references fixed. Do not
rerank the offspring against one another: their average rank would provide little
information about brood quality. Average the eight fixed-reference child scores.
Historical examples, if used in a later experiment, need
sufficient context or an explicit relabelling policy; old scores are not timeless
labels. The accepted first implementation uses separate models for the comprehension
observer and offspring-value predictor, so predictor training cannot change the
value function used to judge a brood. The predictor receives parent pixels, the
previous-grid mean and 23 scoring-context values; mutation settings stay fixed.

Expected offspring value is still different from optionality: many near-identical,
high-value offspring can have a high mean. A later variant could predict diversity
among acceptable offspring. Start with expected value before adding this second
prediction target or extending beyond the next generation.

### Small comparison

The first comparison, revised during planning, is:

- Current-image value alone.
- Current-image value plus predicted offspring value, with the predictor trained
  from scratch online in that run.

The implemented control also trains and records forecasts, but does not use them for
selection. This permits prediction diagnostics under the baseline trajectory.
The earlier pretrained/frozen/online three-way comparison is deferred alongside
historical pretraining; it is not required for this first batch.

Record prediction error before each update, alongside the value and diversity
actually reached. If historical pretraining is added later, keep evaluation examples
separate, preferably by lineage or run. Online observations cover selected parents
only, so they do not establish prediction accuracy for rejected alternatives.
Exploration or broader historical training data may help address that limitation.

The ordinary breeding decision budget is unchanged. Count predictor inference,
online training and any historical pretraining cost separately. Neither predictor
training nor selection randomness should alter the breeding RNG stream.

## Evaluation without a universal interestingness score

We lack a definitive metric of interestingness, but can evaluate narrower claims:

| Claim | Possible check |
| --- | --- |
| Selection maintains discovery | Novelty over time, coverage and duplicate rates, preferably also in a representation not used for selection |
| The observer learns regularities | Prediction improvement on fresh examples under matched training budgets |
| Offspring value can be forecast | Error against realised offspring value, measured before updating the predictor |
| Parents offer productive variation | Diversity and quality of fresh offspring; later, performance on held-out adaptation tasks |
| Outputs are more interesting to people | Blinded comparisons of equally sized, consistently sampled image sets or lineages |

Inspect failures and whole trajectories, not just attractive final images. Keep
automatic proxy measurements separate from human judgments. Improving the score
used for selection is not sufficient evidence of success.

The CPPN representation question remains a separate assessment: use interventions
to investigate selective semantic control and shared computation, as described in
the experiment brief. A better learned observer does not establish a better CPPN
representation.

## Fit with the current project

Preserve the nine-candidate loop: retain one selected parent unchanged and generate
eight offspring. Hold rendering, mutation settings and decision budgets fixed when
comparing strategies. Match seed sets, while recognising that candidate sets diverge
after strategies make different choices.

Express each experiment as a complete selection strategy owning its measurements,
scoring and choice. Avoid introducing separate scorer/selector configuration.
The current `choose(images, *, rng)` interface can support image-based scoring.
Experiment 2 associates the selected parent with the next chronological grid
inside its strategy; the runner interface is unchanged. The accepted first model
uses images and scoring context, with mutation settings held fixed. It does not
receive genomes: similar images can arise from CPPNs with different mutation
behaviour. Genome-conditioned prediction remains a possible later experiment.

Decision metadata records candidate forecasts, original selected-parent inputs
and scoring references, delayed observed targets, errors and both models' updates.
Existing saved images and seeds support fresh chronological replay. Preserve
rejected alternatives and run artifacts; they do not acquire offspring labels.

The intended sequence is score inspection, short pilot runs, then a defined
comparison protocol. Models, hyperparameters, run lengths and seed counts are still
open. These notes do not prescribe an implementation plan.

## Research informing the experiments

- [Schmidhuber (2010), Formal Theory of Creativity, Fun, and Intrinsic Motivation](https://people.idsia.ch/~juergen/ieeecreative.pdf): motivates compression progress rather than static compressibility.
- [Oudeyer, Kaplan and Hafner (2007), Intrinsic Motivation Systems for Autonomous Mental Development](https://www.pyoudeyer.com/ims.pdf): implemented exploration driven by learning progress.
- [Herrmann and Schmidhuber (2026), Interestingness as an Inductive Heuristic for Future Compression Progress](https://arxiv.org/html/2605.14831v1): investigates predicting further compression progress from past progress; not a ready-made image-interest metric.
- [Pathak, Gandhi and Gupta (2019), Self-Supervised Exploration via Disagreement](https://proceedings.mlr.press/v97/pathak19a.html): a useful contrast, because it rewards predictive disagreement rather than existing predictability.
- [Mengistu, Lehman and Clune (2016), Evolvability Search](https://researcher.itu.dk/en/publications/evolvability-search-directly-selecting-for-evolvability-in-order-/): directly selects for immediate-offspring behavioural diversity.
- [Katona, Franks and Walker (2021), Quality Evolvability ES](https://arxiv.org/abs/2103.10790): combines offspring diversity with performance.

Our experiments would adapt these ideas to CPPN images. In particular, the proposed
offspring-value predictor learns from completed breeding outcomes within each run;
it does not directly evaluate trial offspring at selection time as in the cited
evolvability methods. These would not be exact replications or demonstrations of
the same psychological mechanisms.
