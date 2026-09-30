# Scientific objective

This project asks whether automated selection in a Picbreeder-style breeding loop
can produce CPPNs with **unified factored representations (UFR)**, as described by
[Kumar et al.](https://arxiv.org/html/2505.11581v1):

- **Factored:** meaningful image properties can be varied independently, with
  limited unintended changes.
- **Unified:** repeated structure depends on shared computation rather than
  separate implementations of the same structure.

This is a sufficiency question about the complete system: CPPN representation,
evolutionary operators and selection strategy. It does not require the automated
selector to reproduce human preferences.

## What is implemented

Human and automated runs use the same single-parent loop: begin with nine random
CPPNs, select one, retain it unchanged and generate eight independently mutated
children. The networks map pixel coordinates to hue, saturation and brightness.
NEAT-Python supplies genome initialization and mutation; the project does not use
its population, speciation or crossover algorithm.

The [seven selection strategies](../README.md#automated-selection-experiments)
cover random choice, pixel novelty, frozen ImageNet confidence, learned masked-pixel
accuracy and forecasts of immediate offspring value. Observer and predictor
networks are separate from the CPPNs being evolved. The strategy guides describe
their exact rules, defaults, warm-up schedules and limitations.

Runs save all generated genomes and images, rejected alternatives, ancestry,
choices, measurements and configuration. [Batch reports](batch-experiments.md)
provide image-change diagnostics, prediction errors, computational costs and an
offline viewer. Human exports additionally preserve backtracking, resets and
repeated selections as distinct events.

[Network inspection](usage.md#interpret-a-saved-network) provides computation
graphs, activation maps, weight sweeps, node clamps and connection removal. The
bundled human-evolved skull provides a reference for these tools, not a matched
experimental control. The local representation and mutation rules differ from
original Picbreeder and Picbreeder-VLM; see the
[parameter comparison](neat-parameter-comparison.md).

## What the results can establish

The tools support exploratory intervention on evolved networks. Quantitative
representation assessment and a frozen comparison protocol are not implemented.
There is currently no validated UFR or interestingness score.

Selective semantic control and shared computation require separate evidence.
Weight sweeps can suggest a control, but assessing it requires defined attributes,
intervention ranges and tolerated unintended changes. Shared influence on two
image regions does not alone establish a shared representation of their structure.
Discovery of candidate controls must be separated from their evaluation.

Classifier confidence is a relative score across ImageNet classes, not validated
naturalness, creativity or human preference. Masked-pixel accuracy can favour flat
or familiar images. Pixel diversity can reward noise. Offspring forecasts concern
the next eight children, not long-term potential. Improvements in these quantities
do not by themselves demonstrate UFR or sustained open-endedness.

Comparisons need matched mutation settings, image sizes, independent seed schedules
and decision budgets, with inference and training costs reported separately.
Equal seeds do not preserve candidate sets after selections diverge. Human
backtracking and parameter adjustments introduce additional differences. Attractive
examples should be reported alongside duplicates, failures and uncertainty.

## References

- [Kumar et al., Questioning Representational Optimism in Deep Learning](https://arxiv.org/html/2505.11581v1): the UFR hypothesis and CPPN representation analysis.
- [Earle et al., In Search of the Ingredients of Open-Endedness](https://arxiv.org/html/2605.23908v1): Picbreeder with vision-language models. Its [implementation](https://github.com/smearle/picbreeder-vlm) informs the local interaction loop and HSB-to-RGB mapping.
- [Nguyen, Yosinski and Clune, Understanding Innovation Engines](https://yosinski.com/media/papers/Nguyen__2016__Understanding_Innovation_Engines_Automated_Creativity.pdf): related classifier-guided image evolution using a different search procedure.
