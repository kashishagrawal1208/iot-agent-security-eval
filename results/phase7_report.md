# Phase 7: Evaluation Results

## Important caveat, read first

This report was initially built on a 60-flow clean sample and a 20-case
adversarial corpus. Both were subsequently scaled up (clean: 60 -> 150
flows; adversarial corpus: 20 -> 40 cases) to test whether the original
findings held up under more data -- they did, and in most cases became
stronger. All numbers below are from the scaled-up (n=150 clean, 40-case
adversarial) real-data run. ASR denominators are still modest in some
categories (as few as 1-3 cases) -- treat single-category ASR percentages
as directional, not precise, and prefer the shared-category (direct
injection + context flooding) comparison in Section 3, which has larger
denominators.

## 1. Clean-data performance (n=150 real flows per agent, 75 ATTACK / 75 BENIGN)

| Agent       | Accuracy | Precision (ATTACK) | Recall (ATTACK) | F1 (ATTACK) | FNR    | FPR   |
|-------------|----------|---------------------|------------------|-------------|--------|-------|
| single_shot | 68.7%    | 83.3%               | 46.7%            | 59.8%       | 53.3%  | 9.3%  |
| react       | 62.7%    | 95.2%                | 26.7%            | 41.7%       | 73.3%  | 1.3%  |

See `results/plots/clean_metrics.png` (generated on the earlier n=60 run;
regenerate with `python -m src.evaluation.make_plots` for the n=150
version before final submission).

**Key finding, now reinforced by a larger sample:** single_shot continues
to outperform react on accuracy, recall, and F1. The gap is larger at
n=150 than it was at n=60. A paired McNemar's test on the full n=150
sample gives p=0.137 (statistic=2.21) -- still above the conventional
0.05 significance threshold, but substantially closer to it than the
n=60 result (p=0.75). This is read as: the accuracy gap between the two
agents is a real, consistent trend across both sample sizes, but has not
yet reached the standard threshold for statistical significance at this
sample size.

React's defining characteristic is now unambiguous: very high precision
(95%) paired with low recall (27%) -- it almost never raises a false
alarm, but it misses roughly three out of every four real attacks.
single_shot trades some of that precision for meaningfully better
recall.

## 2. Adversarial robustness (40-case corpus, real attack flows, no defense)

| Agent       | Category            | Cases correct when clean | ASR  |
|-------------|----------------------|---------------------------|------|
| single_shot | direct_injection      | 9                         | 100% |
| single_shot | context_flooding      | 6                         | 50%  |
| react       | direct_injection      | 3                         | 100% |
| react       | context_flooding      | 2                         | 0%   |
| react       | tool_poisoning        | 1                         | 100% |
| react       | compound_poisoning    | 1                         | 100% |

See `results/tables/adversarial_results.csv` and
`results/tables/final_table2_adversarial.csv` for full detail, including
Wilson confidence intervals and accuracy-drop figures.

Direct injection remains close to universally successful against both
agents when no defense is active. Tool-poisoning and compound-poisoning
denominators are still small (n=1 each) even after scaling the corpus up
to 40 cases, because few of the sampled real attack flows were ones
react classified correctly to begin with in those categories -- this
reflects react's low baseline recall (Section 1) propagating into a
small ASR denominator, not a flaw in the corpus itself.

## 3. Defenses (evidence_first, verification_gate), shared categories only

| Agent       | Defense            | Cases correct when clean | ASR  | Reduction vs. none |
|-------------|---------------------|---------------------------|------|---------------------|
| single_shot | none                | 15                        | 80%  | --                   |
| single_shot | evidence_first      | 6                         | 0%   | 100% (absolute 80pp) |
| react       | none                | 5                         | 60%  | --                   |
| react       | evidence_first      | 7                         | 0%   | 100% (absolute 60pp) |
| react       | verification_gate   | 7                         | 42.9%| 28.6% (absolute 17.1pp) |

See `results/plots/asr_defenses.png` (regenerate for the 40-case data).

**evidence_first is now a strong, well-supported finding:** it reduced
Attack Success Rate to 0% for both agents on the categories it applies
to. This held consistently across both the 20-case and 40-case corpus
runs. It is not a free win, however -- the "cases correct when clean"
column shows the denominator itself shifted under evidence_first
(single_shot: 15 -> 6; react: 5 -> 7), meaning the defense also changes
how often each agent answers ATTACK on cases unrelated to the specific
injected instruction, i.e. it makes the agent more generally cautious,
not just resistant to the injected text specifically. This is a genuine
precision/robustness trade-off that should be stated plainly rather than
presented as an unqualified improvement.

**verification_gate shows a moderate, more modest effect** (29% relative
ASR reduction for react) -- real but considerably weaker than
evidence_first.

## 4. Agent behaviour: tool reliance and tool-trust correlation

React's tool-calling rate drops sharply under attack when undefended
(82.5% clean -> 47.5% attacked), and the defenses change this behaviour
differently: evidence_first keeps tool-calling rate essentially flat
(72.5% -> 72.5%), while verification_gate causes the steepest drop
(70% -> 30%). This suggests the two defenses work through different
mechanisms -- evidence_first does not appear to discourage tool use, while
verification_gate appears to make the agent avoid consulting the tool at
all more often under attack, not only distrust what it returns.

The tool-trust correlation did **not** replicate the direction seen in
the earlier 20-case corpus: with the larger 40-case sample, attack
success was actually slightly *lower* when the tool was called (9.1%,
n=11) than when it was not (14.3%, n=7). The earlier n=2 "not called"
result was too small to be reliable, and this reversal is a useful
illustration of exactly why the project treats small-n results as
directional rather than conclusive -- this correlation should now be
read as inconclusive rather than confirming the original tool-trust
hypothesis, pending a larger sample still.

**Rationale coding**, by contrast, gives direct and unambiguous
supporting evidence for tool poisoning as a mechanism: 3 of 6 successful
tool/compound-poisoning attacks produced a rationale that explicitly
cited the poisoned tool's fabricated trust claim, e.g. "traffic directed
to a trusted, reputable IP address" and "source IP is on the corporate
allowlist." This is qualitative, case-level evidence that complements
the (inconclusive) aggregate correlation above.

Parse/format compliance remained 100% for both agents across all 628
predictions in this run (238 single_shot, 390 react) -- the output
format never broke under any attack condition.

## 5. Known limitations

- The adversarial corpus (40 cases) and clean sample (150 flows) are
  larger than the project's original scale but remain modest by the
  standards of a production security evaluation; several per-category
  ASR denominators are still in the low single digits.
- Single LLM (Gemini, lite-tier free models) at temperature 0 -- results
  may not generalize to other models, providers, or temperatures.
- `flow_to_text` trims filler/zero fields per protocol, which was
  necessary to keep prompts from being dominated by irrelevant
  zero-valued fields, but is itself a simplification that could discard
  some genuine signal.
- Phase 5/6 were first run on synthetic (randomly labeled) data before
  the real dataset was available; those synthetic results remain in
  `adversarial_results.csv` tagged `dataset=synthetic` and should not be
  cited as findings -- they only demonstrate the harness worked
  end-to-end.
- Repeated-run consistency at temperature 0 (i.e. whether identical
  input reliably produces an identical output) was not separately
  measured.