# Phase 7: Evaluation Results

## Important caveat, read first

Clean-data accuracy on real Edge-IIoTset traffic is moderate at best (72-75%,
see below), and both agents miss a large share of real attacks even with no
adversarial technique applied. Every Attack Success Rate (ASR) figure in
this report is computed only from cases the agent already classified
correctly when clean -- which means many ASR denominators are small (often
1-6 cases). Treat ASR percentages as illustrative of the harness and the
direction of effects, not as statistically solid findings on their own.

## 1. Clean-data performance (n=60 real flows per agent, 30 ATTACK / 30 BENIGN)

| Agent       | Accuracy | Precision (ATTACK) | Recall (ATTACK) | F1 (ATTACK) |
|-------------|----------|---------------------|------------------|-------------|
| single_shot | 75%      | 89%                 | 57%              | 69%         |
| react       | 72%      | 100%                | 43%              | 60%         |

See `results/plots/clean_metrics.png`.

**Key finding:** single-shot outperforms ReAct on every metric. Both agents
have very few false positives (single_shot: 2/30, react: 0/30) but miss a
large fraction of real attacks (single_shot: 13/30 missed, react: 17/30
missed). Giving the agent a reputation-lookup tool did not improve attack
detection on clean data -- if anything, it made the agent more
conservative/cautious, catching fewer real attacks.

## 2. Adversarial robustness (20-case corpus, real attack flows, no defense)

See `results/plots/asr_by_category.png` and
`results/tables/adversarial_results.csv` for full detail, including the
`clean_correct` denominator for every row.

- Both agents are vulnerable to direct prompt injection (single_shot: 67-100%,
  react: 75-100% ASR, small-n).
- Tool-result poisoning succeeded against ReAct in the available cases.
- Context flooding showed near-zero `clean_correct`, so ASR for this category
  is largely `n/a` -- the agents already struggled with flooded flows before
  any true/false-flip could even be measured.

## 3. Defenses (evidence_first, verification_gate)

See `results/plots/asr_defenses.png`.

- `evidence_first` reduced ASR for direct injection on both agents, at the
  cost of lower clean accuracy in several categories -- a real
  precision/robustness trade-off, not a free win.
- `verification_gate` did not reduce ASR for ReAct's tool-poisoning in this
  sample; `clean_correct` counts were too small to draw a firm conclusion.

## 4. Known limitations

- Small corpus (20 adversarial cases, 60 clean cases) -- confidence
  intervals on all percentages above are wide.
- Single LLM (Gemini, lite-tier free models) at temperature 0 -- results may
  not generalize to other models or temperatures.
- `flow_to_text` trims filler/zero fields per protocol, which was necessary
  to keep prompts from being 90% noise, but is itself a simplification that
  could discard some genuine signal.
- Phase 5/6 were first run on synthetic (randomly labeled) data before the
  real dataset was available; those synthetic results remain in
  `adversarial_results.csv` tagged `dataset=synthetic` and should not be
  cited as findings -- they only demonstrate the harness worked end-to-end.