# Security Evaluation of LLM-Based Agents for IoT Attack Detection

A small, reproducible research project studying one focused question:

> How easily can an LLM-based IoT security agent be misled by malicious or
> misleading content embedded in its input, and can simple defenses reduce
> that risk?

The project will eventually compare a single-shot classifier agent against a
multi-step (ReAct-style) agent with tool use, under both clean and
adversarial conditions, and measure whether lightweight defenses help.

This is a research/evaluation project, **not** a production intrusion
detection system.

## Project status

**Phase 1 — Project scaffolding:** done. Repository structure, config,
virtual environment, and a passing smoke test (`verify_setup.py`).

**Phase 2 — IoT dataset and data pipeline: done.** Raw data loading,
cleaning, BENIGN/ATTACK labeling, and reproducible train/val/test splits.
No agents, adversarial examples, or defenses yet — that's later phases.

## Dataset

**Chosen dataset: Edge-IIoTset** (specifically the `DNN-EdgeIIoT-dataset.csv`
selected-features file), over CICIoT2023.

| | CICIoT2023 | Edge-IIoTset (DNN CSV) |
|---|---|---|
| Size | ~47M flows, ~13GB across many CSV files | ~2.2M rows, single CSV file |
| Labels | 33 attack types across 7 attack classes + benign | 14 attack types + Normal, plus a ready-made binary `Attack_label` |
| Features | 47 statistical/flow-summary features (packet counts, inter-arrival times, byte rates) | 61 protocol-level features (`ip.src_host`, `http.request.method`, `dns.qry.name`, `mqtt.topic`, TCP flags, etc.) |
| Preprocessing difficulty | High — many files to combine, large memory footprint, sampling required on a laptop | Moderate — one file, but needs dropping ~15 identifier/payload columns per the dataset's own documented recipe |
| Text-for-LLM suitability | Lower — features are numeric aggregates, less naturally describable in a sentence | Higher — protocol fields read naturally into a flow summary (e.g. "an HTTP GET request from device X to Y", "a DNS query for Z") |
| Practicality for a student project | Harder — large multi-file download, needs a beefy machine or heavy sampling | Easier — single manageable file, well-documented, freely available on Kaggle |

**Why Edge-IIoTset:** it's a single, manageable CSV rather than a
multi-file, ~13GB download; it already ships with both a clean binary label
(`Attack_label`) and a detailed attack-type label (`Attack_type`), which
maps directly onto this project's BENIGN/ATTACK scheme; and its
protocol-level features (HTTP methods, DNS queries, MQTT topics, TCP flags)
translate much more naturally into the plain-language flow summaries this
project will eventually feed to an LLM, compared to CICIoT2023's purely
statistical features. CICIoT2023 is larger and arguably more realistic at
scale, but that scale is a practical burden rather than a benefit here,
since the research question is about adversarial robustness, not
state-of-the-art detection accuracy.

Dataset paper / source: M. A. Ferrag et al., "Edge-IIoTset: A New
Comprehensive Realistic Cyber Security Dataset of IoT and IIoT
Applications," IEEE Access, 2022. Available on Kaggle:
`mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`.

The real dataset is **not** included in this repository (it's large and
requires a Kaggle account) — see "Running the pipeline" below for how to
get it. A small synthetic sample file,
`data/raw/sample_synthetic_flows.csv`, is included purely so the pipeline
can be smoke-tested without the real download.

## Repository structure

```
data/
  raw/           # original downloaded dataset goes here (gitignored, except the synthetic sample)
  processed/     # pipeline output: cleaned data + train/val/test splits (gitignored)
  adversarial/   # will hold adversarial test cases (Phase 5, not yet added)
src/
  data/
    preprocess.py   # Phase 2 pipeline: load, clean, label, split
  agents/        # will hold agent code (Phase 3/4, not yet added)
  tools/         # will hold the mock reputation tool (Phase 4, not yet added)
  defenses/      # will hold defense strategies (Phase 6, not yet added)
  evaluation/    # will hold metrics/experiment runner (Phase 7, not yet added)
notebooks/       # for exploratory analysis
results/
  tables/        # class_distribution.csv and future result tables
  plots/         # future charts
run_preprocessing.py  # command-line entry point for the Phase 2 pipeline
verify_setup.py        # Phase 1 smoke test
config.yaml             # project + dataset configuration
requirements.txt        # Python dependencies
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Phase 1 smoke test: `python verify_setup.py`

## Running the pipeline

1. Download `DNN-EdgeIIoT-dataset.csv` from Kaggle
   (`mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`,
   under "Selected dataset for ML and DL") and place it at
   `data/raw/DNN-EdgeIIoT-dataset.csv`.
2. Run:
   ```
   python run_preprocessing.py --input data/raw/DNN-EdgeIIoT-dataset.csv
   ```
3. Output appears in `data/processed/`: `processed_full.csv`, `train.csv`,
   `val.csv`, `test.csv`, and `dataset_stats.json`.

To smoke-test the pipeline without the real dataset:
```
python run_preprocessing.py --input data/raw/sample_synthetic_flows.csv --output-dir data/processed
```

## Roadmap (not yet built)

- Phase 3: Single-shot classifier agent.
- Phase 4: Multi-step ReAct agent with mock tool.
- Phase 5: Adversarial test corpus.
- Phase 6: Defense layer.
- Phase 7: Evaluation and reporting.
