# Supply Chain Forecasting Research

Reproducible comparison of a project-reference **Spatio-Temporal Graph
Transformer (STGT)** and **Adaptive Fusion GAT-LSTM** for hierarchical demand
forecasting.

## Dataset confirmed

The supplied archive is the **Corporación Favorita Grocery Sales Forecasting**

```text
train.csv.7z
items.csv.7z
stores.csv.7z
transactions.csv.7z
oil.csv.7z
holidays_events.csv.7z
test.csv.7z
sample_submission.csv.7z
```

The forecasting pipeline needs `train.csv`, `items.csv`, and `stores.csv`.
Calendar, oil, transactions, test, and submission files are preserved because
current baseline uses only train/items/stores so that both models receive the
same information.

The Favorita data has `state`, `city`, `store_nbr`, `item_nbr`, `family`, and
`unit_sales` fields. Because it has no explicit region field, preprocessing
uses `state` as the geographic **Region** level:

```text
State/Region -> City -> Store -> Product Family
```

Negative `unit_sales` values are counted and clipped to zero for demand
modeling; the raw data is never changed. Missing date/node cells are filled
with zero under the documented Favorita ledger policy and counted in
`data_quality.json`. Use `--missing-policy error` to reject them instead.

## Repository and Drive separation

GitHub stores source code, configuration, tests, and documentation. Google
Drive stores the large raw archives, generated processed CSVs, checkpoints,

Recommended Drive layout:

```text
MyDrive/supply_chain_research/
├── data/raw/
├── data/processed/
├── checkpoints/
├── predictions/
└── results/
```

Do not commit raw data or model artifacts to GitHub.

## Local setup

The local project is intended to live at `C:\Projects\research` and is
portable to Linux/Colab. Do not put Windows paths in the configuration.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Colab/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The Downloads archive contains nested `.7z` members. Install `py7zr` from the
requirements file, or extract the members with 7-Zip into `data/raw/` before
running preprocessing. The script can materialize a missing `*.csv.7z` when
`py7zr` is available.

## Preprocessing

Do not run this command on a local machine unless sufficient disk/RAM is
available. Run it in Colab after putting the raw archives in Drive:

```bash
python src/preprocessing.py \
  --raw-dir /content/drive/MyDrive/supply_chain_research/data/raw \
  --processed-dir /content/drive/MyDrive/supply_chain_research/data/processed \
  --start-date 2015-01-01 \
  --end-date 2017-12-31 \
  --missing-policy zero
```

Generated files:

| File | Schema |
| --- | --- |
| `nodes.csv` | `node_id,node_type,label,parent_id,region,city,store_nbr,family` |
| `edges.csv` | `source,target,edge_type` |
| `sales.csv` | `Date` plus one numeric column per `node_id` |
| `data_quality.json` | counts, policies, date range, and graph statistics |

The script uses chunked `train.csv` processing and aggregates only to the
small node/time table. It does not smooth, interpolate, or normalize data;
normalization is fit on the training prefix inside the shared experiment
loader to prevent leakage.

## Models

### STGT

`src/stgt.py` implements the supplied project-reference architecture:

1. Temporal self-attention using a Transformer encoder.
2. Learned node-type embeddings.
3. Two `TransformerConv` spatial graph layers.
4. Configurable multi-step output head.

There is no single canonical architecture universally identified by the STGT
acronym. The code therefore calls this `ReferenceSTGT` and does not claim it
is an exact reproduction of an unnamed external paper.

### Adaptive Fusion GAT-LSTM

`src/gat_lstm.py` implements the proposed model with:

1. Bidirectional LSTM temporal encoding.
2. Temporal attention pooling.
3. Multi-layer/multi-head GAT spatial encoding.
4. Per-node adaptive temporal/spatial fusion gate.
5. Configurable multi-step prediction head.

Residual-baseline support is available in the dataset interface but disabled
in the initial fair comparison. A causal EMA baseline must be evaluated only
with a forecast-safe multi-step rollout; using future actual values to build a
multi-step baseline would leak information.

## Fair comparison

Both models use the same `ExperimentData` object, graph, node order, scaler,

The default configuration uses:

```text
seed: 42
split: 70% train / 15% validation / 15% test
lookback: 14
horizon: 1
scaler: StandardScaler
```

The data loader uses historical context across split boundaries but never
targets a future date in an earlier split. MAPE is masked to actual values
above `0.1`; WAPE and sMAPE remain defined for zero-demand observations.

Train each model independently:

```bash
python src/train.py --model stgt --config configs/experiment.yaml
python src/train.py --model gat_lstm --config configs/experiment.yaml
```

Then compare saved checkpoints:

```bash
python src/evaluate.py \
  --stgt-checkpoint results/checkpoints/stgt.pt \
  --gat-lstm-checkpoint results/checkpoints/gat_lstm.pt \
  --config configs/experiment.yaml
```

Outputs include JSON reports, test predictions, checkpoints, per-horizon and

## Colab workflow

Open `notebooks/colab_experiment.ipynb`, or execute its cells in order:

1. Clone `https://github.com/Akifali-1/research`.
2. Mount Google Drive.
3. Install `requirements.txt`.
4. Print package versions and GPU information.
5. Run preprocessing from the repository against raw data in Drive.
6. Train STGT.
7. Train Adaptive Fusion GAT-LSTM.
8. Evaluate both checkpoints with the shared evaluator.
9. Save outputs to Drive.

The standard GitHub + Drive + Colab workflow is the supported execution path.
No API, quota bypass, browser automation, or guaranteed free-T4 allocation is
assumed.

## Semi-automated execution and safety gates

`orchestrator.py` is a sequential controller for the authorized Colab session.
It uses a JSON registry stored in Drive to enforce:

- maximum 10 submitted GPU runs;
- one active run at a time;
- at most two retries per configuration signature;
- no repeat submission for a completed configuration;
- explicit records for reserved, running, completed, failed, and interrupted runs;
- configuration, Git commit, dataset archive hashes, logs, and artifact paths.

The notebook refuses to proceed unless the cloned commit matches `origin/HEAD`,
Drive has been mounted interactively, all eight raw archives and their inner
members validate, and a budget slot remains. A separate confirmation cell
requires the user to type `RUN`. Until that happens, no training subprocess is
started. A restarted session marks stale active records as interrupted and can
resume them as a new counted attempt; completed records are skipped by their
deterministic signature.

The first source push must happen before the notebook can pass its repository
gate:

```bash
git switch -c research/initial-pipeline
git add .
git commit -m "feat: add reproducible supply chain experiment pipeline"
git push -u origin research/initial-pipeline
```

The notebook should then be opened from that pushed branch or the repository's
default branch after it is selected on GitHub. Do not start Colab experiments
until the pushed commit and Drive archive checks both pass.

## Lightweight verification

The tests use only synthetic data and never load the Favorita dataset or train
a model:

```bash
pytest -q
python -m compileall -q src tests
```

On a machine without PyTorch/PyTorch Geometric, the model-forward test is
skipped; install `requirements.txt` in Colab to run the complete suite.
