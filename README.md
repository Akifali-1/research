# Adaptive Fusion GAT-LSTM for Interpretable Supply Chain Demand Forecasting

Semi-automated, reproducible comparison against a **project-reference STGT**.
Code lives in GitHub; datasets/artifacts live in Google Drive; large processing
and model experiments run in an interactively authorized Google Colab session.

**Status:** source prepared locally; GitHub push requires user authentication.
No real preprocessing, GPU training, or test-set evaluation has been run. No
result or model superiority is claimed. See `RESEARCH_STATUS.md` for limitations.

## Structure

```text
research/
├── configs/experiment.yaml        # seed, shared protocol, hard limits
├── configs/search_space.yaml      # search disabled until baseline evidence
├── data/raw/                      # local source archives, ignored by Git
├── data/processed/                # generated/versioned data, ignored
├── notebooks/colab_experiment.ipynb
├── orchestrator.py                # Colab/Drive/push/quality/budget gates
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── stgt.py
│   ├── gat_lstm.py
│   ├── train.py
│   ├── evaluate.py
│   ├── checkpointing.py
│   ├── experiment_registry.py
│   ├── metrics.py
│   └── utils.py
├── tests/
├── results/                       # directory markers only
├── requirements.txt
└── README.md
```

## Dataset and source preservation

The local Favorita ZIP contains eight `.7z` archives:
`train.csv.7z`, `test.csv.7z`, `items.csv.7z`, `stores.csv.7z`,
`transactions.csv.7z`, `oil.csv.7z`, `holidays_events.csv.7z`,
`sample_submission.csv.7z` (approximately 480 MB in total).

Inspected metadata headers confirm **Corporación Favorita Grocery Sales
Forecasting**: items contain `item_nbr,family,class,perishable`; stores contain
`store_nbr,city,state,type,cluster`. Preprocessing checks train columns
`id,date,store_nbr,item_nbr,unit_sales` when reading the CSV. The large train
CSV has not been locally expanded. Preserve all eight archives, including
metadata/covariate files unused by the initial comparison. Nothing is deleted.

Upload either those eight archives, eight extracted CSVs, or the original
`favorita-grocery-sales-forecasting.zip` to the Drive raw folder below. The
preflight recognizes a single nested source folder, accepts mixed CSV/7z
layouts, and can unpack the **outer ZIP only**, leaving the large train 7z
compressed. Original ZIP/source files are preserved. If the ZIP is elsewhere
in Drive, set `SOURCE_ZIP` in the notebook to its exact path.

Files absent from Drive cannot be recovered from local Windows paths in Colab.
For missing-source errors, inspect the printed raw-folder listing, correct
`RAW_DIR`, or upload the ZIP; do not fabricate an empty missing source.

Drive layout:

```text
MyDrive/supply_chain_research/
├── data/raw/                      # all eight original archives
├── data/processed/<version>/      # output + completion/data-quality manifests
├── experiment_registry.json      # durable budget and attempts
└── experiments/
    ├── <model>-<signature>-attempt1/
    │   ├── config.yaml
    │   ├── runtime.json
    │   ├── dataset_manifest.json
    │   ├── training.log
    │   ├── checkpoints/{best,latest}.pt
    │   ├── logs/history.json
    │   ├── predictions/validation.npz
    │   └── metrics/validation.json
    └── final_test/                # frozen selection, test predictions/report
```

The Colab preflight hashes the selected CSV or 7z sources, checks inner member names/sizes, and
extracts only small compressed metadata to verify schemas. Direct CSV schemas
are inspected from their headers. When both representations exist, 7z is the
consistent preference of preflight and preprocessing. This is **not** a claim of a
full CRC scan of the large train archive. Its full extraction in Colab checks
integrity before a processed completion marker can exist or training can start.

## Preprocessing and graph semantics

State is used as **Region** (Favorita has no region field). Each geographic
region, state-qualified city, store, and **global** product family is one node.
Region→City and City→Store edges are metadata relationships. Store→Family
edges connect each metadata store to each metadata family; they are not chosen
using validation/test sales. Families aggregate across all stores. Consequently
this is a typed aggregation graph, **not a strict summation tree** at the
Store→Family level; parent-child consistency loss is not appropriate as-is.

Node IDs are deterministic; slug collisions and missing/duplicate metadata
keys cause errors. Source IDs/dates must be ordered; duplicate/unordered IDs
are rejected across chunk boundaries. Distinct ledger rows with the same
business key are summed; within-chunk duplicate business-key rows are counted.
This is not global deduplication of legitimate ledger entries.

Chunked processing aggregates into a small `[dates,nodes]` float64 matrix, not
an 80-million-row in-memory table. Negative sales/returns are counted and
clipped to zero for the shared nonnegative-demand task. Invalid dates, sales,
and unknown item/store references stop preprocessing rather than becoming zero.

The explicit default `zero` policy interprets an absent aggregate ledger cell
as **zero recorded sales**, not as imputed latent demand. It is a documented
modeling assumption. Use `missing_policy: error` to reject absent cells. Actual
zero and missing cells are separately tracked. The timeline ends at the last
labeled date, never at a manufactured future zero-filled date.

| Output | Schema |
|---|---|
| `nodes.csv` | `node_id,node_type,label,parent_id`, optional region/city/store/family metadata |
| `edges.csv` | `source,target,edge_type` (unique, no self-loops) |
| `sales.csv` | `Date`, then one numeric column per `node_id` |
| `data_quality.json` | date range, counts, policies, graph/source validation information |

CSV extraction uses a Colab-local cache; raw Drive archives remain unchanged.
Processed outputs are versioned. Completed versions are hash-verified and
reused. Incomplete preprocessing stops for inspection; choose a new processed
base for an explicit restart, keeping the incomplete logs.

For manual **Colab-only** preprocessing after validation:

```bash
python src/preprocessing.py \
  --raw-dir /content/drive/MyDrive/supply_chain_research/data/raw \
  --cache-dir /content/favorita-csv-cache \
  --processed-dir /content/drive/MyDrive/supply_chain_research/data/processed/manual-v1 \
  --start-date 2015-01-01 --end-date 2017-08-15 --chunksize 500000
```

Prefer the notebook's `prepare_data` stage, which also produces the required
raw/processed provenance and completion markers.

## Architecture and comparison protocol

**ReferenceSTGT:** the supplied Colab project implementation: temporal
Transformer encoder, learnable positional encoding and temporal pooling,
node-type embedding, two `TransformerConv` spatial layers, direct forecast
head. The horizon extension changes the output from one value to `H` values.
No identified original STGT paper has been verified; this is transparently a
project-reference baseline, not a claimed published-model reproduction.

**AdaptiveFusionGATLSTM:** supplied optimized design: BiLSTM, temporal
attention pooling, LayerNorm, stacked multi-head GAT with projected residual,
adaptive scalar temporal/spatial fusion gate, and prediction MLP. Temporal
attention is before GAT/fusion, matching the supplied code. Gate weights are
available through `return_attention=True`; they are model diagnostics, not
causal explanations. Input perturbation is disabled in the initial experiment.

Both models accept `x=[B,N,L,1]` and return `[B,N,H]`. Graph nodes are ordered
identically. Batched edges are offset for each independent time-window graph,
so attention cannot mix different windows. Types are available from shared
metadata; the reference baseline uses an embedding, the proposed architecture
does not. Edge labels are documented but unused by both initial spatial layers.

Initial protocol (`configs/experiment.yaml`):

- seed 42, configurable;
- 14-day history and 7-day **direct** multi-step horizon;
- chronological 70%/15%/15% train/validation/test;
- all metadata nodes, identical forecast origins and targets;
- no smoothing, full-series outlier filtering, or future-dependent pruning;
- per-node StandardScaler fit on training dates only;
- shared Huber loss, AdamW, epoch limit, scheduler and early-stopping rule;
- residual/EMA adapter disabled for both; enabled adapter currently supports
  horizon 1 only, explicitly rejecting unsafe multi-step residual construction.

Targets in an earlier split cannot extend into the next split. Validation/test
inputs may use already observed past dates across a boundary. Test evaluation
is rolling-origin: later test forecasts use past observed test values, never
future values for that origin. Overlapping multi-horizon forecast tasks are
not independent samples, and this is not one long open-loop rollout.

Models have different parameter counts; equal epochs and shared rules are a
comparable starting budget, not exact FLOP parity. Report both counts/times.
GPU scatter/reduction kernels can remain nondeterministic; deterministic
settings warn rather than falsely guaranteeing identical GPU results.

## Budget, interruption, and artifact integrity

Hard caps: **10 submitted training attempts**, one active run, **2 retries** per
signature. Failed/interrupted submitted attempts also consume slots. Rejected
preflight attempts do not. The initial two model runs are inside the ten slots.
No automatic budget increases or paid resources are used.

Job limit: 7200 seconds. Campaign limit: 21600 seconds from its first training
submission. The persisted deadline cannot reset on Colab reconnection.
Completed signatures skip even at a spent budget; missing/corrupted completed
artifacts stop execution rather than triggering hidden retraining.

Every epoch atomically saves weights, optimizer/scheduler, best validation
state, history, and Python/NumPy/Torch RNG states on Drive. `RETRY` authorizes
a new counted attempt that resumes after the last committed epoch. A partial
epoch is explicitly replayed. If no epoch checkpoint exists, `RESTART` is an
explicit counted restart, not a silent retry. Old attempt directories remain.

Confirm no old process is alive before typing `RECOVER` for stale active
records. A stale `.lock` after interruption stops safely: inspect it and confirm
all users/runtimes are inactive before manually resolving it. One authorized
Colab session should control this registry; Drive does not provide a tested
multi-machine distributed locking service here.

## Live training logs

Training stdout/stderr is streamed live to the Colab cell and saved to the
attempt's `training.log`. Logs include experiment/device/window counts,
batch progress, epoch train/validation loss, best loss, patience, learning rate,
elapsed time, and checkpoint location. A silent/hung child still respects the
configured timeout. This applies to newly launched processes; an already
running process from an older commit retains its original logging behavior.

## Notebook stages

Open `notebooks/colab_experiment.ipynb` from branch `research/initial-pipeline`.

1. Clone/reuse the repository and verify clean source matches the pushed branch.
2. Mount Drive **interactively**. Upload the archives if absent.
3. Install dependencies, print GPU/RAM/disk/package versions, run synthetic tests.
4. Verify archives, registry state, and remaining budget.
5. Type `PREPROCESS` to prepare or verify the versioned processed dataset.
6. Create the portable Drive-path experiment configuration.
7. Type `RUN` for sequential validation-only training; `RETRY`/`RESTART` is explicit.
8. Review validation reports. Type `TEST` and choose the two completed run IDs
   only after configurations are final. The selection is frozen before test
   access. Final artifacts are verified/reused, not overwritten.

Final outputs: `experiments/final_test/metrics/model_comparison.csv`,
per-type/per-horizon JSON, `comparison_report.md`, `comparison.png`, and test
prediction NPZ files with node names, dates, and target indices. No metrics are
invented; these files appear only after the actual authorized evaluation.

Metrics are calculated on original units with predictions clipped to zero:
MAE, MSE, RMSE; WAPE=sum absolute errors/sum absolute actuals; sMAPE averages
2|error|/(|actual|+|prediction|) with 0/0=0; pooled R²; MAPE only for actual>0.1,
with valid fraction. Undefined WAPE/MAPE/R² are NaN, not a misleading zero.
Ratio metrics are ratios (multiply by 100 to display percentages). Overall
averages pool origin×node×horizon entries; overlapping geographic/family
aggregate levels require the reported per-type breakdowns for interpretation.

## Local verification and Git

WSL checks with no ML dependencies:

```bash
python3 -m compileall -q src tests orchestrator.py
python3 -m unittest discover -s tests -p 'test_registry_stdlib.py' -v
```

Full CPU synthetic tests, after dependencies exist:

```bash
python -m pytest -q
```

Tests never load the real dataset or train a model. They cover tiny chunked
preprocessing, indexing/window leakage, forward-pass batch isolation and loss,
train-only scaling, checkpoint/RNG round trips, budget/retry/concurrency,
completed skips, persistent deadlines, and notebook cell syntax/order.
The controller also runs the complete synthetic quality gate before reserving
an expensive attempt. Local ML-dependent tests currently skip until Colab
installs PyTorch/PyG.

Git source is committed on `research/initial-pipeline`. Authenticate using your
chosen GitHub credential manager, then push:

```bash
git push -u origin research/initial-pipeline
```

Do not paste a token into a command, notebook, or this repository. Raw data,
checkpoints, logs, predictions, credentials, and generated metrics are ignored.
`main` has not been merged or modified. Public GitHub read access does not
authorize writing to the repository.

There is no tested direct Colab API/MCP/Drive integration in this environment.
The first Drive authorization is a user action. This project prepares and
controls an authorized session; **unattended remote execution is not claimed**.
