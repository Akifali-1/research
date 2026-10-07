# Verified reliability and scientific-control review

## Classification

| Finding | Classification | Resolution |
|---|---|---|
| Fixed `.train.csv.extracting` staging name | Confirmed recovery bug | Unique, owned temporary directories; handled failures clean only their own scratch files. Abandoned directories from killed sessions cannot block later extractions. |
| Cache trusted by size alone | Confirmed integrity bug | Archive-content namespace, complete extraction/CRC verification, SHA-256 sidecars, and checksum validation on reuse. Suspect old cache versions remain preserved while a new verified version is generated. |
| Failed final-test gate cannot resume | Deliberate fail-closed design with a reliability gap | Explicit `--resume-final-test`; immutable frozen IDs; checksummed per-model and report stages; completed inference is reused. |
| Only a subset of input schemas checked | Confirmed validation gap | Required columns for all eight sources; bounded header-only 7z decoding; recheck materialized train schema before chunk aggregation. |
| Different architectures/parameter counts; one seed | Scientific limitation, not a code bug | Transparent project-reference label; parameter/runtime/selected-epoch reporting; seed-42 results remain exploratory. |
| Data-load/epoch timeouts require explicit retry | Intended operational control | Limits and counted retries retained; interruption/optimizer/RNG tests added. Partial epochs replay only in an authorized attempt. |
| Timeout after a parent exits can leave its child alive | Additional confirmed process-control bug | On Linux/Colab the session process group is terminated even if the parent has already exited. |

## Cache recovery

CSV archives are materialized under the cache's `.verified-csv/<archive-sha256>/`
namespace. A verified CSV has an integrity sidecar containing archive SHA-256,
CSV SHA-256, byte count, and CRC32. Every reuse checks its complete contents
against that record. Size alone is never sufficient.

The extractor verifies successful decompression, member size and available
archive CRC32, and CSV schema before atomic promotion. A source archive change
creates a separate namespace. A corrupted, orphaned or otherwise suspect cache
is preserved; a new unique verified directory and atomic `active.json` pointer
allow later calls to reuse the replacement without repeatedly extracting it.

Only scratch directories created by the current invocation are cleaned after
handled failures. Old raw archives, cached versions, historical results and
legacy `.train.csv.extracting` directories are not removed. A hard runtime kill
can leave scratch behind; unique names make it harmless to subsequent attempts.
Repeated full-file checksum reads cost I/O; integrity is preferred over a fast
but unsafe size/mtime-only cache check.

## Source schemas

`src/source_schema.py` defines all eight Favorita schemas. Direct CSVs are read
only through their headers; compressed CSVs use a bounded writer factory that
stops decoding after the first header. This is explicitly **not** a full CRC
scan of the large archives. Full train/items/stores extraction in preprocessing
checks archive integrity and verifies the resulting CSV schema before use.

## Explicit final-test recovery

After an interrupted evaluation, inspect the saved status and latest
`final_test/evaluation_attempts/*/evaluation.log`. Confirm the previous Colab
process is no longer running. Then run the same final-test command with the
same selected IDs and add:

```text
--resume-final-test
```

Example in an authorized Colab runtime (replace the IDs with the frozen pair):

```bash
python orchestrator.py \
  --project-root /content/research \
  --branch research/initial-pipeline \
  --config /content/experiment_colab.yaml \
  --raw-dir /content/drive/MyDrive/supply_chain_research/data/raw \
  --registry /content/drive/MyDrive/supply_chain_research/experiment_registry.json \
  --finalize-test --experiment-ids <frozen-stgt-id> <frozen-gat-lstm-id> \
  --resume-final-test --allow-execution
```

The notebook exposes this as `EVAL_RESUME`. It is separate from `RETRY`/`RESTART`,
which authorize budget-counted **training** attempts. Evaluation recovery never
reserves a training slot or launches the trainer.
The persisted campaign deadline, when present, also caps final evaluation and
cannot be reset by recovery. An expired campaign requires user review rather
than silently extending the configured runtime allowance.

- `selection.json` is retained byte-for-byte. Different IDs/data/config are rejected.
- Verified completed model stages are reused, not rerun.
- Failed scratch stages are retried; handled failures clean only that attempt's scratch.
- If report generation fails, verified model predictions remain reusable.
- Corrupt **completed** stages/results stop for inspection instead of being overwritten.
- Existing legacy partial outputs are preserved; new stages use `final_test/stages/`.
- Each evaluation attempt has its own configuration, provenance, status and live/saved log.
- Concurrent evaluators are blocked by an exclusive evaluation lock. A stale
  lock after a hard kill requires user-confirmed inspection before removal.

New output paths are:

```text
final_test/stages/stgt/{result.json,predictions.npz,complete.json}
final_test/stages/gat_lstm/{result.json,predictions.npz,complete.json}
final_test/stages/report/model_comparison.csv
final_test/stages/report/comparison_report.md
```

Older **completed** evaluations keep their original output paths and are
checksum-verified without recomputation.

### Updating evaluator code for existing checkpoints

Training provenance is never rewritten. If an evaluation-only maintenance
update is needed for runs from an older commit, use the additional explicit flag:

```text
--allow-evaluator-upgrade
```

The notebook calls this `COMPATIBLE_UPGRADE`. Models, dataset and metric modules
are AST-fingerprinted; shared loader/scaling/prediction definitions are also
checked against the recorded training commit. Changed scientific definitions
are rejected. Both original training and current evaluator Git commits are
recorded. Once evaluation stages exist, compatible code fingerprints, package
versions and frozen configuration must remain identical for recovery.

Keep the checkout at its recorded commit when resuming training. Updating source
mid-run or before a training retry changes the run identity; maintenance updates
do not authorize silently repeating completed GPU runs.

## Scientific interpretation and budget

The STGT baseline remains a project-reference implementation, not a verified
reproduction of an identified original paper. Equal epoch ceilings do not imply
equal parameter count, FLOPs, or wall time. Both counts and measured times are
reported, along with the validation-selected epoch. One seed is exploratory.

Additional predeclared seed pairs may be run only with available registry slots
and before test selection is frozen. The ten-run budget includes failed attempts
and training retries. Do not increase it, discard failures, or tune from test
results. Multi-seed aggregate evidence is still pending; no extra run is started
by this reliability update.
