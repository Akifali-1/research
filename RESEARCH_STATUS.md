# Research status and unresolved details

## Evidence available

- Supplied Favorita archives and small CSV metadata schemas.
- Supplied Colab STGT implementation and a few diagnostic predictions.
- Supplied optimized BiLSTM/GAT/adaptive-fusion reference code.
- Local source code and synthetic/static verification.

## Evidence not yet available

- A specifically identified original STGT paper and verified protocol.
- Authorized Drive access and verified uploaded dataset in this session.
- Colab runtime test, training logs, checkpoints, or actual comparison metrics.
- Multi-seed results and ablations.

The proposed model and baseline are labeled transparently. The current GAT-LSTM
attention placement follows the supplied code (temporal pooling before GAT and
fusion), not an invented post-fusion sequence. The original paper's dataset
statistics and claimed results are not reused as results from this experiment.

## Current experiment plan

1. Push the dedicated branch after user GitHub authentication.
2. Authorize Colab Drive mount and upload all source archives.
3. Pass full synthetic tensor/checkpoint tests in Colab.
4. Prepare/hash-validate versioned Favorita aggregates on Drive.
5. Run seed-42 STGT and GAT-LSTM sequentially; select checkpoints using validation.
6. Inspect validation evidence; no test-driven tuning.
7. Freeze a pair of completed run IDs and authorize the final-test cell.
8. Report actual metrics, runtime, parameters, and failures. Additional seeds
   or focused ablations require explicit configs inside the remaining budget.

Search is deliberately disabled pending evidence. No GPU experiment has been
submitted locally. No dataset, credential, or generated model artifact belongs
in GitHub. There is no results table yet because no real experiment has run.

## Limitations to disclose

- Global family nodes overlap geographic aggregates and are not store-specific
  leaves. Store→Family is not a strict conservation relation.
- Missing ledger cells are zero recorded sales under an explicit assumption;
  they are not estimates of unobserved demand or out-of-stock sales.
- The baseline's horizon extension and proposed multi-step output are adaptations,
  not verified reproduction of a published experiment.
- Epoch parity does not guarantee equal compute; time/parameter counts are reported.
- GPU determinism and Drive locking across separate sessions are not guaranteed.
- Partial epoch recovery replays that epoch in a new budget-counted retry.
- Final evaluation failure preserves logs/selection; automatic replacement of
  frozen test results is intentionally unavailable and requires inspection.
