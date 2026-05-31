# Trained Palm Guard model (real TreeVibes data)

These are committed, persistent artifacts from training the CNN on a **real
TreeVibes RPW subset** (so they survive beyond the ephemeral training sandbox and
are downloadable for demos). They are committed intentionally — note the repo
otherwise gitignores `*.tflite` / model files.

## Files

| File | What |
|---|---|
| `palmguard.tflite` | Quantised edge model (40 KB). Load via `palmguard_ml.inference.TFLiteEngine` or the edge agent (`EDGE_MODEL_PATH`). |
| `metrics.json` | Held-out **site-split** metrics for the CNN. |
| `parity.json` | Keras-vs-TFLite agreement check. |

## Provenance

* **Data:** real TreeVibes subset — folders 1,2,3,4 (infested) + 7,24,25 (clean);
  396 clips, 187 infested / 209 clean, 7 sites, **0 dropped**.
* **Split:** by **site** (tree), never by clip. Test sites: folder_4 (infested),
  folder_7 (clean). No tree spans train/test.
* **Labels:** folder-level from the published TreeVibes lists, cross-checked
  against the annotation CSV's `AUDIO` column (0 disagreements). Site = FOLDER.

## Results (held-out site split)

| Metric | CNN | Classical baseline |
|---|---|---|
| Infested recall | **0.926** ✓ (target ≥ 0.90) | 0.926 |
| Infested precision | 0.595 | 0.291 |
| **PR-AUC** | **0.921** | 0.372 |
| Accuracy | 0.896 | 0.654 |

Confusion (rows=true clean/infested, cols=pred): clean `[138, 17]`,
infested `[2, 25]` — caught 25 of 27 infested clips.

**TFLite parity:** mean |Δ| = **0.025** (tolerance 0.05) over 7098 windows →
PASSED. The quantised edge model faithfully reproduces the float model.

> Caveat: this is a 7-folder subset of the 35-folder corpus, so the test set is
> ~1 tree per class. Numbers are strong but directional; train on more folders
> for statistically robust figures. Reproduce with:
> `TREEVIBES_LOCAL=<extracted folder> make data && make train && make eval && make export`.
