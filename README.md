# 🌴 Palm Guard — حارس النخيل

> **Acoustic early-detection of Red Palm Weevil (RPW) in date palms.** A contact
> sensor listens inside the trunk, an on-device CNN classifies the sound as
> `infested` or `clean`, results sync to a backend, and infested trees surface on
> an Arabic-RTL map with alerts — *before* visible (terminal) damage appears.

**Sensor → Edge (DSP + TFLite inference) → API → DB → Dashboard + Alert**

## Results on real data

Trained and evaluated on the public **TreeVibes** RPW dataset, on a **held-out
*site* split** (test trees the model never saw):

| Metric | **CNN** | Classical baseline |
|---|---|---|
| **Infested recall** | **0.926** ✓ (target ≥ 0.90) | 0.926 |
| Infested precision | 0.595 | 0.291 |
| **PR-AUC** | **0.921** | 0.372 |
| Accuracy | 0.896 | 0.654 |

Caught **25 / 27** infested clips on unseen trees. The quantised edge model is
verified against the float model (**TFLite parity mean |Δ| = 0.025**, passed) and
runs on-device with no connectivity. The trained model ships in
[`models/`](models/).

> Evaluated on a held-out-site subset (7 trees, 396 clips, 0 dropped). The
> pipeline is dataset-agnostic and scales to the full corpus unchanged.

## Why it matters

RPW larvae feed hidden inside the trunk. By the time a palm looks sick it's
usually dead — and a reservoir for re-infestation. But the larvae are *audible* —
short feeding bursts in the ~0.2–2.5 kHz band. Catch that early and the tree is
treatable. **Priority above all: never silently miss an infested tree.**

## Repository layout

| Path | What |
|---|---|
| `packages/ml` | DSP, features, dataset pipeline (site-split), baseline + CNN, TFLite export |
| `packages/api` | FastAPI backend, debounced status engine, alerts, in-memory + Supabase |
| `packages/edge` | Raspberry Pi capture → on-device inference → store-and-forward uploader |
| `packages/web` | Next.js Arabic-RTL dashboard: map, tree detail, alerts inbox |
| `models/` | Trained TreeVibes CNN: `palmguard.tflite` + metrics + parity report |
| `docs/` | `BUILD_SPEC.md` (design + science), `HARDWARE.md`, `DEPLOY.md` |
| `CLAUDE.md` | Operating rules for working in this repo |

## Quickstart

```bash
cp .env.example .env

# ML pipeline (CPU, no download needed)
make data        # synthetic RPW-like audio + manifest -> data/raw/
make baseline    # RandomForest baseline on the held-out SITE split
make test        # ml + api test suites

# Backend (in-memory DB if Supabase env is unset)
make api         # http://localhost:8000  (docs at /docs)

# Dashboard
cd packages/web && npm install && npm run dev   # http://localhost:3000

# Edge (no hardware needed): classify a clip and store-and-forward
cd packages/edge && python run.py --sim ../../data/raw/infested/*/clip_000.wav --once

# Full end-to-end demo (API + edge + alert)
scripts/demo.sh
```

CNN training/export need TensorFlow (`pip install tensorflow`), then
`make train && make eval && make export`.

### Train on the real TreeVibes data

TreeVibes is on Kaggle (`potamitis/treevibes`). Point the ingester at a local
copy, a Kaggle slug, or a direct URL — labelling is CSV-driven from the published
folder lists, and the **site = folder number** (one folder ≈ one tree):

```bash
export TREEVIBES_LOCAL=/path/to/extracted/treevibes   # or TREEVIBES_KAGGLE / TREEVIBES_URL
make data            # builds manifest + prints a full ingest report (0 silent drops)
make train && make eval && make export
```

## Guarantees worth knowing

- **Site-split, not clip-split** — clips from one tree never span train/test.
- **Recall-first** — models are selected/reported on infested-recall + PR-AUC.
- **Edge-first** — inference runs on-device via TFLite with no connectivity.
- **Debounced alerts** — one noisy hit can't flip a tree; alerts are rate-limited.
- **Offline-safe edge** — detections are durably queued and flushed on reconnect.
- **Arabic-first, bilingual** — RTL default, AR/EN throughout.

See `docs/BUILD_SPEC.md` for the full rationale and `CLAUDE.md` for contributor
rules.
