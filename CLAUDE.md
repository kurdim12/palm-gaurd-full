# CLAUDE.md — Operating Instructions for Palm Guard
> This file is read automatically by Claude Code at the start of every session. It tells you
> how to work in **this repo**. The full design rationale lives in `docs/BUILD_SPEC.md` — read
> it before any non-trivial change. Do not re-derive decisions already made there.
---
## What this project is
Palm Guard is an IoT + ML system that detects **Red Palm Weevil (RPW) larvae** acoustically,
*before* a date palm shows visible (terminal) damage. A contact sensor listens inside the trunk,
an on-device CNN classifies the sound as `infested` or `clean`, results sync to a backend, and
infested trees surface on an Arabic-RTL map with alerts.
**Mental model:** Sensor → Edge (DSP + features + TFLite inference) → API → DB → Dashboard + Alert.
---
## Repo map & current state
| Path | What it is | State |
|---|---|---|
| `packages/ml` | DSP, feature extraction, dataset pipeline, training, eval, TFLite export | **Working.** Tests pass. Baseline trains on synthetic data. CNN code complete but not yet run on real data. |
| `packages/api` | FastAPI backend, status engine, alerts, Supabase + in-memory fallback | **Working.** Tests pass. Runs with no DB configured. |
| `packages/edge` | Pi capture → TFLite inference → store-and-forward uploader | **Scaffold.** Runs in `--sim` mode; untested on real hardware. |
| `packages/web` | Next.js Arabic-RTL dashboard, Leaflet farm map | **Scaffold.** Map + status counts only; tree-detail and alerts inbox not built yet. |
| `docs/BUILD_SPEC.md` | Full specification + scientific basis | Reference. |
| `data/`, `*/artifacts/` | Gitignored. Generated locally. | — |
When you start a session, **verify state yourself** (`make test`) rather than trusting this table — it can drift.
---
## Golden rules (do not violate without flagging first)
1. **Scientific constants are fixed.** All of them live in `packages/ml/palmguard_ml/config.py`
   (sample rate 8 kHz, band 200–2,500 Hz, peak 2,250 Hz, burst structure). They come from the
   RPW acoustics literature. Never inline these numbers elsewhere and never "tune" them silently.
2. **Recall on `infested` is the metric that matters most.** A missed infested tree is a dead tree.
   Select and report models on infested-recall + PR-AUC, never raw accuracy alone.
3. **Split by site, never by clip.** Clips from the same tree must never span train/test. The
   pipeline already enforces this — keep it that way; don't switch to a random clip split.
4. **Edge-first.** Inference must be able to run on-device (TFLite) with no connectivity. The cloud
   is for storage, mapping, alerts, retraining. Keep inference behind the `InferenceEngine` interface.
5. **Bilingual, Arabic-first.** All user-facing strings are AR + EN; default locale Arabic, `dir="rtl"`.
   No hard-coded English in the UI.
6. **Ask before adding a dependency** not already in the relevant `requirements.txt` / `package.json`.
   Prefer boring, well-supported libraries — this runs on cheap hardware with bad connectivity.
7. **The dataset is swappable.** Everything downstream of the manifest is dataset-agnostic. Don't
   couple feature/training code to TreeVibes specifics; extend `ingest/` instead.
---
## How to run / test
```bash
# from repo root
cp .env.example .env
make data       # generate synthetic RPW-like audio -> data/raw/manifest.csv
make baseline   # classical RandomForest baseline (CPU, no dataset, fast sanity check)
make train      # CNN transfer-learning (needs TensorFlow + GPU recommended)
make eval       # threshold-aware metrics (PR-AUC, confusion matrix)
make export     # quantize to artifacts/palmguard.tflite + parity check
make api        # FastAPI on :8000 (in-memory DB if Supabase env not set)
make web        # Next.js dashboard on :3000
make test       # ml + api test suites
```
Edge (simulate without hardware): `cd packages/edge && python run.py --sim ../../some_clip.wav --once`
---
## Working style
- **Work in small, reviewable steps.** One module/change per commit; conventional commit messages.
- **Test-first for new logic.** Add/extend a test before implementing. A change isn't done until
  `make test` is green.
- **When you finish a unit, stop and report** what changed, what passed, and what's next — don't
  chain many large changes without a checkpoint.
- **Match existing style.** Python: type hints, `ruff`/`black`, Google-style docstrings.
  TS: strict mode, no `any`.
- **Mirror training in inference.** The edge feature path must use the exact same DSP/feature code
  as training (`palmguard_ml.features`), not a re-implementation.
## Definition of done for the remaining work
- **Real data + CNN:** `TREEVIBES_URL` set in `config.py`, `ingest/treevibes.py` builds a valid
  manifest, CNN trained, `make eval` reports infested-recall + PR-AUC on a held-out **site** split,
  TFLite parity verified.
- **Backend live:** Supabase env wired; schema applied from `app/db/schema.sql`; status engine
  debounce verified by tests; alert dispatch wired to a real provider.
- **Edge on Pi:** runs the hourly capture loop on a Raspberry Pi, survives offline→online without
  losing queued detections.
- **Dashboard:** tree-detail view (timeline + audio player + "mark as treated") and an alerts inbox,
  both AR/EN.
## What NOT to do
- Don't change values in `config.py` to make metrics look better.
- Don't introduce `localStorage`/browser storage hacks; use real state/API.
- Don't commit audio, models, or `.env`.
- Don't replace the site-split with a random split.
- Don't turn the edge classifier into a cloud-only call (breaks the offline guarantee).
- Don't add heavyweight dependencies casually — ask first.
---

*Priority above all: catch infestation early and never silently miss an infested tree.*
