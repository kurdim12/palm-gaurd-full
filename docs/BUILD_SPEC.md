# Palm Guard — Build Specification & Scientific Basis

> The design rationale of record. CLAUDE.md is the day-to-day operating manual;
> this document is *why* the system is shaped the way it is. Decisions here are
> made — don't silently re-derive them.

## 1. Problem

The Red Palm Weevil (*Rhynchophorus ferrugineus*, RPW) is the most destructive
pest of date palms. Larvae feed **inside** the trunk, hidden from view; by the
time external symptoms (wilting fronds, oozing, a hollow crown) appear, the tree
is usually beyond saving and a reservoir for re-infestation. Early detection —
while the infestation is still treatable — is the whole game.

RPW larvae are not silent. Feeding and locomotion inside the trunk produce
characteristic **acoustic emissions**: brief, broadband *bursts* (clicks/feeding
sounds) separated by gaps. A contact (piezo/structure-borne) sensor pressed to
the trunk can pick these up well before visual symptoms. The literature reports:

- Useful signal energy concentrated roughly in the **0.2–2.5 kHz** band.
- A perceptual / spectral emphasis near **~2.25 kHz**.
- Activity organised as **short bursts** (a few ms to a few tens of ms) with
  variable inter-burst intervals (< ~0.25 s), rather than continuous tones.
- Recordings are single-channel at a low rate; **8 kHz is the canonical working
  rate** (matches the TreeVibes corpus — resample sources *down* to it, never up).
- Audio is analysed in **1.0 s windows with 0.5 s hop**; per-window scores are
  aggregated (max-pool, recall-first) to a clip/tree decision.

These numbers drive the fixed constants in
`packages/ml/palmguard_ml/config.py`. They are **scientific constants** — not
hyper-parameters. Tuning them to make a metric look better invalidates the
biological grounding and is explicitly forbidden (CLAUDE.md golden rule #1).

## 2. System architecture

```
Contact sensor → Edge (DSP → features → TFLite inference) → API → DB
                                                              ↓
                                                   Dashboard + Alerts
```

- **Edge** runs the full detection path **on-device**, offline-capable. The
  cloud is for storage, mapping, alerts, and retraining — never a dependency of
  inference (golden rule #4).
- **API** ingests detections, runs the **status engine** (debounce), persists to
  the DB, and dispatches alerts.
- **Dashboard** is Arabic-first, RTL, bilingual; surfaces infested trees on a map
  with a tree-detail view and an alerts inbox.

## 3. Signal processing & features

All DSP/feature code lives in `palmguard_ml.dsp` and `palmguard_ml.features` and
is shared bit-for-bit between training and the edge (golden rule: *mirror
training in inference*). The pipeline:

1. **Mono + resample** to 8 kHz.
2. **Band-pass** (4th-order Butterworth, zero-phase) to 200–2500 Hz to reject
   out-of-band environmental noise; peak-normalise.
3. **Window** into fixed **1.0 s windows, 0.5 s hop** — the classification unit.
4. Two representations from each window:
   - **Feature vector** (10 interpretable scalars): RMS, ZCR, spectral centroid /
     bandwidth / flatness, in-band energy ratio, peak-band (~2.25 kHz) energy
     ratio, and **burst statistics** (rate, mean/std energy) from short-time
     energy thresholding within the literature burst-length window. Used by the
     classical baseline and cheap on-device sanity checks.
   - **Log-mel spectrogram** (`n_fft=1024`, `hop=256`, `n_mels=64`, mel range
     **100–3000 Hz**) — the CNN input.

Window scores aggregate to a file/tree score by **max-pool** (recall-first: a
tree is as infested as its most infested window), mirrored exactly in training
(`evaluate.aggregate_to_files`) and inference (`InferenceEngine.predict`).

## 4. Models

- **Baseline:** `StandardScaler → RandomForest` (class-weighted) on the feature
  vector. CPU-only, fast, no deep-learning deps. A floor that must pass before
  the CNN is trusted.
- **CNN:** compact `small_cnn` by default, or a `mobilenet` transfer-learning
  backbone (`PALMGUARD_BACKBONE`). Sigmoid output = P(infested). Trained
  recall-first (tracks Recall + PR-AUC), class-weighted for the minority class.
- **Export:** the Keras model is quantised to **TFLite** for the edge, with a
  **parity check** asserting TFLite ≈ Keras before the artifact is trusted.

All three are exposed through one `InferenceEngine` interface so the edge, tests,
and cloud share a contract.

## 5. Evaluation policy (non-negotiable)

- **Split by site, never by clip** (golden rule #3). Clips from one tree must not
  span train/test, or metrics leak and overstate performance. The split is
  *stratified by class at the site level* and deterministic.
- **Recall on `infested` is the headline metric**, alongside **PR-AUC**. A missed
  infested tree is a dead tree. Target infested-recall ≥ 0.90
  (`config.TARGET_INFESTED_RECALL`). Accuracy alone is never the selection
  criterion.
- Thresholds are chosen recall-first: the highest threshold that still meets the
  recall target (best precision subject to the recall floor).

## 6. Status engine (debounce)

A single noisy infested hit must **not** flip a tree to `infested` — false alarms
destroy farmer trust as surely as misses destroy trees. Transitions:

- `clean → suspect` on any infested evidence.
- `→ infested` only after **N consecutive** infested detections at/above a
  confidence threshold (defaults: N=3, conf≥0.6; both env-configurable).
- A `clean` detection resets the streak; sub-threshold infested hits don't
  advance it.
- `infested` is **sticky** — only an explicit *treatment* clears it. `treated`
  persists until a confirmed re-infestation.

The engine is a pure function (tree + detection → next tree), so it's fully unit
tested independent of the DB.

## 7. Alerts

Bilingual (AR-first) templates. **Rate-limited + deduped**: one infested tree
cannot raise repeated alerts within the rate-limit window. Transport is
pluggable (`log` default; `twilio`/`whatsapp` when env keys are present).

## 8. Data

The dataset is **swappable** (golden rule #7). Everything downstream of the
manifest is dataset-agnostic. Ingest adapters live in
`palmguard_ml/ingest/`:

- `synthetic.py` — generates RPW-*like* audio (ambient floor + periodic tone-packet
  bursts near 2.25 kHz) so the whole pipeline runs with no download. A fixture,
  not a substitute for real data.
- `treevibes.py` — ingests the real TreeVibes RPW corpus. Labels are
  **folder-level from the published lists** (infested: folders 1–6,11–23; clean:
  7,8,9,10,24,25,35), cross-checked against each folder's `AUDIO` value in the
  annotation CSV (disagreements are counted, never silently resolved); folders
  outside the lists are labelled from `AUDIO`. **SITE = FOLDER number** (one
  folder ≈ one tree) — *not* IMEI, since devices were reused across trees. Every
  clip is accounted for via an `IngestReport` (kept / dropped / disagreements).
  Source: `TREEVIBES_LOCAL` (downloaded archive or folder), `TREEVIBES_KAGGLE`,
  or `TREEVIBES_URL`.

Manifest schema (the contract): `path,label,source,site,sample_rate,duration`.

## 9. What "done" looks like

See CLAUDE.md → *Definition of done*. In short: real TreeVibes CNN with
site-split metrics + verified TFLite parity; Supabase-backed API with a
debounce-tested status engine and a real alert provider; the edge surviving
offline→online on a Pi; and the bilingual dashboard with tree-detail + alerts
inbox.
