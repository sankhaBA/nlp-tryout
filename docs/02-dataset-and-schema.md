# 02 — Dataset & Schema

> **Read this when** you need the exact `input`/`target` grammar, the action/direction
> vocabulary, the dataset version history, or details of the held-out evaluation set. This is
> the **data contract** the model learns and the app must produce.

Source files:
[data/](../data/) (training batches) ·
[evaluation_tflite/dataset/nav_eval_android_v1.csv](../evaluation_tflite/dataset/nav_eval_android_v1.csv) (eval set)

---

## CSV format

Every dataset is a 2-column CSV with header `input,target`:

```csv
input,target
action: continue distance: 10 steps,Walk forward for 10 steps.
action: turn direction: right distance: 15 steps,"In 15 steps, turn right."
action: stop landmark: information desk,Stop. You are at the information desk.
action: continue distance: 45 steps landmark: Sephora,Walk straight for 45 steps to reach Sephora.
```

- `input` — a **structured command** (a flat key-space string, see grammar below).
- `target` — the **natural-language instruction**. Quoted with `"`only when it contains a comma
  (standard CSV quoting); otherwise bare.

---

## `input` grammar (the command contract)

```
action: <action> [direction: <direction>] [distance: <N> steps] [landmark: <name>]
```

Fields always appear in this order; optional fields are omitted (not left blank) when not
applicable.

| Field | Required | Vocabulary / form | Notes |
|-------|----------|-------------------|-------|
| `action` | **always** | `continue` · `turn` · `stop` · `board` · `exit` | the 5-verb action space |
| `direction` | optional | `left` · `right` · `straight` · `up` · `down` | `up`/`down` used with floor-changers |
| `distance` | optional | `<integer> steps` | **steps only**, never metres |
| `landmark` | optional | free text (shop / facility / floor-changer) | only a destination or a floor-changer (see [01](01-data-generation.md)) |

### Action semantics

| Action | Meaning | Typical fields |
|--------|---------|----------------|
| `continue` | walk forward a distance, optionally to a destination | `distance`, sometimes `direction: straight`, optional `landmark` |
| `turn` | change heading | `direction: left/right`, optional `distance` (turn after N steps), optional `landmark` |
| `stop` | halt, usually at arrival | optional `landmark`, sometimes `distance: 0 steps` |
| `board` | step onto a floor-changer | `direction: up/down`, `landmark: escalator/elevator/stairs/…`, optional `distance` (stairs) |
| `exit` | step off a floor-changer and proceed | `direction`, `distance`, `landmark: elevator/escalator/stairs` |

### Example input → target pairs (from the eval set)

```
action: turn direction: left distance: 8 steps landmark: escalator   → "In 8 steps, turn left towards the escalator."
action: board direction: up landmark: escalator                      → "Step onto the escalator going up."
action: exit direction: straight distance: 9 steps landmark: elevator → "Exit the elevator and walk straight for 9 steps."
action: stop landmark: prayer room                                   → "Stop here. You have reached the prayer room."
```

---

## `target` style

Targets are short imperative sentences (typically 4–12 words), screen-reader friendly:

- **`continue`** → "Walk straight for N steps." / "Continue forward for N steps." / "…to reach <landmark>."
- **`turn`** → "In N steps, turn left." / "Turn right towards the <landmark>."
- **`stop`** → "Stop here." / "Stop. You have reached the <landmark>."
- **`board`** → "Step onto the escalator going up." / "Enter the elevator and go up."
- **`exit`** → "Exit the elevator and walk straight for N steps."

Note that **multiple valid phrasings exist** for the same command ("Walk forward" vs "Walk
straight" vs "Go straight"). This is intentional for natural variety, but it is also why
**exact-match accuracy is low (~35%) while ROUGE is high (~0.80)** — the model picks *a* correct
phrasing that often differs lexically from the single reference. See
[04-rouge-evaluation.md](04-rouge-evaluation.md).

---

## Training dataset version history (`data/`)

Generation was done in batches (see [01](01-data-generation.md)); each batch is a cumulative,
curated snapshot. **`nav_dataset_b4.3.csv` is the production training set** referenced by the
training notebook.

| File | Role |
|------|------|
| [nav_dataset_b1.csv](../data/nav_dataset_b1.csv) | initial seed batch |
| [nav_dataset_b2.csv](../data/nav_dataset_b2.csv) | expansion |
| [nav_dataset_b3.csv](../data/nav_dataset_b3.csv) | expansion |
| [nav_dataset_b4.1.csv](../data/nav_dataset_b4.1.csv) | batch 4 variant |
| [nav_dataset_b4.2.csv](../data/nav_dataset_b4.2.csv) | batch 4 variant |
| [nav_dataset_b4.3.csv](../data/nav_dataset_b4.3.csv) | **production** — used to train `b4.3.2` |

> Early batches (e.g. `b1`) contain some **metre-based** distances and `reason:`/`description:`
> fields from an earlier schema draft. The current contract standardized on **steps** and the
> four fields above; later batches reflect that.

> To get exact row counts and duplicate stats for any batch, run the validator:
> `python dataset_generator/dataset_validator.py --file data/nav_dataset_b4.3.csv`.

---

## Held-out evaluation set

[evaluation_tflite/dataset/nav_eval_android_v1.csv](../evaluation_tflite/dataset/nav_eval_android_v1.csv)
is a **separate, curated 196-sample** test set used by the TFLite/on-device evaluation (it is
*not* the notebook's random test split). It is organized by action type — blocks of `continue`,
`turn`, `stop`, `board`, `exit` with varied distances and landmarks — to give per-action coverage.
`config.default_dataset()` auto-selects the latest `*.csv` in that folder.

This is the dataset behind the published numbers in
[tflite_eval_results.json](../tflite_eval_results.json) and
[evaluation_tflite/results/](../evaluation_tflite/results/) — see
[05-tflite-and-on-device.md](05-tflite-and-on-device.md).
