---
title: Skin Disease Detection Demo
emoji: 🩺
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🩺 Skin Disease Detection — Demo

A demo Space with four tabs:
- **📷 Quick Image Check** — upload a photo (or click a test image), get a
  classifier prediction plus an LLM-generated plain-language assessment with
  an explicit **Low / Medium / High severity** call.
- **💬 Chat Assistant** — a multi-turn triage conversation. It asks about
  onset, evolution, symptoms, triggers, and history over several exchanges
  before giving a narrowed, severity-rated preliminary read — more targeted
  than a single photo alone could support, without claiming false certainty.
- **📊 Model Performance** — real test-set metrics (accuracy, per-class
  precision/recall/F1, confusion matrix) for whichever fine-tuned model(s)
  are backing the app, pulled from `assets/`.
- **ℹ️ About**

> ⚠️ **This is a research/portfolio demo, not a medical device.** Always see a licensed dermatologist or
> doctor if conditions worsen.

## How it works

| Component | What it does | Default model |
|---|---|---|
| Image classifier | Predicts a skin condition category from a photo | [`Anwarkh1/Skin_Cancer-Image_Classification`](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification) (ViT, 7-class HAM10000) — recommended upgrade: train `finetune_unified_skin_model.ipynb` (24-class, see below) and point `IMAGE_MODEL_ID` at it |
| Second-opinion lesion model *(optional, redundant if using the unified model)* | Re-checks images the primary model flags as an ambiguous melanoma/mole label | Off by default — set `LESION_MODEL_ID` to a HAM10000-based model to enable |
| Chat / explanation | Multi-turn triage with enforced minimum question rounds, turns predictions into plain language | `Qwen/Qwen2.5-7B-Instruct` via the Hugging Face Inference Providers router |

All three model IDs are configurable via Space variables — see below.

## Multi-round chat behavior

The chat tab doesn't just *ask* the model to have a longer conversation —
`app.py` counts user turns explicitly and injects a stage-aware instruction
on every call (`MIN_ROUNDS_BEFORE_ASSESSMENT = 4` by default, adjustable
in code): before that many exchanges, the model is told to keep gathering
information; after, it's told enough rounds have happened and to give its
assessment. Red-flag symptoms override this and get addressed immediately
regardless of turn count.

## Severity levels

Every assessment states one of three levels explicitly:
- 🔴 **High** — cancer-related, ambiguous-but-could-be-serious, systemic/
  autoimmune conditions, or red-flag symptoms. Recommends seeing a doctor soon.
- 🟡 **Medium** — likely needs a proper diagnosis and prescription
  (infections, infestations). Recommends a doctor visit, not urgent.
- 🟢 **Low** — typically manageable with general skin care. Doctor visit
  optional.

Tiering is **keyword-based** (see `classify_tier()` in `app.py`), so it works
across different classifier taxonomies without hardcoding every exact label
string — including correctly flagging DermNet's mixed
*"Melanoma Skin Cancer, Nevi and Moles"* label as High by default, since that
one label can't distinguish the dangerous case from the harmless one.

## 🚀 Deploy in 3 steps

1. **Create a new Space**
   Go to [huggingface.co/new-space](https://huggingface.co/new-space) →
   choose **Gradio** as the SDK → CPU basic hardware is fine for the classifier
   (or ZeroGPU if you want faster inference — see the `@spaces.GPU` decorator
   already applied to `diagnose_image`).

2. **Upload these files**
   Upload `app.py`, `requirements.txt`, this `README.md`, the `examples/`
   folder (if you generated test images), and the `assets/` folder (if you
   exported model performance metrics — see below), either via the web UI
   ("Add file") or:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
   cp -r app.py requirements.txt README.md examples assets <your-space-name>/
   cd <your-space-name>
   git add . && git commit -m "Initial commit" && git push
   ```

3. **Add an `HF_TOKEN` secret (to enable the chatbot)**
   In your Space → **Settings → Variables and secrets** → **New secret**:
   - Name: `HF_TOKEN`
   - Value: a Hugging Face access token (create one at
     [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
     "Read" scope is enough)

   Without this secret, image classification still works — only the
   LLM-generated explanations and the chat tab need the token.

That's it — the Space will build and the app will be live.

## 🖼️ Adding test images (optional but recommended)

So users have something to click without needing their own photo, generate a
few sample images once, locally, before you push:

```bash
pip install datasets pillow
python scripts/download_examples.py
```

This saves a handful of JPEGs into `examples/`. Include that folder when you
push to your Space. You can also just drop in your own sample `.jpg`/`.png`
files instead.

## ⚙️ Configuration

Set these as Space **variables** (not secret, unless noted) to customize:

- `IMAGE_MODEL_ID` — any Hugging Face image-classification model compatible
  with `transformers.pipeline("image-classification", ...)`.
- `LESION_MODEL_ID` *(optional)* — a second, pigmented-lesion-specific model
  (like the HAM10000 fine-tune below) used automatically as a second opinion
  when the primary model's top prediction is an ambiguous melanoma/mole label.
  Leave unset to disable.
- `CHAT_MODEL_ID` — any chat-completion-capable model available via the HF
  Inference Providers router.
- `HF_TOKEN` *(secret)* — required for the chat tab and the LLM explanations.

## 🧠 Fine-tuning your own model

Three self-contained Colab notebooks are included:

### `finetune_unified_skin_model.ipynb` — ⭐ recommended: 24-class unified model
- **Retrained from scratch**, not resumed — fresh weights each run
- Merges DermNet's 22 clean categories with HAM10000's melanoma and nevi
  as their own separate classes, **replacing DermNet's ambiguous combined
  "Melanoma Skin Cancer, Nevi and Moles" label entirely** — melanoma finally
  gets a clean training signal instead of being merged with benign moles
- Early stopping (up to 14 epochs, stops automatically once validation
  macro-F1 plateaus) + cosine LR schedule — addresses the previous
  DermNet-only run's validation loss not having converged at 6 epochs
- Melanoma threshold-tuning (now meaningful, since melanoma is unambiguous)
- Exports `unified_metrics.json` + `unified_confusion_matrix.png`
- **Honest tradeoff:** melanoma/nevi images come from HAM10000's
  dermatoscope close-ups, not regular phone photos like the rest of the
  dataset — a real domain gap, flagged in the notebook rather than hidden

### `finetune_skin_lesion_model.ipynb` — 7-class pigmented lesions (HAM10000 only)
- Useful if you want a smaller, faster, cancer-focused model specifically
- class-weighted loss, melanoma recall tracking, threshold tuning
- exports `ham10000_metrics.json` + `ham10000_confusion_matrix.png`

### `finetune_dermnet_23class.ipynb` — 23-class DermNet only (superseded by the unified notebook above, kept for reference)
- Same as the unified notebook's DermNet portion, but keeps DermNet's
  original ambiguous melanoma/nevi label instead of replacing it
- exports `dermnet_metrics.json` + `dermnet_confusion_matrix.png`

**To use any of them:** upload the `.ipynb` to
[Google Colab](https://colab.research.google.com), set **Runtime → Change
runtime type → T4 GPU**, and run cells top to bottom. The unified notebook
takes roughly 60–100 minutes on a free T4 depending on when early stopping
kicks in; keep the tab active so the session doesn't disconnect.

**Before treating any of these as more than a research project**, see the
"Next steps" section at the end of each notebook — a higher benchmark score
is not the same as a clinically validated model.

## 📊 Populating the Model Performance tab

After running a fine-tuning notebook, download the two files it exports
(`*_metrics.json` and `*_confusion_matrix.png`) from Colab's file browser and
place them in this Space's `assets/` folder, then redeploy. The tab picks up
any matching pair automatically — you can have both the HAM10000 and DermNet
reports side by side if you've fine-tuned both.

## Limitations & responsible use

- Neither bundled classifier has been clinically validated — treat their
  output as a talking point, not a result.
- Performance depends heavily on image quality, lighting, and skin tone
  representation in the training data; public dermatology datasets skew
  toward lighter skin tones.
- DermNet's melanoma/nevi label limitation (above) means that category's
  severity is intentionally conservative rather than precise.
- This app must not be used as a substitute for professional medical
  evaluation. The chatbot is instructed to narrow toward likely categories
  and state a severity level, but never to claim certainty, prescribe
  treatment, or give dosages.

## Local development

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here   # optional, for chat
python app.py
```
