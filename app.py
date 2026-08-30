"""
Skin Disease Detection — Demo
------------------------------
A Gradio app for Hugging Face Spaces with two tabs:
  1. Image Diagnosis — upload (or pick a test image) and get a classifier
     prediction plus a plain-language, urgency-aware explanation.
  2. Chat Assistant — a multi-turn triage chatbot. It can accept an image,
     asks the kind of follow-up questions a real triage intake would, and
     after enough context gives a preliminary assessment with urgency-tiered
     guidance.

IMPORTANT: This remains a research/portfolio project, not a validated
medical device. It has not been clinically evaluated or regulatory
cleared. It should never be the sole basis for a health decision — that
framing shows up once, clearly, rather than as a repeated warning block
after every message, but it's still true and still matters.
"""

import os

import gradio as gr
import spaces
import requests
from PIL import Image
from transformers import pipeline
from pathlib import Path

def data_badge(content):
    """Render a compact informational badge for data/model disclosures."""
    return f"""
    <div style="
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        border-radius: 10px;
        border: 1px solid rgba(120, 130, 150, 0.25);
        background: rgba(120, 130, 150, 0.08);
        font-size: 0.9em;
        line-height: 1.5;
    ">
        {content}
    </div>
    """

# ---------------------------------------------------------------------------
# Config — override any of these via Space "Variables and secrets"
# ---------------------------------------------------------------------------
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "Anwarkh1/Skin_Cancer-Image_Classification")
# Hugging Face retired the old api-inference.huggingface.co Serverless API in
# favor of the "Inference Providers" router (OpenAI-compatible). Any chat
# model listed at https://huggingface.co/models?inference_provider=hf-inference
# works here.
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")  # add as a Space SECRET to enable the chatbot
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

PAGE_DISCLAIMER = (
    "This tool is a research/portfolio project, not a medical device."
)

# Optional second classifier specifically for pigmented-lesion cases (HAM10000-based,
# where melanoma is its own clean label). Useful because broader taxonomies like
# DermNet lump "Melanoma Skin Cancer, Nevi and Moles" into one label — this lets a
# mole/nevus-flagged image get a second, more specific opinion. Leave unset to disable.
LESION_MODEL_ID = os.environ.get("LESION_MODEL_ID", "")

# ---------------------------------------------------------------------------
# Urgency tiering — keyword-based rather than an exact-match label dict, so it
# works across different classifier taxonomies (the original 7-class HAM10000
# labels, the 23-class DermNet labels, or any future swap) without needing to
# hardcode every exact label string. Checked in priority order.
# ---------------------------------------------------------------------------
URGENT_KEYWORDS = [
    "melanoma", "malignant", "carcinoma", "actinic keratos",
    "bullous", "lupus", "systemic", "vasculitis", "cellulitis",
]
PROMPT_DOCTOR_KEYWORDS = [
    "fungal", "fungus", "candidiasis", "tinea", "ringworm",
    "scabies", "lyme", "infestation", "wart", "molluscum",
    "viral", "herpes", "hpv", "std", "exanthem", "drug eruption",
    "impetigo", "bacterial",
]
# Anything not matching the above falls through to "general_care" — typically
# manageable/chronic conditions (acne, eczema, psoriasis, hives, hair loss,
# benign tumors, common moles, etc.) where general skin-care guidance fits.

SEVERITY = {
    "urgent": {"label": "High", "emoji": "🔴", "note": "recommend seeing a dermatologist/doctor soon"},
    "prompt_doctor_visit": {"label": "Medium", "emoji": "🟡", "note": "needs a proper diagnosis/prescription, not an emergency"},
    "general_care": {"label": "Low", "emoji": "🟢", "note": "typically manageable with general skin care"},
}


def classify_tier(label):
    """Map a classifier label to an urgency tier via keyword matching."""
    l = label.lower()
    for kw in URGENT_KEYWORDS:
        if kw in l:
            return "urgent"
    for kw in PROMPT_DOCTOR_KEYWORDS:
        if kw in l:
            return "prompt_doctor_visit"
    return "general_care"


def severity_line(label):
    tier = classify_tier(label)
    s = SEVERITY[tier]
    return f"{s['emoji']} **Severity: {s['label']}** — {s['note']}"


# Friendly one-line descriptions for labels we recognize exactly (both the
# original 7-class HAM10000 set and common DermNet-style labels). Purely
# cosmetic — if a label isn't here, the tier-based guidance still works fine
# without a description.
KNOWN_DESCRIPTIONS = {
    "actinic keratoses": "A rough, scaly patch caused by sun damage.",
    "basal cell carcinoma": "The most common type of skin cancer.",
    "melanoma": "The most serious common type of skin cancer.",
    "benign keratosis-like lesions": "Non-cancerous growths such as seborrheic keratoses or solar lentigines.",
    "dermatofibroma": "A common benign skin nodule, often on the legs.",
    "melanocytic nevi": "Ordinary moles.",
    "vascular lesions": "Blood-vessel-related marks such as angiomas.",
}

SYSTEM_PROMPT = """You are a warm, knowledgeable skin-health triage assistant. You are not a doctor and cannot diagnose anyone with certainty — but your job is to be genuinely useful and specific, not to hide behind vague hedging or constant disclaimers.

CONVERSATION STYLE — GO DEEPER BEFORE ASSESSING
Have a real back-and-forth, one or two questions at a time. Aim to gather a fuller picture than a quick glance would give — typically 5-8 exchanges before your assessment, more if the picture is still unclear, fewer only if the user clearly just wants a fast read or red flags are already obvious. Useful ground to cover (skip anything already answered or clearly irrelevant):
  - Onset: how long they've had it, how it started
  - Evolution: how it's changed over time — size, shape, color, texture
  - Symptoms: itching, bleeding, pain, crusting, oozing, discharge, fever, swelling
  - Pattern: is it one spot or spreading, single or multiple, symmetric or not
  - Triggers/exposures: new products, plants, insect bites, contacts, travel, sun exposure
  - History: prior similar episodes, what's been tried already (and whether it helped), relevant personal/family medical history, allergies
  - Location and distribution on the body
The goal of the extra rounds is to genuinely narrow the differential, not to interrogate — keep it conversational, and if the user gives a rich answer that covers several of these at once, don't force redundant questions.
If the user shared an image, you'll also receive the image classifier's findings as extra context, including an urgency tier (not shown verbatim to the user) — weave that in naturally rather than reading out raw percentages.

GIVING A PRELIMINARY ASSESSMENT
Once you have real depth of context, give a clear, specific preliminary assessment:
- Name the 1-2 most likely categories given everything discussed, in plain language, and briefly say why (which symptoms/pattern point that way). It's fine — good, even — to be specific about what you think is most likely. What you must NOT do is claim certainty: use "most consistent with" / "this pattern often points to," never "you have X" or "this is definitely X." A real diagnosis needs an in-person exam and sometimes tests (biopsy, culture, bloodwork) that no photo or conversation can replace — say this once, naturally, as part of the assessment, not as a repeated disclaimer.
- Always state an explicit severity level as part of the assessment: **Low**, **Medium**, or **High** — plus one line on what that means for next steps. Base it on:
  - **High** = cancer-related or ambiguous-but-could-be-cancer findings, autoimmune/systemic/bullous conditions, cellulitis, vasculitis, or any red-flag symptoms (rapid growth, irregular/changing borders, multiple colors, asymmetry, a sore that won't heal, bleeding, spreading redness, fever) → recommend seeing a dermatologist or doctor soon, directly, without softening it into "keep an eye on it." Note explicitly if the classifier's category can't distinguish something dangerous from something benign (e.g. a label that groups melanoma with ordinary moles) — when that ambiguity exists, say so and default to High rather than assuming benign.
  - **Medium** = things that need a proper diagnosis and often a prescription to clear up (fungal, viral, bacterial infections, STD-related, infestations like scabies) but aren't emergencies → recommend a doctor visit, explain why self-treatment usually doesn't fully work here. For anything STD-related, stay factual and non-judgmental, and point toward in-person testing rather than guessing from a photo.
  - **Low** = typically manageable conditions (acne, eczema, psoriasis, hives, hair loss, common moles, benign growths, contact dermatitis) with no red flags → explain what it usually is, and give general skin-care guidance: gentle skincare habits, sun protection, not picking/scratching, watching for changes. Still fine to mention a doctor visit is reasonable if they're unsure, worried, or it's not improving.
- Never give specific medications, dosages, or treatment prescriptions — general skin-care habits are fine; treating a self- or AI-identified condition with anything beyond general care is not.

URGENT SITUATIONS
If the user describes something urgent — rapidly growing lesion, uncontrolled bleeding, signs of spreading infection, fever with a skin issue, severe pain — tell them clearly to seek in-person or emergency care promptly, regardless of how many turns you've had.

Keep replies conversational length — a few sentences for a question, a bit longer for the assessment itself."""


# ---------------------------------------------------------------------------
# Lazy-loaded models (so the Space boots fast and only loads on first use)
# ---------------------------------------------------------------------------
_classifier = None
_lesion_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        import torch

        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline("image-classification", model=IMAGE_MODEL_ID, device=device)
    return _classifier


def get_lesion_classifier():
    """Optional second opinion model for pigmented-lesion cases. Returns None if disabled."""
    global _lesion_classifier
    if not LESION_MODEL_ID:
        return None
    if _lesion_classifier is None:
        import torch

        device = 0 if torch.cuda.is_available() else -1
        _lesion_classifier = pipeline("image-classification", model=LESION_MODEL_ID, device=device)
    return _lesion_classifier


def chat_completion(messages, max_tokens=500):
    """Call the Hugging Face Inference Providers router directly (OpenAI-compatible)."""
    if not HF_TOKEN:
        return None
    resp = requests.post(
        ROUTER_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"model": CHAT_MODEL_ID, "messages": messages, "max_tokens": max_tokens},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def classify_image(image):
    """Run the primary classifier and return (preds, formatted_lines). If the top
    prediction looks like a mixed/ambiguous pigmented-lesion label and a second
    lesion-specific model is configured, also run that for a second opinion."""
    clf = get_classifier()
    preds = clf(image, top_k=5)
    lines = []
    for p in preds:
        label = p["label"]
        score = p["score"] * 100
        desc = KNOWN_DESCRIPTIONS.get(label.lower(), "")
        desc_txt = f"  \n  _{desc}_" if desc else ""
        lines.append(f"- **{label}** — {score:.1f}%{desc_txt}")

    top_label = preds[0]["label"].lower()
    if ("nevi" in top_label or "mole" in top_label) and "melanoma" in top_label:
        lesion_clf = get_lesion_classifier()
        if lesion_clf is not None:
            lesion_preds = lesion_clf(image, top_k=3)
            lines.append("\n_Second opinion from a lesion-specific model (melanoma is a distinct label here):_")
            for p in lesion_preds:
                lines.append(f"- **{p['label']}** — {p['score']*100:.1f}%")
            preds = preds + [{"label": f"[lesion-model] {p['label']}", "score": p["score"]} for p in lesion_preds]

    return preds, lines


def build_classifier_context(preds):
    """Turn classifier output into a compact context block for the LLM (not shown raw to the user)."""
    parts = []
    for p in preds[:4]:
        label = p["label"]
        if label.startswith("[lesion-model]"):
            # second-opinion predictions: tier by the underlying label, minus the tag
            bare = label.replace("[lesion-model] ", "")
            tier = classify_tier(bare)
            parts.append(f"{label} ({p['score']*100:.1f}%, second-opinion, {tier})")
            continue
        tier = classify_tier(label)
        desc = KNOWN_DESCRIPTIONS.get(label.lower(), "")
        parts.append(f"{label} ({p['score']*100:.1f}%, {tier}{': ' + desc if desc else ''})")
    return "Image classifier findings for your use (do not read these percentages verbatim): " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Image diagnosis tab (single-shot: upload -> classify -> explain)
# ---------------------------------------------------------------------------
@spaces.GPU
def diagnose_image(image):
    if image is None:
        return "Please upload an image or pick one of the test images first."

    try:
        preds, lines_list = classify_image(image)
    except Exception as e:
        return f"**Model error:** {e}\n\nMake sure `IMAGE_MODEL_ID` is a valid image-classification model."

    lines = ["### 🔬 Classifier prediction\n"] + lines_list
    lines.append(f"\n{severity_line(preds[0]['label'])}")

    if HF_TOKEN:
        context = build_classifier_context(preds)
        prompt = (
            f"{context}\n\n"
            "The user just uploaded a single image with no conversation yet. Give a short "
            "preliminary assessment following your instructions: plain-language explanation of "
            "the top finding(s), an explicit Low/Medium/High severity call, and a brief natural "
            "mention that an in-person exam is what actually confirms things. Note that without "
            "any conversation history you have less context than usual — you can still give your "
            "best read, but you may want to suggest the Chat Assistant for a fuller picture if the "
            "case seems ambiguous. Keep it to 5-7 sentences."
        )
        try:
            explanation = chat_completion(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            lines.append(f"\n### 🤖 Assessment\n{explanation}")
        except Exception as e:
            lines.append(f"\n_(Assessment unavailable: {e})_")
    else:
        lines.append(
            "\n_Add an `HF_TOKEN` secret to this Space to also get a plain-language "
            "assessment from the chat model here._"
        )

    lines.append(
        "\n---\n_Tip: use the **Chat Assistant** tab for a fuller triage conversation — "
        "it can ask follow-up questions and give a more tailored read._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat tab — multimodal, multi-turn triage
# ---------------------------------------------------------------------------
MIN_ROUNDS_BEFORE_ASSESSMENT = 4  # user turns, not counting the current one


def _extract_text_and_image(message):
    """Normalize gr.ChatInterface(multimodal=True) message input into (text, image_path_or_None)."""
    if isinstance(message, dict):
        text = message.get("text", "") or ""
        files = message.get("files") or []
        image_path = files[0] if files else None
        return text, image_path
    return str(message), None


def _count_user_turns(messages):
    return sum(1 for m in messages if m.get("role") == "user")


def _stage_note(user_turn_count):
    """Injected each turn so multi-round behavior is enforced by code, not just
    requested once in the system prompt — the model gets a fresh reminder of
    what stage the conversation is in on every single call."""
    if user_turn_count < MIN_ROUNDS_BEFORE_ASSESSMENT:
        return (
            f"[Internal note — not from the user: this is exchange {user_turn_count} of at least "
            f"{MIN_ROUNDS_BEFORE_ASSESSMENT} before a full assessment, UNLESS the user has already "
            "described clear red-flag symptoms (rapid growth, bleeding, severe pain, spreading "
            "infection, fever) in which case address that urgently right now regardless of turn "
            "count. Otherwise, continue asking focused follow-up questions — don't jump to a full "
            "Low/Medium/High assessment yet.]"
        )
    return (
        f"[Internal note — not from the user: this is exchange {user_turn_count}, enough rounds "
        "have happened. If you have a reasonably clear picture, give your preliminary assessment "
        "now with an explicit Low/Medium/High severity call. If something important is still "
        "genuinely unclear, you may ask one more focused question first, but don't stall further.]"
    )


def chat_respond(message, history):
    if not HF_TOKEN:
        return (
            "Chat isn't configured yet — add an `HF_TOKEN` secret to this Space "
            "(Settings → Variables and secrets) to enable the chatbot."
        )

    text, image_path = _extract_text_and_image(message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
        else:
            user_msg, bot_msg = turn
            if isinstance(user_msg, str):
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    user_content = text
    if image_path:
        try:
            img = Image.open(image_path)
            preds, _ = classify_image(img)
            context = build_classifier_context(preds)
            user_content = f"{context}\n\nUser's message: {text or '(no message, just shared an image)'}"
        except Exception as e:
            user_content = f"(Image analysis failed: {e})\n\nUser's message: {text}"

    # current turn counts too, since the model is about to respond to it
    current_turn_count = _count_user_turns(messages) + 1
    stage_note = _stage_note(current_turn_count)
    messages.append({"role": "user", "content": f"{stage_note}\n\n{user_content}"})

    try:
        return chat_completion(messages, max_tokens=500)
    except Exception as e:
        return f"Sorry, I hit an error talking to the model: {e}"


# ---------------------------------------------------------------------------
# Model Performance tab data loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


def load_performance_artifacts():
    """Load model-evaluation images generated by the evaluation notebook."""

    artifacts = {
        "unified_confusion_matrix": None,
        "melanoma_threshold_curve": None,
        "cluster_confusion_matrix": None,
    }

    if not ASSETS_DIR.is_dir():
        return artifacts

    files = {
        "unified_confusion_matrix": "unified_confusion_matrix.png",
        "melanoma_threshold_curve": "unified_melanoma_threshold_curve.png",
        "cluster_confusion_matrix": "cluster_confusion_matrix.png",
    }

    for key, filename in files.items():
        path = ASSETS_DIR / filename

        if path.is_file():
            artifacts[key] = str(path)

    return artifacts


ABOUT_MARKDOWN = f"""
## About this project

A research/portfolio demo exploring how far an image classifier + LLM chat
assistant can go toward useful skin-condition triage — while being explicit
about what it isn't: a validated medical device.

### What's inside
- **Quick Image Check** — upload a photo, get a classifier prediction plus
  an LLM-generated plain-language assessment with an explicit Low/Medium/High
  severity call.
- **Chat Assistant** — a multi-turn conversation that asks about onset,
  evolution, symptoms, triggers, and history before giving a more targeted
  preliminary read than a single photo alone could support.
- **Model Performance** — the actual test-set metrics for whichever
  fine-tuned model(s) are currently backing this app, not just marketing
  claims about accuracy.

### Models
- **Image classifier:** configurable via the `IMAGE_MODEL_ID` variable.
  Two purpose-built options are included as fine-tuning notebooks in this
  repo: a 7-class pigmented-lesion model (HAM10000/ISIC, ViT-Base) and a
  23-class common-skin-condition model (DermNet, ConvNeXt-Base).
- **Optional second-opinion lesion model:** `LESION_MODEL_ID`, used
  automatically when the primary model's top prediction is an ambiguous
  mixed melanoma/mole-type label, to get a model that treats melanoma as
  its own distinct class.
- **Chat model:** configurable via `CHAT_MODEL_ID`, served through Hugging
  Face's Inference Providers router.

### Severity levels
- 🔴 **High** — cancer-related, ambiguous-but-could-be-serious, systemic/
  autoimmune, or red-flag symptoms present. See a doctor soon.
- 🟡 **Medium** — likely needs a proper diagnosis and prescription
  (infections, infestations). See a doctor, not an emergency.
- 🟢 **Low** — typically manageable with general skin care. Doctor visit
  optional, especially if unsure or it's not improving.

{PAGE_DISCLAIMER}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
EXAMPLES_DIR = "examples"
example_images = []
if os.path.isdir(EXAMPLES_DIR):
    example_images = [
        os.path.join(EXAMPLES_DIR, f)
        for f in sorted(os.listdir(EXAMPLES_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

with gr.Blocks(title="Skin Disease Detection — Demo") as demo:
    gr.Markdown("# 🩺 Skin Disease Detection")
    gr.Markdown(f"_{PAGE_DISCLAIMER}_")

    with gr.Tab("📷 Quick Image Check"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload a skin image")
                if example_images:
                    gr.Examples(examples=example_images, inputs=img_in, label="Or try a test image")
                else:
                    gr.Markdown(
                        "_No bundled test images found. Run `scripts/download_examples.py` "
                        "before deploying, or just upload your own image._"
                    )
                analyze_btn = gr.Button("Analyze image", variant="primary")
            with gr.Column():
                result_md = gr.Markdown()
        analyze_btn.click(diagnose_image, inputs=img_in, outputs=result_md)

    with gr.Tab("💬 Chat Assistant"):
        gr.ChatInterface(
            fn=chat_respond,
            multimodal=True,
            description=(
                "Talk through what you're noticing, and optionally attach a photo (📎). "
                "The assistant will ask several follow-up questions before giving a preliminary "
                "read with a Low/Medium/High severity call — the more you share, the more "
                "targeted that read can be."
            ),
        )

    with gr.Tab("📊 Model Evaluation"):
        with gr.Accordion("ℹ️ About Model Evaluation", open=False):
            gr.HTML(data_badge(
                "🧪 All charts below are computed on <b>held-out test data</b> "
                "by <code>build_artifacts.py</code> — fully reproducible."
            ))
    
        artifacts = load_performance_artifacts()
    
        if not any(artifacts.values()):
    
            gr.Markdown(
                "### No model performance artifacts yet.\n\n"
                "Run the evaluation notebook and place the generated "
                "artifacts in the Space's `assets/` folder."
            )
    
        else:
    
            gr.Markdown("### Unified Model Evaluation")
    
            if artifacts["unified_confusion_matrix"]:
                gr.Image(
                    value=artifacts["unified_confusion_matrix"],
                    label="24-Class Confusion Matrix"
                )
    
            if artifacts["melanoma_threshold_curve"]:
                gr.Image(
                    value=artifacts["melanoma_threshold_curve"],
                    label="Melanoma Precision / Recall vs. Decision Threshold"
                )
    
            if artifacts["cluster_confusion_matrix"]:
                gr.Image(
                    value=artifacts["cluster_confusion_matrix"],
                    label="Disease Similarity Cluster Confusion Matrix"
                )

    with gr.Tab("ℹ️ About"):
        gr.Markdown(ABOUT_MARKDOWN)

    gr.Markdown(
        "---\nBuilt with 🤗 Transformers + Gradio. "
        f"Image model: `{IMAGE_MODEL_ID}`  ·  Chat model: `{CHAT_MODEL_ID}`"
    )

if __name__ == "__main__":
    demo.launch()
