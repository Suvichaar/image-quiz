import io
import json
import re
from datetime import datetime

import streamlit as st
from PIL import Image

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="OCR → GPT Structuring (Quiz JSON)",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 OCR → GPT-4 Structuring → JSON")
st.caption("Upload a quiz image (Hindi), run Azure Document Intelligence OCR, structure with Azure OpenAI (GPT), and download JSON.")

# ---------------------------
# Secrets / Config (from st.secrets)
# ---------------------------
try:
    AZURE_DI_ENDPOINT = st.secrets["AZURE_DI_ENDPOINT"]  # e.g., https://<your-di>.cognitiveservices.azure.com/
    AZURE_API_KEY     = st.secrets["AZURE_API_KEY"]

    AZURE_OPENAI_ENDPOINT    = st.secrets["AZURE_OPENAI_ENDPOINT"]  # e.g., https://<your-openai>.openai.azure.com/
    AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_API_KEY     = st.secrets.get("AZURE_OPENAI_API_KEY", AZURE_API_KEY)  # reuse if same
    GPT_DEPLOYMENT           = st.secrets.get("GPT_DEPLOYMENT", "gpt-4")
except Exception:
    st.error("Missing secrets. Please set AZURE_DI_ENDPOINT, AZURE_API_KEY, AZURE_OPENAI_ENDPOINT, and GPT deployment details in secrets.")
    st.stop()

# ---------------------------
# Clients
# ---------------------------
di_client = DocumentIntelligenceClient(
    endpoint=AZURE_DI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY)
)

gpt_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PROMPT = """
You are an assistant that receives extracted Hindi quiz text containing multiple questions,
each with four options labeled (A)-(D), a correct answer indicated by व्याख्या (X), and an explanation.
Return a JSON object with key "questions" mapping to a list of objects, each having:
- question: string
- options: { "A": ..., "B": ..., "C": ..., "D": ... }
- correct_option: one of "A","B","C","D"
- explanation: string
Ensure the JSON is valid and includes all questions.
"""

# ---------------------------
# Helpers
# ---------------------------
def ocr_extract(image_bytes: bytes) -> str:
    poller = di_client.begin_analyze_document(
        model_id="prebuilt-read",
        body=image_bytes
    )
    result = poller.result()
    if getattr(result, "paragraphs", None):
        return "\n".join([p.content for p in result.paragraphs]).strip()
    if getattr(result, "content", None):
        return result.content.strip()
    lines = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            if getattr(line, "content", None):
                lines.append(line.content)
    return "\n".join(lines).strip()

def clean_model_json(txt: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return txt.strip()

def structure_with_gpt(raw_text: str) -> dict:
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": raw_text}
        ],
    )
    content = resp.choices[0].message.content
    content = clean_model_json(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise

def to_download_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

# ===========================
# 🔖 Tabs
# ===========================
tab1, tab2, tab3 = st.tabs([
    "Tab 1: OCR → JSON",
    "Tab 2: JSON → Placeholder JSON",
    "Tab 3: (placeholder)"
])

# ===========================
# ✅ Tab 1 content (your UI)
# ===========================
with tab1:
    uploaded = st.file_uploader("📎 Upload quiz image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        # Read bytes once for both preview and OCR (prevents empty reads)
        image_bytes = uploaded.getvalue()

        # Preview image safely from bytes
        try:
            preview_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(preview_img, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.warning("Could not preview image. Processing anyway...")

        with st.spinner("🔍 Running OCR..."):
            try:
                raw_text = ocr_extract(image_bytes)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()

        if not raw_text.strip():
            st.error("OCR returned empty text. Try a clearer image.")
            st.stop()

        with st.expander("📄 OCR Text (preview)"):
            st.text(raw_text[:4000] if len(raw_text) > 4000 else raw_text)

        with st.spinner("🤖 Structuring with GPT..."):
            try:
                quiz_json = structure_with_gpt(raw_text)
            except Exception as e:
                st.error(f"GPT structuring failed: {e}")
                st.stop()

        st.success("✅ Done!")
        st.subheader("Structured JSON")
        st.json(quiz_json)

        fname = f"quiz_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            label="⬇️ Download JSON",
            data=to_download_bytes(quiz_json),
            file_name=fname,
            mime="application/json"
        )

    else:
        st.info("Upload a JPG/PNG to begin.")

    st.markdown("---")
    with st.expander("⚙️ Setup Notes"):
        st.markdown(
            """
**How this works**

1) Azure Document Intelligence `"prebuilt-read"` extracts Hindi text from your image.  
2) Azure OpenAI (your deployment name in `GPT_DEPLOYMENT`) converts that into a normalized quiz JSON.  

**Safety/Quality Tips**
- Use high-res, well-lit images.
- If options aren’t clearly marked (A–D), adjust the prompt in `SYSTEM_PROMPT`.
- Never commit secrets to GitHub; use Streamlit secrets.
            """
        )
# ===========================
# ✅ Tab 2: JSON → Web Story Placeholders
# ===========================
with tab2:
    st.subheader("JSON → Web Story Placeholders")

    st.caption("Upload the structured questions JSON from Tab 1 (key: `questions`) and I’ll flatten it into web-story placeholders.")

    uploaded_json = st.file_uploader("📎 Upload ‘quiz_structured_*.json’", type=["json"], key="tab2_uploader")

    if uploaded_json:
        # Read & preview input JSON
        try:
            questions_data = json.loads(uploaded_json.getvalue().decode("utf-8"))
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")
            st.stop()

        with st.expander("👀 Input Preview"):
            st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:4000], language="json")

        # ---- System prompt (same mapping rules you shared) ----
        SYSTEM_PROMPT_TAB2 = """
You are given a JSON object with key "questions": a list where each item has:
- question (string)
- options: {"A":..., "B":..., "C":..., "D":...}
- correct_option (A/B/C/D)
- each question explanation should be placed with respective attachment#1

Produce a single flat JSON object with EXACTLY these keys. If something isn’t present, choose short sensible defaults (Hindi) rather than leaving it blank:

pagetitle, storytitle, typeofquiz, potraitcoverurl,
s1title1, s1text1,

s2questionHeading, s2question1,
s2option1, s2option1attr, s2option2, s2option2attr,
s2option3, s2option3attr, s2option4, s2option4attr,
s2attachment1,

s3questionHeading, s3question1,
s3option1, s3option1attr, s3option2, s3option2attr,
s3option3, s3option3attr, s3option4, s3option4attr,
s3attachment1,

s4questionHeading, s4question1,
s4option1, s4option1attr, s4option2, s4option2attr,
s4option3, s4option3attr, s4option4, s4option4attr,
s4attachment1,

s5questionHeading, s5question1,
s5option1, s5option1attr, s5option2, s5option2attr,
s5option3, s5option3attr, s5option4, s5option4attr,
s5attachment1,

s6questionHeading, s6question1,
s6option1, s6option1attr, s6option2, s6option2attr,
s6option3, s6option3attr, s6option4, s6option4attr,
s6attachment1,

results_bg_image, results_prompt_text, results1_text, results2_text, results3_text

Mapping rules:
- sNquestion1 ← questions[N-2].question  (N=2..6)
- sNoption1..4 ← options A..D text
- For the correct option, set sNoptionKattr to the **string** "correct"; for others set "".
- sNattachment1 ← explanation for that question
- sNquestionHeading ← "प्रश्न {N-1}"
- pagetitle/storytitle: derive short, relevant Hindi titles from the overall content.
- typeofquiz: set "शैक्षिक" if unknown.
- s1title1: a 2–5 word intro title; s1text1: 1–2 sentence intro.
- results_*: short friendly Hindi strings. results_bg_image: "" if none.

Return only the JSON object.
        """.strip()

        with st.spinner("🧩 Generating placeholders with GPT..."):
            try:
                resp = gpt_client.chat.completions.create(
                    model=GPT_DEPLOYMENT,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_TAB2},
                        {"role": "user",   "content": json.dumps(questions_data, ensure_ascii=False)}
                    ],
                )
                content = resp.choices[0].message.content
                content = clean_model_json(content)
                try:
                    placeholders = json.loads(content)
                except json.JSONDecodeError:
                    # salvage first JSON block if code fences/noise present
                    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                    if not m:
                        raise
                    placeholders = json.loads(m.group(0))

            except Exception as e:
                st.error(f"GPT failed: {e}")
                st.stop()

        st.success("✅ Placeholders ready!")
        st.subheader("Flat Placeholder JSON")
        st.json(placeholders)

        fname = f"quiz_placeholders_flat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            "⬇️ Download placeholders JSON",
            data=json.dumps(placeholders, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=fname,
            mime="application/json"
        )

        with st.expander("ℹ️ Notes"):
            st.markdown(
                """
- Input must include `questions` with options A–D and `correct_option`.
- The correct option’s `*attr` field becomes `"correct"`; others are empty strings.
- Headings auto-fill as **प्रश्न 1..5**; intro/results fields get short Hindi defaults if missing.
                """
            )
    else:
        st.info("Upload the `quiz_structured_*.json` produced in Tab 1.")


with tab3:
    st.info("Another placeholder tab (e.g., Settings, About).")
