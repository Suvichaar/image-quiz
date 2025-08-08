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
    "Tab 2: (placeholder)",
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
# Placeholders (optional)
# ===========================
with tab2:
    st.info("Add whatever you want here later (e.g., Batch processing, History).")

with tab3:
    st.info("Another placeholder tab (e.g., Settings, About).")
