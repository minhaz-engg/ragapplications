import os
import re
import time
import requests
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# --- Standard Library Imports ---
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam

# 🛑 THE FIX: Import the pipeline initializer
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load Environment Variables
load_dotenv()

# ======================================================
# 1. CONFIGURATION
# ======================================================
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"
WORKING_DIR = "./lightrag_index"

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ======================================================
# 2. THE ROBUST DRIVER (With Embedding Dimension Fix)
# ======================================================

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def my_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content

async def my_embedding_func(texts: list[str]):
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [data.embedding for data in response.data]

# 🛑 CRITICAL FIX: Manually stamping the dimension
my_embedding_func.embedding_dim = 1536 

# ======================================================
# 3. ASYNC WRAPPERS (The Full Initialization Chain)
# ======================================================

def get_rag_instance():
    """Creates the instance (but does NOT initialize storage yet)."""
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=my_llm_model_func,
        embedding_func=my_embedding_func
    )

async def run_ingest_pipeline(texts: List[str]):
    """
    Lifecycle Manager for Ingestion:
    1. Create Instance
    2. Await Storage Init
    3. Await Pipeline Status Init (THE NEW FIX)
    4. Await Insertion
    """
    rag = get_rag_instance()
    
    # Sequence required by LightRAG v1.0+
    await rag.initialize_storages() 
    await initialize_pipeline_status()  # <--- NEW REQUIRED CALL
    
    await rag.ainsert(texts)
    return "Done"

async def run_query_pipeline(query: str, mode: str):
    """
    Lifecycle Manager for Querying
    """
    rag = get_rag_instance()
    
    # Sequence required by LightRAG v1.0+
    await rag.initialize_storages()
    await initialize_pipeline_status() # <--- NEW REQUIRED CALL
    
    param = QueryParam(mode=mode)
    result = await rag.aquery(query, param=param)
    return result

# ======================================================
# 4. DATA PARSING (Standard)
# ======================================================

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    url: Optional[str]
    extracted_specs: Dict[str, str]

def extract_attributes(title: str, category: str) -> Dict[str, str]:
    specs = {}
    title_lower = title.lower()
    cat_lower = (category or "").lower()
    
    if any(x in cat_lower for x in ['laptop', 'phone', 'tab']):
        ram = re.search(r'(\d+)\s?GB', title, re.IGNORECASE)
        if ram: specs['RAM'] = ram.group(1) + "GB"
    return specs

def parse_corpus_text(md_text: str) -> List[ProductDoc]:
    text = (md_text or "").strip()
    parts = [p.strip() for p in re.split(r"\s+---\s+", text) if p.strip()]
    products = []
    
    for part in parts:
        title_m = re.search(r"##\s*(.+?)\n", part)
        title = title_m.group(1).strip() if title_m else "Unknown"
        if title == "Unknown": continue

        price_m = re.search(r"\*\*Price:\*\*\s*([\d,]+)", part)
        price_val = float(price_m.group(1).replace(",", "")) if price_m else 0.0

        specs = extract_attributes(title, "General")
        products.append(ProductDoc(
            doc_id="N/A", title=title, source="Web", category="General",
            price_value=price_val, url=None, extracted_specs=specs
        ))
    return products

def format_products_for_ingestion(products: List[ProductDoc]) -> List[str]:
    texts = []
    for p in products:
        spec_str = ", ".join([f"{k} is {v}" for k, v in p.extracted_specs.items()])
        narrative = (
            f"Product: {p.title}. Price: {p.price_value} Taka. "
            f"Specs: {spec_str}."
        )
        texts.append(narrative)
    return texts

# ======================================================
# 5. STREAMLIT UI
# ======================================================

st.set_page_config(page_title="LightRAG Pro", layout="wide")
st.title("🧠 Neuro-Symbolic LightRAG (Pipeline Fixed)")

with st.sidebar:
    st.header("⚙️ Knowledge Base")
    
    if st.button("1. Fetch Live Data"):
        with st.spinner("Downloading..."):
            try:
                resp = requests.get(DEFAULT_CORPUS_URL)
                products = parse_corpus_text(resp.text)
                st.session_state['products'] = products
                st.success(f"Fetched {len(products)} products!")
            except Exception as e:
                st.error(str(e))

    if st.button("2. Build/Update Graph (Async)"):
        if 'products' in st.session_state:
            with st.spinner("Initializing Pipeline & Building Graph..."):
                texts = format_products_for_ingestion(st.session_state['products'])
                
                # RUN ASYNC IN SYNC CONTEXT
                asyncio.run(run_ingest_pipeline(texts))
                
                st.success("Graph Updated & Saved!")
        else:
            st.warning("Fetch data first.")

    st.markdown("---")
    
    if os.path.exists(WORKING_DIR) and len(os.listdir(WORKING_DIR)) > 0:
        st.success(f"✅ DB Active ({len(os.listdir(WORKING_DIR))} files)")
    else:
        st.warning("⚠️ DB Empty")

    mode = st.selectbox("Mode", ["hybrid", "local", "global"])

query = st.text_input("Ask:", placeholder="Compare prices...")

if query:
    start_time = time.time()
    with st.spinner("Reasoning..."):
        # RUN ASYNC IN SYNC CONTEXT
        response = asyncio.run(run_query_pipeline(query, mode))
        
    st.markdown("### 💡 Answer")
    st.markdown(response)
    st.caption(f"Time: {time.time()-start_time:.2f}s")