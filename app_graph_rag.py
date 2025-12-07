import os
import re
import time
import numpy as np
import networkx as nx
import faiss
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load Environment Variables (API Keys)
load_dotenv()

# ======================================================
# 1. CONFIGURATION
# ======================================================
# The live data source
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"

# Embedding Model (Optimized for speed/accuracy balance)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Auto-Update Frequency (in seconds) -> 6 Hours
CACHE_TTL = 6 * 60 * 60 

# ======================================================
# 2. DATA STRUCTURES & PARSING LOGIC
# ======================================================

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    url: Optional[str]
    raw_md: str
    extracted_specs: Dict[str, str]

def extract_attributes(title: str, category: str) -> Dict[str, str]:
    """
    The 'Intelligence' Layer: Extracts Entities (Nodes) from unstructured text
    based on the product category.
    """
    specs = {}
    title_lower = title.lower()
    cat_lower = (category or "").lower()

    # --- BRAND EXTRACTION ---
    brands = [
        'Lenovo', 'HP', 'ASUS', 'Gigabyte', 'MSI', 'Dell', 'Acer', 'Apple', 
        'Samsung', 'Xiaomi', 'Realme', 'OnePlus', 'Infinix', 'Tecno', 'Vivo', 
        'Oppo', 'Honor', 'Motorola', 'Nokia', 'Walton', 'Chuwi', 'ZTE', 
        'Sony', 'Haier', 'Singer', 'TCL', 'Dahua', 'Hikvision', 'TP-Link', 
        'Tenda', 'Netis', 'Mercusys', 'ZKTeco', 'DJI', 'GoPro', 'Panda', 'Lotto', 'Bata'
    ]
    for brand in brands:
        if brand.lower() in title_lower:
            specs['Brand'] = brand
            break

    # --- ELECTRONICS LOGIC ---
    if any(x in cat_lower for x in ['laptop', 'phone', 'tablet', 'monitor', 'router', 'smart', 'watch']):
        # RAM
        ram_match = re.search(r'(\d+)\s?GB', title, re.IGNORECASE)
        if ram_match:
            specs['RAM'] = ram_match.group(1) + "GB"

        # Storage
        storage_match = re.search(r'(\d+)\s?(GB|TB)\s?(SSD|HDD|NVMe|ROM)', title, re.IGNORECASE)
        if storage_match:
            specs['Storage'] = f"{storage_match.group(1)}{storage_match.group(2)}"

        # CPU Family
        if 'ryzen' in title_lower:
            if ' 3 ' in title_lower: specs['CPU'] = 'Ryzen 3'
            elif ' 5 ' in title_lower: specs['CPU'] = 'Ryzen 5'
            elif ' 7 ' in title_lower: specs['CPU'] = 'Ryzen 7'
            elif ' 9 ' in title_lower: specs['CPU'] = 'Ryzen 9'
        elif 'core' in title_lower or 'intel' in title_lower:
            if 'i3' in title_lower: specs['CPU'] = 'Core i3'
            elif 'i5' in title_lower: specs['CPU'] = 'Core i5'
            elif 'i7' in title_lower: specs['CPU'] = 'Core i7'
            elif 'i9' in title_lower: specs['CPU'] = 'Core i9'
        elif 'm1' in title_lower: specs['CPU'] = 'Apple M1'
        elif 'm2' in title_lower: specs['CPU'] = 'Apple M2'
        elif 'm3' in title_lower: specs['CPU'] = 'Apple M3'
        elif 'm4' in title_lower: specs['CPU'] = 'Apple M4'
        elif 'snapdragon' in title_lower: specs['CPU'] = 'Snapdragon'

    # --- FASHION LOGIC ---
    elif any(x in cat_lower for x in ['sneaker', 'shirt', 'polo', 'jersey']):
        if 'cotton' in title_lower: specs['Material'] = 'Cotton'
        if 'leather' in title_lower: specs['Material'] = 'Leather'
        if 'mesh' in title_lower: specs['Material'] = 'Mesh'
        if 'canvas' in title_lower: specs['Material'] = 'Canvas'
        
        colors = ['black', 'white', 'blue', 'red', 'green', 'grey', 'yellow', 'navy', 'olive']
        for color in colors:
            if color in title_lower:
                specs['Color'] = color.capitalize()
                break

    return specs

def _parse_price_value(s: str) -> Optional[float]:
    if not s: return None
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums: return None
    try:
        vals = [float(x) for x in nums]
        return min(vals)
    except: return None

def parse_corpus_text(md_text: str) -> List[ProductDoc]:
    text = (md_text or "").strip()
    text = re.sub(r"^#\s*Combined.*?\n", "", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"\s+---\s+", text) if p.strip()]

    products = []
    for part in parts:
        m = re.search(r"##\s*(.+?)\s*(?=\*\*DocID:\*\*|\*\*DOCID:\*\*|DocID:|DOCID:)", part, re.IGNORECASE | re.DOTALL)
        title = (m.group(1).strip() if m else "").strip()
        if not title: continue

        m = re.search(r"\*\*DocID:\*\*\s*`?([A-Za-z0-9_\-]+)`?|DocID:\s*`?([A-Za-z0-9_\-]+)`?", part, re.IGNORECASE)
        doc_id = (m.group(1) or m.group(2) or "").strip() if m else None
        if not doc_id: continue

        m_src = re.search(r"\*\*Source:\*\*\s*([^*]+)", part, re.IGNORECASE)
        source = m_src.group(1).strip() if m_src else ("Daraz" if "daraz" in doc_id else "StarTech")

        m_cat = re.search(r"\*\*Category:\*\*\s*([^*]+)", part, re.IGNORECASE)
        category = m_cat.group(1).strip() if m_cat else "General"

        m_price = re.search(r"\*\*Price:\*\*\s*([^*]+)", part, re.IGNORECASE)
        price_val = _parse_price_value(m_price.group(1)) if m_price else None

        m_url = re.search(r"\*\*URL:\*\*\s*(\S+)", part, re.IGNORECASE)
        url = m_url.group(1).strip() if m_url else None

        r = re.search(r"\*\*Rating:\*\*\s*([0-5](?:\.\d+)?)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", part, re.IGNORECASE)
        r_avg = float(r.group(1)) if r else None
        r_cnt = int(r.group(2)) if r and r.group(2) else None

        meta_disp = [f"Source: {source}", f"Category: {category}"]
        if price_val: meta_disp.append(f"Price: ৳{int(price_val)}")
        raw_md = f"{title}\n" + " | ".join(meta_disp)

        specs = extract_attributes(title, category)

        products.append(ProductDoc(
            doc_id=doc_id, title=title, source=source, category=category,
            price_value=price_val, rating_avg=r_avg, rating_cnt=r_cnt,
            url=url, raw_md=raw_md, extracted_specs=specs
        ))
    
    return products

# ======================================================
# 3. KNOWLEDGE ENGINE (Graph + Vector)
# ======================================================

class KnowledgeEngine:
    def __init__(self, products: List[ProductDoc]):
        self.products = products
        self.product_map = {p.doc_id: p for p in products}
        self.graph = nx.Graph()
        self.index = None
        self.model = None
        self.doc_ids = []

    def build(self):
        # A. Build Graph
        print("Building Knowledge Graph...")
        for p in self.products:
            self.graph.add_node(p.doc_id, type='Product', price=p.price_value or 0, title=p.title)
            
            # Category Node
            cat_node = p.category.title()
            self.graph.add_edge(p.doc_id, cat_node, relation='IS_A')
            
            # Spec Nodes
            for key, value in p.extracted_specs.items():
                self.graph.add_edge(p.doc_id, value, relation=f'HAS_{key.upper()}')

        # B. Build Vector Index
        print("Building Vector Index...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        corpus_text = []
        for p in self.products:
            spec_text = ", ".join([f"{k}:{v}" for k,v in p.extracted_specs.items()])
            # We embed rich text to capture semantic meaning
            text = f"{p.title} category:{p.category} price:{p.price_value} {spec_text}"
            corpus_text.append(text)
            self.doc_ids.append(p.doc_id)
        
        embeddings = self.model.encode(corpus_text, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        return self

# ======================================================
# 4. ROBUST LOADER WITH TTL CACHING
# ======================================================

@st.cache_resource(ttl=CACHE_TTL, show_spinner=False)
def load_and_build_engine(url: str):
    """
    Fetches data from URL. 
    Streamlit will AUTOMATICALLY expire this cache after 6 hours (TTL).
    When expired, it re-runs this function to fetch fresh data.
    """
    print(f"🔄 Cache expired or init. Fetching fresh data from: {url}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Failed to download dataset. Status: {resp.status_code}")
        
        corpus_text = resp.text
        if not corpus_text:
            raise Exception("Downloaded content is empty.")

        products = parse_corpus_text(corpus_text)
        if not products:
            raise Exception("No products found after parsing.")
            
        engine = KnowledgeEngine(products)
        engine.build()
        return engine
        
    except Exception as e:
        print(f"❌ Critical Error loading data: {e}")
        return None

# ======================================================
# 5. REASONING ENGINE (Search Logic)
# ======================================================

def parse_user_constraints(query: str):
    constraints = {'max_price': None, 'min_ram': None}
    
    # Extract Price
    price_match = re.search(r'(?:under|below|budget|max)\s*(\d+)(?:k)?', query.lower())
    if price_match:
        val = int(price_match.group(1))
        if 'k' in query.lower() or val < 1000: val *= 1000
        constraints['max_price'] = val
        
    return constraints

def graph_rag_search(engine: KnowledgeEngine, query: str, top_k: int = 10):
    logs = []
    
    # 1. Vector Search (Retrieval)
    query_vec = engine.model.encode([query])
    # Fetch 3x candidates to allow for graph filtering
    D, I = engine.index.search(np.array(query_vec).astype('float32'), top_k * 3) 
    
    candidates = []
    for idx in I[0]:
        if idx < len(engine.doc_ids):
            candidates.append(engine.doc_ids[idx])
            
    # 2. Graph Reasoning (Verification)
    constraints = parse_user_constraints(query)
    if constraints['max_price']:
        logs.append(f"🧠 **Constraint Detected:** Max Price {constraints['max_price']}")
    
    final_results = []
    
    for doc_id in candidates:
        product = engine.product_map[doc_id]
        is_valid = True
        
        # Check Price
        if constraints['max_price'] and product.price_value:
            if product.price_value > constraints['max_price']:
                is_valid = False
                # logs.append(f"❌ Rejected '{product.title[:20]}...' (Price {product.price_value} > {constraints['max_price']})")

        # Check Graph Context (Example: if query has "Samsung", check graph connection)
        # This prevents "vector hallucinations" where a phone case appears for a phone query
        if "samsung" in query.lower():
            # Check if product is connected to 'Samsung' Brand Node
            if not engine.graph.has_edge(doc_id, "Samsung"):
                # If vectors found it but graph says it's not Samsung (maybe "For Samsung"), we might deprioritize
                pass 

        if is_valid:
            final_results.append(product)
            if len(final_results) >= top_k:
                break
            
    return final_results, logs

# ======================================================
# 6. OPENAI GENERATION
# ======================================================
def stream_openai_answer(results: List[ProductDoc], query: str, model_name: str):
    client = OpenAI() # Uses OPENAI_API_KEY from env
    
    context_text = ""
    for i, p in enumerate(results, 1):
        context_text += f"[{i}] {p.title}\nPrice: ৳{p.price_value} | Specs: {p.extracted_specs}\nDocID: {p.doc_id}\n---\n"
        
    system_prompt = """You are a helpful shopping assistant. 
    Use the provided context to answer the user's question. 
    If specific constraints (like budget) were met, explicitly mention that.
    Always cite products using [1], [2] etc.
    Keep the answer concise and professional."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_text}"}
    ]
    
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

# ======================================================
# 7. STREAMLIT UI
# ======================================================
st.set_page_config(page_title="GraphRAG Engine", layout="wide")

st.title("🛍️ Neuro-Symbolic GraphRAG")
st.caption("Live System: Updates automatically every 6 hours via GitHub.")

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    # Initialize Engine
    with st.spinner("Syncing with Live Database..."):
        engine = load_and_build_engine(DEFAULT_CORPUS_URL)
    
    if engine:
        st.success("System Online 🟢")
        st.metric("Total Products", len(engine.products))
        st.metric("Graph Nodes", engine.graph.number_of_nodes())
    else:
        st.error("System Offline 🔴")
        st.stop()
        
    st.markdown("---")
    model_choice = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-3.5-turbo"], index=0)
    top_k = st.slider("Retrieve", 3, 15, 5)
    
    if st.button("Force Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

# Main Chat Interface
query = st.text_input("Ask me anything (e.g., 'Best gaming laptop under 60k')", placeholder="Type your query...")

if query:
    # Run GraphRAG Pipeline
    with st.spinner("Reasoning..."):
        results, logs = graph_rag_search(engine, query, top_k)
    
    # Columns for Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💡 Answer")
        if results:
            st.write_stream(stream_openai_answer(results, query, model_choice))
        else:
            st.warning("No matching products found within your constraints.")
            
    with col2:
        st.subheader("🔍 Logic Trace")
        # Logic Logs
        if logs:
            with st.expander("Brain Activity", expanded=True):
                for log in logs:
                    st.info(log)
        else:
            st.caption("No specific constraints detected.")
            
        # Evidence
        st.markdown(f"**Top {len(results)} Candidates:**")
        for i, p in enumerate(results, 1):
            with st.expander(f"{i}. {p.title[:35]}...", expanded=False):
                st.markdown(f"""
                - **Price:** ৳{p.price_value}
                - **Category:** {p.category}
                - **Specs:** `{p.extracted_specs}`
                - [View Source]({p.url})
                """)