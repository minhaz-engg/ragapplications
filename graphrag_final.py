
import os
import re
import time
import networkx as nx
import requests
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# 1. Configuration & Constants
# ----------------------------
PAGE_TITLE = "🛍️ GraphRAG Application"
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
DEFAULT_MODEL = "gpt-4o-mini"
TOP_K_RETRIEVAL = 20

# Common "Junk" words that appear at start of titles but aren't brands
STOP_WORDS = {
    "new", "sale", "best", "exclusive", "offer", "discount", "hot", "top", 
    "original", "premium", "smart", "super", "mega", "combo", "buy", "get"
}

# Priority list for known tech brands (to fix StarTech missing tags)
KNOWN_BRANDS = {
    "apple", "samsung", "xiaomi", "realme", "oneplus", "oppo", "vivo", "google",
    "hp", "dell", "lenovo", "asus", "acer", "msi", "gigabyte", "intel", "amd",
    "nvidia", "sony", "canon", "nikon", "fujifilm", "dji", "gopro", "amazon",
    "logitech", "razer", "corsair", "tp-link", "d-link", "netgear", "tenda",
    "mikrotik", "cisco", "huion", "wacom", "hoco", "baseus", "anker", "remax",
    "joyroom", "haylou", "qcy", "soundpeats", "jbl", "bose", "sony", "edifier"
}

# ----------------------------
# 2. Data Structures
# ----------------------------

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: str
    category: str
    brand: str
    price_val: float
    url: str
    raw_text: str

    @property
    def clean_text(self) -> str:
        """Returns a normalized string for indexing."""
        return f"{self.title} {self.brand} {self.category} {self.source}"

# ----------------------------
# 3. Robust Utilities
# ----------------------------

class SmartTokenizer:
    """
    Handles alphanumeric segmentation better than .split().
    Preserves: 'i7-13700K', 'RTX-4090', '1000TB'
    """
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Convert to lower
        text = text.lower()
        # Regex to capture alphanumeric sequences, including hyphens inside words
        # e.g., "wi-fi" stays "wi-fi", "rtx 3060" becomes ["rtx", "3060"]
        tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
        return tokens

def infer_brand_robust(title: str, explicitly_tagged_brand: str = None) -> str:
    """
    High-IQ Logic:
    1. If explicit brand exists (from Markdown), use it.
    2. Else, check if any 'Known Brand' exists in the Title.
    3. Else, take the first word of Title, but reject if it's a 'Stop Word'.
    """
    if explicitly_tagged_brand and explicitly_tagged_brand.lower() != "generic":
        return explicitly_tagged_brand.lower()
    
    if not title:
        return "generic"
    
    title_lower = title.lower()
    tokens = SmartTokenizer.tokenize(title_lower)
    
    # Check priority list
    for token in tokens:
        if token in KNOWN_BRANDS:
            return token
            
    # Fallback: First valid word
    if tokens:
        candidate = tokens[0]
        if candidate not in STOP_WORDS and len(candidate) > 2: # Ignore 1-2 letter junk
            return candidate
            
    return "generic"

def parse_price(price_str: str) -> float:
    if not price_str: return 0.0
    nums = re.findall(r'\d+', price_str.replace(',', ''))
    return float("".join(nums)) if nums else 0.0

# ----------------------------
# 4. Production Data Ingestion
# ----------------------------

@st.cache_data(ttl=3600) # Cache raw data fetch for 1 hour
def fetch_raw_data(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10) # Added timeout
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return ""

def parse_corpus(text: str) -> List[ProductDoc]:
    """
    Robust regex parsing that doesn't crash on empty lines.
    """
    if not text: return []
    
    # Split by the standardized separator
    product_blocks = re.split(r'\n---\n', text)
    docs = []
    
    # Pre-compiled regex for speed
    re_docid = re.compile(r"\*\*DocID:\*\*\s*`([^`]+)`")
    re_source = re.compile(r"\*\*Source:\*\*\s*(.+)")
    re_cat = re.compile(r"\*\*Category:\*\*\s*(.+)")
    re_brand = re.compile(r"\*\*Brand:\*\*\s*(.+)")
    re_url = re.compile(r"\*\*URL:\*\*\s*(\S+)")
    re_price = re.compile(r"\*\*Price:\*\*\s*(.+)")
    re_title = re.compile(r"^##\s*(.+)", re.MULTILINE)

    for block in product_blocks:
        block = block.strip()
        if not block or "**DocID:**" not in block:
            continue
            
        doc_id_m = re_docid.search(block)
        doc_id = doc_id_m.group(1).strip() if doc_id_m else f"unknown-{hash(block)}"
        
        title_m = re_title.search(block)
        title = title_m.group(1).strip() if title_m else "Unknown Product"
        
        # Extract Brand (Raw)
        brand_m = re_brand.search(block)
        raw_brand = brand_m.group(1).strip() if brand_m else None
        
        # Intelligent Inference
        final_brand = infer_brand_robust(title, raw_brand)
        
        # Other metadata
        source = re_source.search(block).group(1).strip() if re_source.search(block) else "Unknown"
        category = re_cat.search(block).group(1).strip().lower() if re_cat.search(block) else "general"
        url = re_url.search(block).group(1).strip() if re_url.search(block) else ""
        price_val = parse_price(re_price.search(block).group(1)) if re_price.search(block) else 0.0

        docs.append(ProductDoc(
            doc_id=doc_id, title=title, source=source, category=category,
            brand=final_brand, price_val=price_val, url=url, raw_text=block
        ))
        
    return docs

# ----------------------------
# 5. The GraphRAG Engine
# ----------------------------

class GraphRAGIndex:
    def __init__(self, docs: List[ProductDoc]):
        self.doc_map = {d.doc_id: d for d in docs}
        self.graph = nx.Graph()
        
        # Data Partitioning
        self.docs_daraz = [d for d in docs if 'daraz' in d.source.lower()]
        self.docs_startech = [d for d in docs if 'startech' in d.source.lower()]
        
        # Build Indices (Lazy loading not needed here as we want speed at query time)
        self.bm25_daraz = self._build_bm25(self.docs_daraz)
        self.bm25_startech = self._build_bm25(self.docs_startech)
        
        # Build Graph
        self._build_knowledge_graph(docs)

    def _build_bm25(self, doc_list: List[ProductDoc]):
        if not doc_list: return None
        # Use Smart Tokenizer
        tokenized_corpus = [SmartTokenizer.tokenize(d.clean_text) for d in doc_list]
        return BM25Okapi(tokenized_corpus)

    def _build_knowledge_graph(self, docs: List[ProductDoc]):
        for doc in docs:
            self.graph.add_node(doc.doc_id, type='product', source=doc.source)
            
            # 1. Brand Link (Strict Hygiene)
            # Only link if brand is valid and NOT generic
            if doc.brand and doc.brand not in ["generic", "unknown", "other"]:
                brand_node = f"BRAND:{doc.brand}"
                self.graph.add_node(brand_node, type='brand')
                self.graph.add_edge(doc.doc_id, brand_node)
            
            # 2. Category Link (Less Strict, but useful for broad recall)
            if doc.category and doc.category not in ["general", "unknown"]:
                cat_node = f"CAT:{doc.category}"
                self.graph.add_node(cat_node, type='category')
                self.graph.add_edge(doc.doc_id, cat_node)

    def search(self, query: str, total_k: int = 20) -> List[ProductDoc]:
        half_k = total_k // 2
        
        # 1. Tokenize Query
        tokenized_query = SmartTokenizer.tokenize(query)
        
        # 2. Independent Search
        daraz_hits = self._query_bm25(self.bm25_daraz, self.docs_daraz, tokenized_query, half_k)
        startech_hits = self._query_bm25(self.bm25_startech, self.docs_startech, tokenized_query, half_k)
        
        # 3. Interleave (Round Robin)
        combined = []
        max_len = max(len(daraz_hits), len(startech_hits))
        for i in range(max_len):
            if i < len(daraz_hits): combined.append(daraz_hits[i])
            if i < len(startech_hits): combined.append(startech_hits[i])
            
        # 4. Graph Expansion (The "Smart" part)
        # Use top 2 highly relevant results to find siblings
        expanded_results = []
        seen_ids = {d.doc_id for d in combined}
        
        # Add original hits first
        expanded_results.extend(combined)
        
        if combined:
            seeds = combined[:2] # Only use the very best matches as seeds
            graph_hits = []
            
            for seed in seeds:
                try:
                    neighbors = list(self.graph.neighbors(seed.doc_id))
                    for node in neighbors:
                        # Find siblings (products connected to the same brand/cat)
                        siblings = list(self.graph.neighbors(node))
                        for sib_id in siblings:
                            if sib_id != seed.doc_id and sib_id not in seen_ids and sib_id in self.doc_map:
                                graph_hits.append(self.doc_map[sib_id])
                                seen_ids.add(sib_id)
                except:
                    pass
            
            # Prioritize Graph hits: take top 5
            expanded_results.extend(graph_hits[:5])

        return expanded_results

    def _query_bm25(self, bm25_idx, doc_source, tokenized_query, k):
        if not bm25_idx or not doc_source: return []
        scores = bm25_idx.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [doc_source[i] for i in top_n if scores[i] > 0.0] # Strict > 0 check

# ----------------------------
# 6. Global Resource Management
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_search_engine():
    """
    Critical for Production: This runs ONCE when the app starts.
    Subsequent reloads by user will reuse this object instantly.
    """
    raw_text = fetch_raw_data(DEFAULT_CORPUS_URL)
    if not raw_text:
        return None
    docs = parse_corpus(raw_text)
    return GraphRAGIndex(docs)

# ----------------------------
# 7. UI / Main App
# ----------------------------

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="🛍️")
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(PAGE_TITLE)
        st.caption("Engineered for Fairness: Dual-Index Retrieval + Graph Expansion")
    with col2:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key: os.environ["OPENAI_API_KEY"] = api_key
        
        st.divider()
        st.info("System Status")
        
        # Load Engine
        with st.spinner("Initializing Knowledge Graph..."):
            index = load_search_engine()
            
        if index:
            st.success(f"✅ System Online")
            st.metric("Total Products", len(index.doc_map))
            st.write(f"🔵 Daraz: {len(index.docs_daraz)}")
            st.write(f"🔴 StarTech: {len(index.docs_startech)}")
        else:
            st.error("❌ System Offline (Data Fetch Failed)")
            st.stop()

        if st.button("🧹 Clear Cache & Rebuild"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you compare prices today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: 'Best gaming laptop under 100k' or 'Samsung S24 Ultra'"):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieval Step
        start_time = time.time()
        results = index.search(prompt, total_k=TOP_K_RETRIEVAL)
        latency = time.time() - start_time

        if not results:
            with st.chat_message("assistant"):
                st.warning("No matching products found in the database.")
            return

        # Prepare Context
        context_str = ""
        context_display = [] # For expandable view
        
        for i, doc in enumerate(results, 1):
            context_str += f"[{i}] {doc.title} | Brand: {doc.brand} | Price: {doc.price_val} | Src: {doc.source}\nLink: {doc.url}\n\n"
            context_display.append(doc)

        # Assistant Response
        with st.chat_message("assistant"):
            # 1. Show Sources (Transparency)
            with st.expander(f"🔍 Retrieved {len(results)} items in {latency:.3f}s", expanded=False):
                for doc in context_display:
                    color = "blue" if "daraz" in doc.source.lower() else "red"
                    st.markdown(f":{color}[**{doc.source}**] [{doc.title}]({doc.url}) - **{doc.price_val:,.0f}৳**")

            # 2. LLM Generation
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
            
            if not client:
                st.info("💡 Enter OpenAI Key in sidebar for AI comparison. Showing raw results above.")
            else:
                stream_box = st.empty()
                full_resp = ""
                
                system_prompt = (
                    "You are a sophisticated shopping assistant (OmniShop). \n"
                    "1. Analyze the Context provided.\n"
                    "2. Compare prices between Daraz and StarTech explicitly.\n"
                    "3. Recommend the best value deal based on specs and price.\n"
                    "4. Output clean Markdown."
                )
                
                try:
                    stream = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {prompt}"}
                        ],
                        stream=True
                    )
                    
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_resp += content
                        stream_box.markdown(full_resp + "▌")
                    
                    stream_box.markdown(full_resp)
                    st.session_state.messages.append({"role": "assistant", "content": full_resp})
                    
                except Exception as e:
                    st.error(f"LLM Connection Error: {e}")

if __name__ == "__main__":
    main()