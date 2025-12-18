# -*- coding: utf-8 -*-
"""
OmniShop AI: Enterprise GraphRAG (Async & Weighted Fusion)
==========================================================
Architect: Minhaz Chowdhury
Date: December 2025

Key Architectural Decisions:
1. Asynchronous I/O: Non-blocking data ingestion using aiohttp.
2. Weighted Fusion: Prioritizes relevance over source equality.
3. Graph Density Control: Limits expansion to top-tier semantic neighbors.
4. Robust Typing: Full PEP 484 compliance for maintainability.
"""

import os
import re
import time
import asyncio
import aiohttp
import networkx as nx
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

# Initialize Environment
load_dotenv()

# ----------------------------
# 1. System Configuration
# ----------------------------
@dataclass
class Config:
    PAGE_TITLE: str = "🛍️ OmniShop AI: Enterprise Edition"
    CORPUS_URL: str = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
    MODEL_NAME: str = "gpt-4o-mini"
    TOP_K_RETRIEVAL: int = 15  # Slightly reduced for higher precision
    GRAPH_EXPANSION_LIMIT: int = 5
    CACHE_TTL: int = 3600

    # Noise filtration for clean graph building
    STOP_WORDS: Set[str] = field(default_factory=lambda: {
        "new", "sale", "best", "exclusive", "offer", "discount", "hot", "top", 
        "original", "premium", "smart", "super", "mega", "combo", "buy", "get",
        "limited", "deal", "flash", "arrival"
    })

    # Canonical Brand List (The Knowledge Base)
    KNOWN_BRANDS: Set[str] = field(default_factory=lambda: {
        "apple", "samsung", "xiaomi", "realme", "oneplus", "oppo", "vivo", "google",
        "hp", "dell", "lenovo", "asus", "acer", "msi", "gigabyte", "intel", "amd",
        "nvidia", "sony", "canon", "nikon", "fujifilm", "dji", "gopro", "amazon",
        "logitech", "razer", "corsair", "tp-link", "d-link", "netgear", "tenda",
        "mikrotik", "cisco", "huion", "wacom", "hoco", "baseus", "anker", "remax",
        "joyroom", "haylou", "qcy", "soundpeats", "jbl", "bose", "edifier"
    })

CFG = Config()

# ----------------------------
# 2. Advanced Data Structures
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
    relevance_score: float = 0.0  # Dynamic scoring field

    @property
    def clean_token_string(self) -> str:
        """Optimized string for BM25 indexing (heavily weighted on brand/model)."""
        return f"{self.title} {self.brand} {self.title} {self.category}"

# ----------------------------
# 3. Intelligent Utilities
# ----------------------------

class SmartTokenizer:
    """
    Tokenizer V2: Hyphen-Agnostic.
    Splits 'RTX-4060' into ['rtx', '4060'] to ensure it matches 
    regardless of how the user or database formats it.
    """
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # 1. Replace hyphens with spaces (Standardization)
        clean_text = text.lower().replace("-", " ")
        
        # 2. Remove special chars but keep alphanumerics
        # This splits "rtx 4060" -> ["rtx", "4060"]
        tokens = re.findall(r'[a-z0-9]+', clean_text)
        
        return tokens

def infer_brand_advanced(title: str, raw_brand: str = None) -> str:
    """
    Deterministic Brand Inference Engine.
    Prioritizes explicit tags, then checks Knowledge Base, then falls back to heuristics.
    """
    # 1. Trust explicit tag if it's not generic
    if raw_brand and len(raw_brand) > 2 and raw_brand.lower() not in ["generic", "other", "null"]:
        return raw_brand.lower()
    
    if not title: return "generic"
    
    title_lower = title.lower()
    tokens = SmartTokenizer.tokenize(title_lower)
    
    # 2. Knowledge Base Scan (O(1) lookup per token)
    for token in tokens:
        if token in CFG.KNOWN_BRANDS:
            return token
            
    # 3. Heuristic Fallback (First valid non-stopword)
    if tokens:
        candidate = tokens[0]
        if candidate not in CFG.STOP_WORDS and len(candidate) > 2 and not candidate.isdigit():
            return candidate
            
    return "generic"

# ----------------------------
# 4. Asynchronous Ingestion Engine
# ----------------------------

async def fetch_data_async(url: str) -> str:
    """Non-blocking fetch using aiohttp."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            # Log this in a real app
            return ""

def parse_corpus(text: str) -> List[ProductDoc]:
    """Robust Regex Parsing."""
    if not text: return []
    
    product_blocks = re.split(r'\n---\n', text)
    docs = []
    
    # Pre-compiled patterns for performance
    patterns = {
        'doc_id': re.compile(r"\*\*DocID:\*\*\s*`([^`]+)`"),
        'source': re.compile(r"\*\*Source:\*\*\s*(.+)"),
        'category': re.compile(r"\*\*Category:\*\*\s*(.+)"),
        'brand': re.compile(r"\*\*Brand:\*\*\s*(.+)"),
        'url': re.compile(r"\*\*URL:\*\*\s*(\S+)"),
        'price': re.compile(r"\*\*Price:\*\*\s*(.+)"),
        'title': re.compile(r"^##\s*(.+)", re.MULTILINE)
    }

    for block in product_blocks:
        if "**DocID:**" not in block: continue
        
        # Extraction with safe defaults
        get_val = lambda k: patterns[k].search(block).group(1).strip() if patterns[k].search(block) else ""
        
        title = get_val('title') or "Unknown Product"
        raw_brand = get_val('brand')
        final_brand = infer_brand_advanced(title, raw_brand)
        
        # Price Parsing logic
        price_str = get_val('price')
        price_val = 0.0
        if price_str:
            clean_price = re.sub(r'[^\d.]', '', price_str.replace(',', ''))
            try: price_val = float(clean_price)
            except: pass

        docs.append(ProductDoc(
            doc_id=get_val('doc_id') or f"unk-{hash(block)}",
            title=title,
            source=get_val('source') or "Unknown",
            category=get_val('category').lower() or "general",
            brand=final_brand,
            price_val=price_val,
            url=get_val('url'),
            raw_text=block
        ))
    return docs

# ----------------------------
# 5. The Hybrid Retrieval Engine (Graph + BM25)
# ----------------------------

class HybridSearchEngine:
    def __init__(self, docs: List[ProductDoc]):
        self.docs = docs
        self.doc_map = {d.doc_id: d for d in docs}
        self.graph = nx.Graph()
        
        # Build Index
        self.bm25 = self._build_bm25()
        self._build_knowledge_graph()

    def _build_bm25(self):
        tokenized_corpus = [SmartTokenizer.tokenize(d.clean_token_string) for d in self.docs]
        return BM25Okapi(tokenized_corpus)

    def _build_knowledge_graph(self):
        """Constructs a semantic graph linking Products, Brands, and Categories."""
        for doc in self.docs:
            self.graph.add_node(doc.doc_id, type='product')
            
            # Strong Link: Brand
            if doc.brand and doc.brand != "generic":
                brand_node = f"BRAND:{doc.brand}"
                self.graph.add_edge(doc.doc_id, brand_node, weight=1.0)
            
            # Weak Link: Category (prevents massive clusters)
            if doc.category and doc.category != "general":
                cat_node = f"CAT:{doc.category}"
                self.graph.add_edge(doc.doc_id, cat_node, weight=0.5)

    def search(self, query: str) -> List[ProductDoc]:
        """
        Executes a Weighted Fusion Search.
        1. BM25 Retrieval (Lexical)
        2. Graph Expansion (Semantic)
        3. Score Normalization & Re-ranking
        """
        tokenized_query = SmartTokenizer.tokenize(query)
        if not tokenized_query: return []

        # Step 1: BM25 Retrieval
        # We fetch 2x the needed amount to allow for re-ranking
        raw_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:CFG.TOP_K_RETRIEVAL * 2]
        
        candidates: Dict[str, float] = {}
        
        # Normalize BM25 scores (0-1 range roughly)
        max_score = raw_scores[top_n_indices[0]] if top_n_indices else 1.0
        
        for idx in top_n_indices:
            score = raw_scores[idx]
            if score <= 0: continue
            doc = self.docs[idx]
            # Base Score = Normalized BM25
            candidates[doc.doc_id] = (score / max_score) * 1.0

        # Step 2: Graph Expansion (The "Smart" Boost)
        # Look at the top 3 strongest matches and find their conceptual siblings
        top_seeds = sorted(candidates.keys(), key=lambda k: candidates[k], reverse=True)[:3]
        
        for seed_id in top_seeds:
            try:
                # Get neighbors (Brands/Categories)
                neighbors = list(self.graph.neighbors(seed_id))
                for node in neighbors:
                    # Get siblings (Other products)
                    siblings = list(self.graph.neighbors(node))
                    for sib_id in siblings:
                        if sib_id == seed_id: continue
                        
                        # Apply Graph Boost
                        # If connected via BRAND, high boost. If CATEGORY, low boost.
                        boost = 0.3 if "BRAND:" in node else 0.1
                        
                        if sib_id in candidates:
                            candidates[sib_id] += boost
                        elif sib_id in self.doc_map:
                            # New discovery via graph
                            candidates[sib_id] = boost
            except:
                pass

        # Step 3: Final Selection
        # Sort by final fused score
        final_ids = sorted(candidates.keys(), key=lambda k: candidates[k], reverse=True)[:CFG.TOP_K_RETRIEVAL]
        
        results = []
        for doc_id in final_ids:
            doc = self.doc_map[doc_id]
            doc.relevance_score = candidates[doc_id]
            results.append(doc)
            
        return results

# ----------------------------
# 6. Lifecycle Management
# ----------------------------

@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[HybridSearchEngine]:
    """
    Singleton Pattern for the Search Engine.
    Uses asyncio logic wrapped in sync for Streamlit compatibility.
    """
    try:
        # Create a new event loop for the async fetch
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        raw_text = loop.run_until_complete(fetch_data_async(CFG.CORPUS_URL))
        loop.close()
        
        if not raw_text: return None
        
        docs = parse_corpus(raw_text)
        return HybridSearchEngine(docs)
    except Exception as e:
        print(f"Engine Init Error: {e}")
        return None

# ----------------------------
# 7. UI Presentation Layer
# ----------------------------

def main():
    st.set_page_config(page_title=CFG.PAGE_TITLE, layout="wide", page_icon="🛍️")
    
    # CSS Injection for polished look
    st.markdown("""
    <style>
        .stChatMessage { border-radius: 10px; border: 1px solid #e0e0e0; }
        .metric-card { background-color: black; color: white; padding: 10px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Center")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key: os.environ["OPENAI_API_KEY"] = api_key
        
        st.divider()
        
        with st.spinner("Booting Neural Engine..."):
            engine = get_engine()
            
        if engine:
            st.success("✅ Engine Online")
            st.markdown(f"""
            <div class='metric-card'>
                <b>Indexed Assets:</b> {len(engine.docs)}<br>
                <b>Brands Known:</b> {len(CFG.KNOWN_BRANDS)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Engine Failure")
            st.stop()

        if st.button("🔄 Hot Reload"):
            st.cache_resource.clear()
            st.rerun()

    # Main Area
    st.title(CFG.PAGE_TITLE)
    st.caption("🚀 Powered by Hybrid Retrieval (BM25 + Semantic Graph Fusion)")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat Feed
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    # Input Loop
    if prompt := st.chat_input("Ask about laptops, cameras, or compare prices..."):
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processing
        start_ts = time.time()
        results = engine.search(prompt)
        latency = time.time() - start_ts

        with st.chat_message("assistant"):
            if not results:
                response_text = "I couldn't find any products matching that specific query. Try broadening your terms?"
                st.warning(response_text)
                st.session_state.history.append(("assistant", response_text))
            else:
                # Context Construction
                context_str = "\n".join([
                    f"- {d.title} (Brand: {d.brand}) | Price: {d.price_val} | Source: {d.source}"
                    for d in results
                ])
                
                # UI Display of Sources (Expandable)
                with st.expander(f"⚡ Retrieved {len(results)} items in {latency:.3f}s (Top Matches)", expanded=False):
                    for doc in results:
                        color = "blue" if "daraz" in doc.source.lower() else "red"
                        score_display = f"{doc.relevance_score:.2f}"
                        st.markdown(f":{color}[**{doc.source}**] [{doc.title}]({doc.url}) - **{doc.price_val:,.0f}৳** (Score: {score_display})")

                # Generative Step
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
                
                if client:
                    stream_box = st.empty()
                    full_resp = ""
                    
                    system_prompt = (
                        "You are OmniShop AI, an expert procurement assistant. "
                        "Synthesize the provided product data into a clear comparison. "
                        "Highlight the best value. Be objective."
                    )
                    
                    try:
                        stream = client.chat.completions.create(
                            model=CFG.MODEL_NAME,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Context:\n{context_str}\n\nUser Question: {prompt}"}
                            ],
                            stream=True
                        )
                        
                        for chunk in stream:
                            content = chunk.choices[0].delta.content or ""
                            full_resp += content
                            stream_box.markdown(full_resp + "▌")
                        
                        stream_box.markdown(full_resp)
                        st.session_state.history.append(("assistant", full_resp))
                        
                    except Exception as e:
                        st.error(f"LLM Error: {str(e)}")
                else:
                    st.info("ℹ️ Results retrieved (see above). Add API Key for AI synthesis.")

if __name__ == "__main__":
    main()