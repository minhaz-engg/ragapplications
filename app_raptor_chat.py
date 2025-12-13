# -*- coding: utf-8 -*-
"""
Daraz + StarTech Conversational RAG
Features: RAPTOR (L1) + BM25 + Chat Memory + Smart Ranking + Latency Caching
"""

import os
import re
import io
import json
import pickle
import hashlib
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# App Config
# ----------------------------
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Default Corpus URL (Raw GitHub Link)
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 20
RAPTOR_MODEL = "gpt-4o-mini"

# ----------------------------
# Data structures
# ----------------------------

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

@dataclass
class ChunkRec:
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    url: Optional[str]
    text: str
    level: int # 0 = Product, 1 = RAPTOR Summary

# ----------------------------
# Utilities
# ----------------------------

STOPWORDS = set([
    "the","a","an","and","or","of","for","on","in","to","from","with","by","at","is","are","was","were",
    "this","that","these","those","it","its","as","be","can","will","has","have","i","me","my","show","tell"
])

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _parse_price_value(s: str) -> Optional[float]:
    if not s: return None
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums: return None
    try:
        vals = [float(x) for x in nums]
        return min(vals) if vals else None
    except: return None

def _clean_for_bm25(text: str) -> str:
    clean_lines = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll or ll.lower().startswith("**images"): continue
        if "http://" in ll or "https://://" in ll:
            parts = re.split(r"\s+https?://\S+", ll)
            ll = " ".join([p for p in parts if p.strip()])
        if ll: clean_lines.append(ll)
    return "\n".join(clean_lines)

def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]

@st.cache_resource
def _ensure_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: st.error("Missing OPENAI_API_KEY"); st.stop()
    return OpenAI()

def _generate_search_url(source: str, title: str) -> str:
    """Fallback generator for missing links"""
    q = urllib.parse.quote(title)
    if "startech" in source.lower():
        return f"https://www.startech.com.bd/product/search?search={q}"
    elif "daraz" in source.lower():
        return f"https://www.daraz.com.bd/catalog/?q={q}"
    return f"https://www.google.com/search?q={q}"

# ----------------------------
# 1. Corpus Parsing (Cached)
# ----------------------------

@st.cache_data(show_spinner=False)
def fetch_and_parse_corpus(url: str) -> List[ProductDoc]:
    """Downloads and parses the corpus. Cached to minimize latency."""
    import requests
    try:
        r = requests.get(url, timeout=30)
        if not r.ok: return []
        text = r.text
    except: return []

    # Parsing Logic
    text = re.sub(r"^#\s*Combined.*?\n", "", text, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in re.split(r"\s+---\s+", text) if p.strip()]
    
    products = []
    for part in parts:
        m_title = re.search(r"##\s*(.+?)\s*(?=\*\*DocID)", part, re.IGNORECASE|re.DOTALL)
        title = m_title.group(1).strip() if m_title else ""
        if not title: continue

        m_id = re.search(r"DocID:\*\*\s*`?([A-Za-z0-9_\-]+)`?", part, re.IGNORECASE)
        doc_id = m_id.group(1).strip() if m_id else ""
        
        # Source
        m_src = re.search(r"Source:\*\*\s*([^*]+)", part, re.IGNORECASE)
        source = m_src.group(1).strip() if m_src else ("Daraz" if "daraz" in doc_id.lower() else "StarTech")

        # Category
        m_cat = re.search(r"Category:\*\*\s*([^*]+)", part, re.IGNORECASE)
        category = m_cat.group(1).strip() if m_cat else None

        # Price
        m_prc = re.search(r"Price:\*\*\s*([^*]+)", part, re.IGNORECASE)
        price_val = _parse_price_value(m_prc.group(1)) if m_prc else None

        # URL
        m_url = re.search(r"URL:\*\*\s*(\S+)", part, re.IGNORECASE)
        url = m_url.group(1).strip() if m_url else None
        
        # Fallback URL logic
        if not url:
            url = _generate_search_url(source, title)

        # Reconstruct clean MD
        meta = [f"Source: {source}"]
        if category: meta.append(f"Category: {category}")
        if price_val: meta.append(f"Price: ৳{int(price_val)}")
        raw_md = f"{title}\n" + " | ".join(meta)

        products.append(ProductDoc(doc_id, title, source, category, price_val, None, None, url, raw_md))
    
    return products

# ----------------------------
# 2. RAPTOR L1 Summaries (Cached)
# ----------------------------

@st.cache_data(show_spinner=False)
def get_raptor_summaries(_products: List[ProductDoc], model: str) -> List[ChunkRec]:
    """Generates L1 summaries. Cached so we don't re-call OpenAI on restart."""
    # This uses a dummy hash of products to invalidate cache if corpus changes
    client = _ensure_client()
    groups = defaultdict(list)
    for p in _products:
        groups[p.category or "Other"].append(p)
    
    l1_chunks = []
    # Only process if we haven't loaded from file (Double layer caching: Memory + File)
    # Since st.cache_data handles memory, we just do logic here.
    
    # We will simulate a file check inside this cached function for persistency across sessions
    products_hash = _sha1("".join([p.doc_id for p in _products]))
    cache_path = os.path.join(INDEX_DIR, f"raptor_l1_{products_hash}.pkl")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f: return pickle.load(f)
        except: pass

    # If no file cache, generate (this is slow, but happens once)
    for cat, prods in groups.items():
        if not prods: continue
        # Limit context to avoid huge costs, take top 40 items representative
        context = "\n".join([p.raw_md for p in prods[:40]])
        prompt = (f"Summarize the product category '{cat}'. Listing types, brands, and price range.\n\n{context}")
        
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role":"user", "content": prompt}], temperature=0
            )
            text = resp.choices[0].message.content
            l1_chunks.append(ChunkRec(
                doc_id=f"summary_{cat}", title=f"Overview: {cat}", source="RAPTOR",
                category=cat, price_value=None, rating_avg=None, rating_cnt=None,
                url=None, text=text, level=1
            ))
        except: pass
    
    # Save to file
    with open(cache_path, "wb") as f: pickle.dump(l1_chunks, f)
    
    return l1_chunks

# ----------------------------
# 3. Indexing & Ranking Logic
# ----------------------------

def build_index(products: List[ProductDoc], enable_raptor: bool):
    chunker = RecursiveChunker.from_recipe("markdown", lang="en")
    
    l0_chunks = []
    for p in products:
        # L0: Product level
        l0_chunks.append(ChunkRec(
            p.doc_id, p.title, p.source, p.category, p.price_value, 
            p.rating_avg, p.rating_cnt, p.url, _clean_for_bm25(p.raw_md), 0
        ))
    
    l1_chunks = []
    if enable_raptor:
        l1_chunks = get_raptor_summaries(products, RAPTOR_MODEL)
    
    all_chunks = l0_chunks + l1_chunks
    tokenized = [_tokenize(c.text + " " + c.title) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, all_chunks

def advanced_search(
    bm25: BM25Okapi, 
    chunks: List[ChunkRec], 
    query: str, 
    top_k: int,
    filters: Dict
) -> List[Tuple[ChunkRec, float]]:
    
    q_toks = _tokenize(query)
    scores = bm25.get_scores(q_toks)
    
    # Heuristics for ranking
    # 1. If query contains numbers (e.g. "15", "4060"), it's likely a specific product search -> Boost L0
    # 2. If query is generic (e.g. "gaming laptop"), L1 summaries are good.
    is_specific = any(t.isdigit() for t in q_toks)
    
    final_results = []
    q_set = set(q_toks)

    for i, score in enumerate(scores):
        c = chunks[i]
        
        # -- Filtering --
        if filters.get("min_price") and (c.price_value is None or c.price_value < filters["min_price"]): continue
        if filters.get("max_price") and (c.price_value is None or c.price_value > filters["max_price"]): continue
        if filters.get("source") and c.source and c.source != "RAPTOR" and c.source != filters["source"]: continue
        
        # -- Custom Ranking Score --
        final_score = score
        
        # Title Jaccard Boost (Robust similarity measure)
        # Matches words in query vs words in title
        t_toks = set(_tokenize(c.title))
        intersection = len(q_set.intersection(t_toks))
        if intersection > 0:
            # Huge boost if title matches query terms
            final_score += (intersection * 1.5)
        
        # L0 vs L1 Balancing
        if c.level == 1: # Summary
            if is_specific: 
                final_score *= 0.5 # Penalty for summaries in specific search
            else:
                final_score *= 1.2 # Boost summaries in generic search
        else: # Product
            if is_specific:
                final_score *= 1.1
        
        final_results.append((c, final_score))
    
    # Sort and Diversify
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    # Diversification: Don't show same doc_id twice (unless it's product vs summary)
    seen = set()
    unique_results = []
    for c, s in final_results:
        if c.doc_id in seen: continue
        seen.add(c.doc_id)
        unique_results.append((c, s))
        if len(unique_results) >= top_k: break
        
    return unique_results

# ----------------------------
# 4. UI & Chat Logic
# ----------------------------

st.set_page_config(page_title="Product Copilot", layout="wide", page_icon="🛍️")

# CSS for Chat Style
st.markdown("""
<style>
    .user-msg {background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;}
    .bot-msg {background-color: #e8f0fe; padding: 10px; border-radius: 10px; margin: 5px 0;}
    .product-card {border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 5px; font-size: 0.9em;}
    .price-tag {color: #2e7d32; font-weight: bold;}
    a {text-decoration: none; color: #1976d2;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🛍️ Product Copilot")
    st.markdown("RAG System for Daraz & StarTech")
    
    # Corpus Loader
    if "products" not in st.session_state:
        with st.spinner("Initializing Knowledge Base..."):
            st.session_state.products = fetch_and_parse_corpus(DEFAULT_CORPUS_URL)
            st.session_state.bm25, st.session_state.chunks = build_index(st.session_state.products, True)
            st.success(f"Loaded {len(st.session_state.products)} products!")

    # Filters
    st.markdown("### 🔎 Filters")
    f_source = st.selectbox("Source", ["All", "Daraz", "StarTech"])
    f_price_max = st.number_input("Max Price (BDT)", min_value=0, step=1000)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you find products on Daraz and StarTech. Ask me anything like **'Best gaming laptop under 80k'** or **'Price of iPhone 15'**."}]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("What are you looking for?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Retrieval
    filters = {}
    if f_source != "All": filters["source"] = f_source
    if f_price_max > 0: filters["max_price"] = f_price_max
    
    results = advanced_search(
        st.session_state.bm25, 
        st.session_state.chunks, 
        prompt, 
        DEFAULT_TOPK, 
        filters
    )

    # 3. Context Construction
    context_str = ""
    for i, (c, score) in enumerate(results):
        context_str += f"[{i+1}] {c.title} | Price: {c.price_value} | Source: {c.source} | DocID: {c.doc_id}\n"
        # Add URL if available to context so LLM can reference it
        if c.url: context_str += f"Link: {c.url}\n"
        context_str += f"Desc: {c.text[:300]}...\n\n"

    # 4. LLM Generation (Streaming)
    client = _ensure_client()
    
    # System Prompt with Instructions
    sys_prompt = """
    You are a helpful shopping assistant. Use the Context provided to answer the user's question.
    - If the user asks for recommendations, list top 3-5 distinct items from context.
    - Mention the price (in BDT) and Source (Daraz/StarTech) clearly.
    - If the context has a 'Link', format it as markdown: [Link Text](URL).
    - If a 'Link' is missing in context, do NOT invent one.
    - Be conversational. If the answer isn't in context, say so politely.
    """
    
    # We pass the last few messages for conversation history
    history_msgs = st.session_state.messages[-4:] 
    
    full_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": f"### RETRIEVED CONTEXT:\n{context_str}"}
    ] + history_msgs

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=full_messages,
                stream=True,
                temperature=0.3
            )
            
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # 5. Append Bot Response to History
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # 6. Show "Source" cards below message (The "Next to look" feature)
            with st.expander("View Sources for this answer"):
                for c, s in results[:5]:
                    st.markdown(f"""
                    <div class="product-card">
                        <b>{c.title}</b><br>
                        <span class="price-tag">৳{c.price_value}</span> | {c.source}<br>
                        <a href="{c.url}" target="_blank">View Product</a>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating response: {e}")