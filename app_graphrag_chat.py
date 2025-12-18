# -*- coding: utf-8 -*-
"""
OmniShop AI: Balanced GraphRAG (Dual-Index Edition)
===================================================

Fixes:
1. **Dual-Index Retrieval**: Maintains separate BM25 indices for Daraz and StarTech. 
   This prevents Daraz's long descriptions from drowning out StarTech's shorter listings.
2. **Auto-Brand Extraction**: Heuristically extracts brands from titles for StarTech items 
   (which lack explicit brand tags) to enable GraphRAG connections.
3. **Guaranteed Balancing**: Results are fetched separately and forced into a 50/50 mix.

"""

import os
import re
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Set
from collections import defaultdict, deque

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
import requests
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# App Config
# ----------------------------
PAGE_TITLE = "🛍️ OmniShop AI: Balanced GraphRAG"
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
DEFAULT_MODEL = "gpt-4o-mini"
TOP_K_RETRIEVAL = 20  # Total items to retrieve (10 from each source)

# ----------------------------
# Data Structures
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

# ----------------------------
# 1. Parsing & Data Loading
# ----------------------------

def fetch_data(url: str):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return ""

def parse_price(price_str: str) -> float:
    if not price_str:
        return 0.0
    # Extract numbers, removing commas and currency symbols
    nums = re.findall(r'\d+', price_str.replace(',', ''))
    if nums:
        return float("".join(nums))
    return 0.0

def infer_brand(title: str) -> str:
    """Heuristic: First word of the title is often the brand for tech products."""
    if not title:
        return "generic"
    parts = title.split()
    return parts[0].lower() if parts else "generic"

def parse_corpus(text: str) -> List[ProductDoc]:
    """
    Robust parsing that handles missing fields (like Brand in StarTech)
    by inferring them from the Title.
    """
    # Split by the markdown separator
    product_blocks = re.split(r'\n---\n', text)
    
    docs = []
    
    # Regex patterns
    re_docid = re.compile(r"\*\*DocID:\*\*\s*`([^`]+)`")
    re_source = re.compile(r"\*\*Source:\*\*\s*(.+)")
    re_cat = re.compile(r"\*\*Category:\*\*\s*(.+)")
    re_brand = re.compile(r"\*\*Brand:\*\*\s*(.+)")
    re_url = re.compile(r"\*\*URL:\*\*\s*(\S+)")
    re_price = re.compile(r"\*\*Price:\*\*\s*(.+)")
    re_title = re.compile(r"^##\s*(.+)", re.MULTILINE)

    for block in product_blocks:
        block = block.strip()
        if not block:
            continue
            
        # Basic Validation
        doc_id_match = re_docid.search(block)
        if not doc_id_match:
            continue 
            
        doc_id = doc_id_match.group(1).strip()
        title_match = re_title.search(block)
        title = title_match.group(1).strip() if title_match else "Unknown Product"
        
        source_match = re_source.search(block)
        source = source_match.group(1).strip() if source_match else "Unknown"
        
        cat_match = re_cat.search(block)
        category = cat_match.group(1).strip().lower() if cat_match else "general"
        
        # Smart Brand Extraction
        brand_match = re_brand.search(block)
        if brand_match:
            brand = brand_match.group(1).strip().lower()
        else:
            # Fallback: Infer brand from title (Crucial for StarTech)
            brand = infer_brand(title)
        
        url_match = re_url.search(block)
        url = url_match.group(1).strip() if url_match else ""
        
        price_match = re_price.search(block)
        price_val = parse_price(price_match.group(1)) if price_match else 0.0
        
        docs.append(ProductDoc(
            doc_id=doc_id,
            title=title,
            source=source,
            category=category,
            brand=brand,
            price_val=price_val,
            url=url,
            raw_text=block
        ))
        
    return docs

# ----------------------------
# 2. Dual-Index GraphRAG Engine
# ----------------------------

class GraphRAGIndex:
    """
    Maintains TWO BM25 indices (Daraz & StarTech) to ensure representation.
    Maintains ONE NetworkX Graph for cross-source connections.
    """
    def __init__(self, docs: List[ProductDoc]):
        self.doc_map = {d.doc_id: d for d in docs}
        self.graph = nx.Graph()
        
        # Split datasets for balancing
        self.docs_daraz = [d for d in docs if 'daraz' in d.source.lower()]
        self.docs_startech = [d for d in docs if 'startech' in d.source.lower()]
        
        # Build Dual Indices
        self.bm25_daraz = self._build_bm25(self.docs_daraz)
        self.bm25_startech = self._build_bm25(self.docs_startech)
        
        # Build Unified Graph
        self._build_knowledge_graph(docs)

    def _build_bm25(self, doc_list: List[ProductDoc]):
        """Creates a BM25 index for a specific list of docs."""
        if not doc_list:
            return None
        # We index Title + Brand + Category heavily, Description lightly
        tokenized = []
        for d in doc_list:
            # Boosting Title and Brand in the index
            text = f"{d.title} {d.title} {d.brand} {d.category} {d.raw_text}" 
            tokenized.append(text.lower().split())
        return BM25Okapi(tokenized)

    def _build_knowledge_graph(self, docs: List[ProductDoc]):
        """Nodes: Products, Brands, Categories."""
        for doc in docs:
            self.graph.add_node(doc.doc_id, type='product', source=doc.source)
            
            # Brand Link
            if doc.brand and doc.brand != 'generic':
                brand_node = f"BRAND:{doc.brand}"
                self.graph.add_node(brand_node, type='brand')
                self.graph.add_edge(doc.doc_id, brand_node)
            
            # Category Link
            if doc.category:
                cat_node = f"CAT:{doc.category}"
                self.graph.add_node(cat_node, type='category')
                self.graph.add_edge(doc.doc_id, cat_node)

    def search(self, query: str, total_k: int = 20) -> List[ProductDoc]:
        """
        1. Searches Daraz Index.
        2. Searches StarTech Index.
        3. Interleaves results.
        4. Expands top hits using GraphRAG (Brand/Category neighbors).
        """
        half_k = total_k // 2
        
        # -- Step 1: Independent Search --
        daraz_hits = self._query_bm25(self.bm25_daraz, self.docs_daraz, query, half_k)
        startech_hits = self._query_bm25(self.bm25_startech, self.docs_startech, query, half_k)
        
        # -- Step 2: Interleave (Round Robin) --
        # Ensure 50/50 mix
        combined_candidates = []
        max_len = max(len(daraz_hits), len(startech_hits))
        for i in range(max_len):
            if i < len(daraz_hits):
                combined_candidates.append(daraz_hits[i])
            if i < len(startech_hits):
                combined_candidates.append(startech_hits[i])
                
        # -- Step 3: Graph Expansion --
        # Find siblings of the top matches
        graph_hits = set()
        if combined_candidates:
            # Use top 3 best matches to find neighbors
            seeds = combined_candidates[:3]
            for seed in seeds:
                try:
                    neighbors = list(self.graph.neighbors(seed.doc_id))
                    for node in neighbors:
                        # If node is Brand or Category, get its other products
                        siblings = list(self.graph.neighbors(node))
                        for sib_id in siblings:
                            if sib_id != seed.doc_id and sib_id in self.doc_map:
                                graph_hits.add(self.doc_map[sib_id])
                except:
                    pass # Node might not exist in graph if generic
        
        # Add a few graph hits to the end if not already present
        final_results = []
        seen_ids = set()
        
        # Add ranked keyword results first
        for doc in combined_candidates:
            if doc.doc_id not in seen_ids:
                final_results.append(doc)
                seen_ids.add(doc.doc_id)
        
        # Add graph discoveries (limit 5)
        for doc in list(graph_hits)[:5]:
            if doc.doc_id not in seen_ids:
                final_results.append(doc)
                seen_ids.add(doc.doc_id)
                
        return final_results

    def _query_bm25(self, bm25_idx, doc_source, query, k):
        """Helper to query a specific index."""
        if not bm25_idx or not doc_source:
            return []
        tokenized_query = query.lower().split()
        scores = bm25_idx.get_scores(tokenized_query)
        # Sort and take top k
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # Filter out zero-score results (irrelevant)
        return [doc_source[i] for i in top_n if scores[i] > 0]

# ----------------------------
# 3. Streamlit UI
# ----------------------------

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(f"🕸️ {PAGE_TITLE}")
    st.caption("Dual-Index Engine: Guarantees visibility for both **Daraz** (Blue) and **StarTech** (Red).")

    # -- Sidebar --
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        if st.button("🔄 Rebuild Index"):
            st.cache_resource.clear()
            st.rerun()

    # -- Initialize Index --
    if "graph_index" not in st.session_state:
        with st.spinner("Fetching data and building Dual Indices..."):
            raw_text = fetch_data(DEFAULT_CORPUS_URL)
            if not raw_text:
                st.stop()
            
            docs = parse_corpus(raw_text)
            index = GraphRAGIndex(docs)
            st.session_state["graph_index"] = index
            
            d_count = len(index.docs_daraz)
            s_count = len(index.docs_startech)
            st.success(f"Indexed {len(docs)} products. (Daraz: {d_count}, StarTech: {s_count})")

    # -- Chat History --
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I am ready! I search Daraz and StarTech separately to ensure you get the best deal from both."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -- Input --
    if prompt := st.chat_input("Search for laptops, watches, or gadgets..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieval
        index: GraphRAGIndex = st.session_state["graph_index"]
        results = index.search(prompt, total_k=TOP_K_RETRIEVAL)
        
        if not results:
            st.warning("No products found matching your query.")
            return

        # Prepare Context
        context_str = ""
        for i, doc in enumerate(results, 1):
            context_str += f"[{i}] ({doc.source}) {doc.title} - {doc.price_val} BDT\nLink: {doc.url}\n\n"

        # LLM Generation
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        
        with st.chat_message("assistant"):
            if not client:
                st.write("### Retrieved Products (LLM Disabled - No Key)")
                for doc in results:
                    color = "blue" if "daraz" in doc.source.lower() else "red"
                    st.markdown(f":{color}[**{doc.source}**] [{doc.title}]({doc.url}) - **{doc.price_val}৳**")
            else:
                stream_box = st.empty()
                full_resp = ""
                
                system_prompt = (
                    "You are a shopping assistant. Use the provided Context to answer.\n"
                    "ALWAYS compare products from Daraz and StarTech if available.\n"
                    "Be objective about price and specs."
                )
                
                try:
                    stream = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"}
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
                    st.error(f"LLM Error: {e}")

                # Debug View
                with st.expander("Debugging: View Retrieval Source Balance"):
                    for doc in results:
                        color = "blue" if "daraz" in doc.source.lower() else "red"
                        st.markdown(f":{color}[{doc.source}] {doc.title}")

if __name__ == "__main__":
    main()