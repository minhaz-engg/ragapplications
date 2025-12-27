# -*- coding: utf-8 -*-
"""
Daraz + StarTech RAG App (Live Scraping + OpenAI)
=================================================
Features:
1. Load existing corpus from URL.
2. LIVE SCRAPE a new Category URL (Daraz or StarTech) via crawl4ai.
3. RAG with OpenAI + Chonkie + BM25.
"""

import os
import re
import json
import asyncio
import pickle
import hashlib
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv

# --- RAG & AI Imports ---
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker

# --- Scraping Imports ---
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, JsonCssExtractionStrategy

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()
load_dotenv()

# ----------------------------
# App Config
# ----------------------------
INDEX_DIR = "index_cache"
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 10

# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: str
    category: str
    price_value: Optional[float]
    url: Optional[str]
    raw_md: str
    rating_avg: Optional[float] = None
    rating_cnt: Optional[int] = None

@dataclass
class ChunkRec:
    doc_id: str
    title: str
    source: str
    text: str
    metadata: Dict[str, Any]

# ----------------------------
# 1. LIVE SCRAPING LOGIC
# ----------------------------

def normalize_price(price_str: str) -> str:
    """Clean price string for display."""
    if not price_str: return "N/A"
    return re.sub(r"[^\d,.]", "", price_str)

def get_startech_schema():
    # StarTech is often SSR friendly, but we use crawl4ai as requested for consistency
    return {
        "baseSelector": ".p-item",
        "fields": [
            {"name": "name", "selector": "h4.p-item-name a", "type": "text"},
            {"name": "url", "selector": "h4.p-item-name a", "type": "attribute", "attribute": "href"},
            {"name": "price", "selector": ".p-item-price span", "type": "text"},
            {"name": "status", "selector": ".p-item-stock", "type": "text"}
        ]
    }

def get_daraz_schema():
    # Hardcoded CSS schema for Daraz to avoid Gemini dependency
    # Note: Selectors (.Bm3ON, .RfADt etc) are based on common Daraz obfuscated classes 
    # observed in your uploaded files. If Daraz changes classes, this needs update.
    return {
        "baseSelector": "div[data-qa-locator='product-item']",
        "fields": [
            {"name": "name", "selector": "a[title]", "type": "attribute", "attribute": "title"},
            {"name": "url", "selector": "a[href]", "type": "attribute", "attribute": "href"},
            {"name": "price", "selector": "span:not([style*='text-decoration'])", "type": "text"}, 
            # Note: Price selector is tricky in schema without complex logic, 
            # we often grab the first visible span in the price container.
        ]
    }

async def crawl_category(url: str, source: str) -> List[Dict]:
    """
    Scrapes a single category page using crawl4ai. 
    Returns raw product dictionaries.
    """
    print(f"Starting scrape for {source}: {url}")
    
    # Configure extraction based on source
    if source == "StarTech":
        # StarTech CSS Strategy
        schema = get_startech_schema()
        strategy = JsonCssExtractionStrategy(schema)
        wait_for = "css:.p-item"
    else:
        # Daraz Strategy (Basic CSS extraction)
        # Often Daraz classes are dynamic. For reliability in this "Simple" app,
        # we might rely on the raw HTML and BS4 parsing if Schema fails, 
        # but let's try a generic strategy first.
        schema = get_daraz_schema()
        strategy = JsonCssExtractionStrategy(schema)
        wait_for = "css:body"

    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS,
        wait_for_images=False,
        delay_before_return_html=True,
        mean_delay=2.0,
        verbose=True
    )

    products = []
    
    async with AsyncWebCrawler() as crawler:
        # We scroll to trigger lazy loading
        js_scroll = """
            window.scrollTo(0, document.body.scrollHeight/2);
            await new Promise(r => setTimeout(r, 1000));
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 1000));
        """
        
        result = await crawler.arun(url=url, config=config, js_code=js_scroll, wait_for=wait_for)
        
        if result.success:
            # 1. Try extracted JSON
            try:
                data = json.loads(result.extracted_content)
                if isinstance(data, list):
                    products.extend(data)
                elif isinstance(data, dict):
                    products.append(data)
            except:
                pass
            
            # 2. Fallback: Parse raw HTML with BeautifulSoup if extraction failed or returned empty
            # (Crucial for sites like Daraz where CSS modules change hash)
            if not products and result.html:
                soup = BeautifulSoup(result.html, 'html.parser')
                if source == "StarTech":
                    for item in soup.select(".p-item"):
                        name_tag = item.select_one("h4 a")
                        price_tag = item.select_one(".p-item-price")
                        if name_tag:
                            products.append({
                                "name": name_tag.get_text(strip=True),
                                "url": name_tag.get('href'),
                                "price": price_tag.get_text(strip=True) if price_tag else "N/A"
                            })
                elif source == "Daraz":
                    # Generic fallback for Daraz Grid
                    for item in soup.select("div[data-qa-locator='product-item']"):
                        # Try finding title in anchors
                        a_tag = item.select_one("a[title]")
                        price_tag = item.select_one("span") # Heuristic
                        if a_tag:
                            products.append({
                                "name": a_tag.get("title"),
                                "url": a_tag.get("href"),
                                "price": item.get_text() # Dirty capture, cleaned later
                            })

    return products

def process_scraped_data(raw_items: List[Dict], source: str, category_name: str) -> List[ProductDoc]:
    """Convert raw scraped dicts into RAG ProductDocs."""
    docs = []
    for item in raw_items:
        title = item.get("name") or "Unknown Product"
        raw_url = item.get("url") or ""
        
        # Normalize URL
        if raw_url.startswith("//"): raw_url = "https:" + raw_url
        elif raw_url.startswith("/"):
             base = "https://www.startech.com.bd" if source == "StarTech" else "https://www.daraz.com.bd"
             raw_url = urljoin(base, raw_url)
             
        # Generate ID
        doc_id = f"{source.lower()}_{hashlib.md5((title+raw_url).encode()).hexdigest()[:8]}"
        
        # Price parsing
        price_raw = item.get("price", "")
        price_val = None
        # Extract first number
        nums = re.findall(r"[\d,]+", str(price_raw))
        if nums:
            try:
                price_val = float(nums[0].replace(",", ""))
            except:
                pass

        # Create Markdown representation for RAG
        meta = [f"**Source:** {source}", f"**Category:** {category_name}"]
        if price_val: meta.append(f"**Price:** {price_val}")
        if raw_url: meta.append(f"**URL:** {raw_url}")
        
        raw_md = f"## {title}\n**DocID:** `{doc_id}`\n" + " • ".join(meta) + "\n---"

        docs.append(ProductDoc(
            doc_id=doc_id,
            title=title,
            source=source,
            category=category_name,
            price_value=price_val,
            url=raw_url,
            raw_md=raw_md
        ))
    return docs

# ----------------------------
# 2. RAG LOGIC (Chonkie + BM25 + OpenAI)
# ----------------------------

def build_index(products: List[ProductDoc]):
    """Chunks products and builds BM25 index."""
    chunker = RecursiveChunker.from_recipe("markdown", lang="en")
    chunks = []
    
    for p in products:
        # Simple chunking: usually 1 product = 1 chunk due to size
        sub_chunks = chunker(p.raw_md)
        for c in sub_chunks:
            if not c.text.strip(): continue
            chunks.append(ChunkRec(
                doc_id=p.doc_id,
                title=p.title,
                source=p.source,
                text=c.text,
                metadata={"url": p.url, "price": p.price_value, "category": p.category}
            ))

    # Tokenize for BM25
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks

def query_rag(query: str, bm25: BM25Okapi, chunks: List[ChunkRec], model: str, top_k: int):
    """Retrieves chunks and asks OpenAI."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_chunks = [chunks[i] for i in top_n_indices]

    # Build Context
    context_text = "\n\n".join([
        f"Product: {c.title}\nDocID: {c.doc_id}\nSource: {c.source}\nContent: {c.text}" 
        for c in top_chunks
    ])

    messages = [
        {"role": "system", "content": "You are a helpful shopping assistant. Answer based ONLY on the provided context. Cite products using [DocID]. Format as Markdown."},
        {"role": "user", "content": f"User Query: {query}\n\nContext Products:\n{context_text}"}
    ]

    client = OpenAI() # Expects OPENAI_API_KEY in env
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        stream=True
    )
    return response, top_chunks

# ----------------------------
# 3. STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="ShopRAG: Live & Loaded", layout="wide")
st.title("🛍️ ShopRAG: Daraz & StarTech (Live Scraping Supported)")

# --- Sidebar: Data Source ---
with st.sidebar:
    st.header("1. Data Source")
    mode = st.radio("Choose Mode:", ["Load from GitHub API", "Live Scrape Category"])
    
    products_db = []
    
    if mode == "Load from GitHub API":
        url_input = st.text_input("Corpus URL", value=DEFAULT_CORPUS_URL)
        if st.button("Load Data"):
            with st.spinner("Fetching corpus..."):
                try:
                    resp = requests.get(url_input)
                    if resp.status_code == 200:
                        # Parse the MD file (using regex similar to provided files)
                        # Minimal parser for the demo
                        raw_text = resp.text
                        # Simple split by ---
                        items = raw_text.split("---")
                        parsed_count = 0
                        for item in items:
                            if "DocID" not in item: continue
                            # Extract minimal info for the Object
                            title_m = re.search(r"##\s*(.+)", item)
                            title = title_m.group(1).strip() if title_m else "Unknown"
                            doc_id_m = re.search(r"DocID:\*\*\s*`?([^`\n]+)", item)
                            doc_id = doc_id_m.group(1) if doc_id_m else "unknown"
                            products_db.append(ProductDoc(
                                doc_id=doc_id, title=title, source="Preloaded", 
                                category="Mixed", price_value=0, url=None, raw_md=item
                            ))
                        st.session_state['products'] = products_db
                        st.success(f"Loaded {len(products_db)} products.")
                    else:
                        st.error("Failed to load URL.")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif mode == "Live Scrape Category":
        st.info("Enter a category URL from Daraz.com.bd or Startech.com.bd")
        scrape_url = st.text_input("Category URL")
        
        if st.button("Start Scraping"):
            if "daraz" in scrape_url.lower():
                source = "Daraz"
            elif "startech" in scrape_url.lower():
                source = "StarTech"
            else:
                st.error("URL must be from Daraz or StarTech.")
                source = None
            
            if source:
                with st.spinner(f"Scraping {source} (this may take 10-20s)..."):
                    try:
                        # Run async scraper in sync streamlit
                        raw_data = asyncio.run(crawl_category(scrape_url, source))
                        
                        if raw_data:
                            category_slug = urlparse(scrape_url).path.split('/')[-1] or "New Category"
                            new_docs = process_scraped_data(raw_data, source, category_slug)
                            st.session_state['products'] = new_docs
                            st.success(f"Scraped {len(new_docs)} items from {source}!")
                        else:
                            st.warning("Scraper finished but found no products. Check URL or selectors.")
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")

# --- Main Content: Indexing & Query ---
st.header("2. AI Search")

if 'products' in st.session_state and st.session_state['products']:
    products = st.session_state['products']
    
    # Auto-build index if new data arrives
    if 'bm25' not in st.session_state or st.session_state.get('last_count') != len(products):
        with st.spinner("Building Search Index..."):
            bm25, chunks = build_index(products)
            st.session_state['bm25'] = bm25
            st.session_state['chunks'] = chunks
            st.session_state['last_count'] = len(products)
    
    query = st.text_input("Ask about the products:", placeholder="e.g. Which laptop is best under 50k?")
    
    if st.button("Search") and query:
        st.subheader("Answer")
        resp_stream, used_chunks = query_rag(
            query, 
            st.session_state['bm25'], 
            st.session_state['chunks'], 
            DEFAULT_MODEL, 
            DEFAULT_TOPK
        )
        st.write_stream(resp_stream)
        
        with st.expander("View Source Products"):
            for c in used_chunks:
                st.markdown(f"**{c.title}**\n\n{c.text}\n\n---")

else:
    st.info("👈 Please load data or scrape a URL from the sidebar first.")