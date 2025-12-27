# -*- coding: utf-8 -*-
"""
Daraz + StarTech RAG Chat App (Fixed: Singular/Plural Search)
=============================================================
Updates:
1. Added 'simple_tokenize' to handle plurals (laptop == laptops).
2. Robust Parsing to ensure all products are loaded.
3. Interactive Chat Interface.
"""

import os
import re
import json
import asyncio
import hashlib
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
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

# Apply nest_asyncio for Streamlit
nest_asyncio.apply()
load_dotenv()

# ----------------------------
# App Config
# ----------------------------
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 50

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

@dataclass
class ChunkRec:
    doc_id: str
    title: str
    source: str
    text: str
    metadata: Dict[str, Any]

# ----------------------------
# 1. HELPER: SMART TOKENIZER (The Fix)
# ----------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    Splits text into words and removes trailing 's' to match singular/plural.
    E.g. "Gaming Laptops" -> ["gaming", "laptop"]
    """
    if not text: return []
    words = re.findall(r'\w+', text.lower())
    # Simple stemming: remove 's' at end if word is long enough
    return [w[:-1] if (w.endswith('s') and len(w) > 3) else w for w in words]

# ----------------------------
# 2. ROBUST PARSING LOGIC
# ----------------------------

def parse_corpus_text(raw_text: str) -> List[ProductDoc]:
    """
    Parses the corpus using Regex. Robust against missing separators.
    """
    docs = []
    
    # Matches blocks: ## Title ... **DocID:** id ... (content) ... until next ##
    product_pattern = re.compile(
        r"(##\s*(?P<title>.+?)\n"          
        r"\*\*DocID:\*\*\s*`(?P<id>[^`]+)`" 
        r"(?P<content>[\s\S]+?))"           
        r"(?=\n##\s|\Z)",                   
        re.MULTILINE
    )

    for match in product_pattern.finditer(raw_text):
        try:
            full_block = match.group(1).strip()
            title = match.group("title").strip()
            doc_id = match.group("id").strip()
            content_body = match.group("content")

            # Extract Metadata
            category_m = re.search(r"\*\*Category:\*\*\s*(.+)", content_body)
            category = category_m.group(1).strip() if category_m else "Unknown"

            source_m = re.search(r"\*\*Source:\*\*\s*(.+)", content_body)
            source = source_m.group(1).strip() if source_m else "Unknown"

            url_m = re.search(r"\*\*URL:\*\*\s*(.+)", content_body)
            url = url_m.group(1).strip() if url_m else None

            price_m = re.search(r"\*\*Price:\*\*\s*([\d,.]+)", content_body)
            price_val = 0.0
            if price_m:
                try:
                    price_val = float(price_m.group(1).replace(",", ""))
                except:
                    pass

            docs.append(ProductDoc(
                doc_id=doc_id, title=title, source=source, category=category,
                price_value=price_val, url=url, raw_md=full_block
            ))
        except Exception as e:
            print(f"Skipping item: {e}")
            continue
            
    return docs

# ----------------------------
# 3. LIVE SCRAPING LOGIC
# ----------------------------

def get_startech_schema():
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
    return {
        "baseSelector": "div[data-qa-locator='product-item']",
        "fields": [
            {"name": "name", "selector": "a[title]", "type": "attribute", "attribute": "title"},
            {"name": "url", "selector": "a[href]", "type": "attribute", "attribute": "href"},
            {"name": "price", "selector": "span:not([style*='text-decoration'])", "type": "text"}, 
        ]
    }

async def crawl_category(url: str, source: str) -> List[Dict]:
    """Scrapes a single category page using crawl4ai."""
    
    if source == "StarTech":
        schema = get_startech_schema()
        strategy = JsonCssExtractionStrategy(schema)
        wait_for = "css:.p-item"
    else:
        schema = get_daraz_schema()
        strategy = JsonCssExtractionStrategy(schema)
        wait_for = "css:body"

    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS,
        wait_for_images=False,
        delay_before_return_html=True,
        mean_delay=1.5,
        verbose=True
    )

    products = []
    
    async with AsyncWebCrawler() as crawler:
        js_scroll = """
            window.scrollTo(0, document.body.scrollHeight/2);
            await new Promise(r => setTimeout(r, 1000));
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 1000));
        """
        
        try:
            result = await crawler.arun(url=url, config=config, js_code=js_scroll, wait_for=wait_for)
            
            if result.success:
                try:
                    data = json.loads(result.extracted_content)
                    if isinstance(data, list):
                        products.extend(data)
                    elif isinstance(data, dict):
                        products.append(data)
                except:
                    pass
                
                # Fallback Parsing
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
                        for item in soup.select("div[data-qa-locator='product-item']"):
                            a_tag = item.select_one("a[title]")
                            if a_tag:
                                products.append({
                                    "name": a_tag.get("title"),
                                    "url": a_tag.get("href"),
                                    "price": item.get_text() 
                                })
        except Exception as e:
            st.error(f"Crawl Error: {e}")

    return products

def process_scraped_data(raw_items: List[Dict], source: str, category_name: str) -> List[ProductDoc]:
    docs = []
    for item in raw_items:
        title = item.get("name") or "Unknown Product"
        raw_url = item.get("url") or ""
        
        if raw_url.startswith("//"): raw_url = "https:" + raw_url
        elif raw_url.startswith("/"):
             base = "https://www.startech.com.bd" if source == "StarTech" else "https://www.daraz.com.bd"
             raw_url = urljoin(base, raw_url)
             
        doc_id = f"{source.lower()}_{hashlib.md5((title+raw_url).encode()).hexdigest()[:8]}"
        
        price_raw = item.get("price", "")
        price_val = None
        nums = re.findall(r"[\d,]+", str(price_raw))
        if nums:
            try:
                price_val = float(nums[0].replace(",", ""))
            except:
                pass

        meta = [f"**Source:** {source}", f"**Category:** {category_name}"]
        if price_val: meta.append(f"**Price:** {price_val}")
        if raw_url: meta.append(f"**URL:** {raw_url}")
        
        raw_md = f"## {title}\n**DocID:** `{doc_id}`\n" + " • ".join(meta) + "\n---"

        docs.append(ProductDoc(
            doc_id=doc_id, title=title, source=source, category=category_name,
            price_value=price_val, url=raw_url, raw_md=raw_md
        ))
    return docs

# ----------------------------
# 4. RAG LOGIC (Index & Search)
# ----------------------------

def build_index(products: List[ProductDoc]):
    """Chunks products and builds BM25 index with SMART TOKENIZATION."""
    chunker = RecursiveChunker.from_recipe("markdown", lang="en")
    chunks = []
    
    for p in products:
        sub_chunks = chunker(p.raw_md)
        for c in sub_chunks:
            if not c.text.strip(): continue
            chunks.append(ChunkRec(
                doc_id=p.doc_id, title=p.title, source=p.source, text=c.text,
                metadata={"url": p.url, "price": p.price_value, "category": p.category}
            ))

    # USE SIMPLE_TOKENIZE HERE
    tokenized_corpus = [simple_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks

def get_rag_stream(query: str, bm25: BM25Okapi, chunks: List[ChunkRec], model: str, top_k: int):
    """Retrieves chunks and yields streaming response from OpenAI."""
    # USE SIMPLE_TOKENIZE HERE ALSO
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_chunks = [chunks[i] for i in top_n_indices]

    context_text = "\n\n".join([
        f"Product: {c.title}\nDocID: {c.doc_id}\nSource: {c.source}\nCategory: {c.metadata['category']}\nPrice: {c.metadata['price']}\nContent: {c.text}" 
        for c in top_chunks
    ])

    messages = [
        {"role": "system", "content": (
            "You are a helpful shopping assistant. "
            "Answer the user's question based ONLY on the provided Context Products. "
            "If the user asks for a recommendation, provide it and explain why. "
            "Always cite the product using its [DocID] when mentioning it. "
            "Format your answer in Markdown."
        )},
        {"role": "user", "content": f"User Query: {query}\n\nContext Products:\n{context_text}"}
    ]

    client = OpenAI()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        stream=True
    )
    return stream, top_chunks

# ----------------------------
# 5. STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="ShopRAG Chat", layout="wide", initial_sidebar_state="expanded")
st.title("🛍️ ShopRAG: Interactive Assistant")

# --- Sidebar: Data Setup ---
with st.sidebar:
    st.header("⚙️ Data Setup")
    mode = st.radio("Source Mode:", ["Load from GitHub API", "Live Scrape Category"])
    
    if mode == "Load from GitHub API":
        url_input = st.text_input("Corpus URL", value=DEFAULT_CORPUS_URL)
        if st.button("📥 Load Data"):
            with st.spinner("Fetching corpus..."):
                try:
                    resp = requests.get(url_input)
                    if resp.status_code == 200:
                        # Use robust parser
                        products_db = parse_corpus_text(resp.text)
                        
                        st.session_state['products'] = products_db
                        st.success(f"Successfully loaded {len(products_db)} products!")
                        
                        # Show categories for debug
                        cats = sorted(list(set(p.category for p in products_db)))
                        st.expander("Detected Categories").write(cats)
                    else:
                        st.error("Failed to load URL.")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif mode == "Live Scrape Category":
        st.info("Enter a category URL (Daraz or StarTech)")
        scrape_url = st.text_input("Category URL")
        
        if st.button("🕷️ Start Scraping"):
            source = "Daraz" if "daraz" in scrape_url.lower() else "StarTech" if "startech" in scrape_url.lower() else None
            
            if not source:
                st.error("Invalid URL. Must be Daraz or StarTech.")
            else:
                with st.spinner(f"Scraping {source}..."):
                    try:
                        raw_data = asyncio.run(crawl_category(scrape_url, source))
                        if raw_data:
                            cat_slug = urlparse(scrape_url).path.split('/')[-1] or "New Category"
                            new_docs = process_scraped_data(raw_data, source, cat_slug)
                            st.session_state['products'] = new_docs
                            st.success(f"Scraped {len(new_docs)} items!")
                        else:
                            st.warning("No products found.")
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")

    if 'products' in st.session_state and st.session_state['products']:
        st.markdown(f"**Active Corpus:** {len(st.session_state['products'])} items")
    else:
        st.warning("⚠️ No data loaded.")

# --- Main Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ready. Load data in the sidebar and ask away!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about products (e.g., 'best laptop under 50k')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if 'products' not in st.session_state or not st.session_state['products']:
        err_msg = "Please load data from the sidebar first!"
        with st.chat_message("assistant"):
            st.warning(err_msg)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
    else:
        products = st.session_state['products']
        if 'bm25' not in st.session_state or st.session_state.get('last_count') != len(products):
            with st.spinner("Indexing data..."):
                bm25, chunks = build_index(products)
                st.session_state['bm25'] = bm25
                st.session_state['chunks'] = chunks
                st.session_state['last_count'] = len(products)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                stream, used_chunks = get_rag_stream(
                    prompt, 
                    st.session_state['bm25'], 
                    st.session_state['chunks'], 
                    DEFAULT_MODEL, 
                    DEFAULT_TOPK
                )
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                
                source_text = "\n\n---\n**Sources used:**\n"
                for i, c in enumerate(used_chunks[:5], 1):
                    price_display = f"৳{c.metadata.get('price', 'N/A')}"
                    source_text += f"{i}. [{c.title}]({c.metadata['url'] or '#'}) - `{price_display}`\n"
                
                full_response += source_text
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")