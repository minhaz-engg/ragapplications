
import os
import re
import json
import asyncio
import hashlib
import pickle
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv

# --- AI & Search Imports ---
from openai import OpenAI
from rank_bm25 import BM25Okapi

# --- Scraping Imports ---
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, JsonCssExtractionStrategy

nest_asyncio.apply()
load_dotenv()

# ----------------------------
# 1. Configuration & Caching
# ----------------------------
CACHE_DIR = "rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 75

# ----------------------------
# 2. Data Structures
# ----------------------------
@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: str
    category: str
    price_value: float
    url: Optional[str]
    raw_md: str
    embedding: Optional[List[float]] = field(default=None)

@dataclass
class SearchResult:
    doc: ProductDoc
    score: float
    reason: str = ""

# ----------------------------
# 3. Robust Parsing & Tokenization (FIXED)
# ----------------------------

def simple_tokenize(text: str) -> List[str]:
    """Smart stemming: handles plurals (laptops -> laptop) for BM25."""
    if not text: return []
    words = re.findall(r'\w+', text.lower())
    return [w[:-1] if (w.endswith('s') and len(w) > 3) else w for w in words]

def parse_price(price_str: str) -> float:
    """
    Extracts price from messy strings.
    FIX: Replaces currency symbols with SPACE to prevent number merging.
    e.g. "13,500৳15,000৳" -> "13,500 15,000" -> Picks 13500.
    """
    if not price_str: return 0.0
    
    # Replace symbols with space to ensure separation
    # Handles "Tk.", "Tk", "৳", "BDT"
    clean_str = re.sub(r'(৳|Tk\.?|BDT)', ' ', str(price_str), flags=re.IGNORECASE)
    
    # Extract all number groups (handling commas)
    matches = re.findall(r'[\d,]+(?:\.\d+)?', clean_str)
    
    for match in matches:
        clean_num = match.replace(',', '')
        try:
            val = float(clean_num)
            # Filter out tiny numbers (like '1' year warranty) or obvious bad parses
            if val > 100: 
                return val
        except: continue
    return 0.0

def parse_corpus_text(raw_text: str, filter_source: str = "Both") -> List[ProductDoc]:
    """Parses the Markdown dataset with Source Filtering."""
    docs = []
    pattern = re.compile(
        r"(##\s*(?P<title>.+?)\n"          
        r"\*\*DocID:\*\*\s*`(?P<id>[^`]+)`" 
        r"(?P<content>[\s\S]+?))"           
        r"(?=\n##\s|\Z)", re.MULTILINE
    )

    for match in pattern.finditer(raw_text):
        try:
            full_block = match.group(1).strip()
            content = match.group("content")
            
            src_m = re.search(r"\*\*Source:\*\*\s*(.+)", content)
            src = src_m.group(1).strip() if src_m else "Unknown"

            # FILTERING LOGIC
            if filter_source != "Both":
                if filter_source.lower() not in src.lower():
                    continue

            title = match.group("title").strip()
            doc_id = match.group("id").strip()
            
            cat_m = re.search(r"\*\*Category:\*\*\s*(.+)", content)
            cat = cat_m.group(1).strip() if cat_m else "General"
            
            url_m = re.search(r"\*\*URL:\*\*\s*(.+)", content)
            url = url_m.group(1).strip() if url_m else "#"
            
            price_m = re.search(r"\*\*Price:\*\*\s*(.+)", content)
            price = parse_price(price_m.group(1)) if price_m else 0.0

            docs.append(ProductDoc(doc_id, title, src, cat, price, url, full_block))
        except: continue
    return docs

# ----------------------------
# 4. Deep Live Scraping (FIXED)
# ----------------------------

async def crawl_category(url: str, source: str) -> List[ProductDoc]:
    """Deep Scraper: Aggressive scrolling + Safe Price Extraction."""
    print(f"🕷️ Crawling {source}: {url}")
    
    if source == "StarTech":
        schema = {
            "baseSelector": ".p-item",
            "fields": [
                {"name": "name", "selector": "h4.p-item-name a", "type": "text"},
                {"name": "url", "selector": "h4.p-item-name a", "type": "attribute", "attribute": "href"},
                {"name": "price", "selector": ".p-item-price", "type": "text"}
            ]
        }
        wait_for = "css:.p-item"
    else: # Daraz
        schema = {
            "baseSelector": "div[data-qa-locator='product-item']",
            "fields": [
                {"name": "name", "selector": "a[title]", "type": "attribute", "attribute": "title"},
                {"name": "url", "selector": "a[href]", "type": "attribute", "attribute": "href"},
                {"name": "price", "selector": "span:not([style])", "type": "text"}
            ]
        }
        wait_for = "css:body"

    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema),
        cache_mode=CacheMode.BYPASS,
        wait_for_images=False,
        verbose=True
    )

    # Aggressive Scroll (20 loops)
    js_scroll = """
        let lastH = 0;
        for(let i=0; i<20; i++) {
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 800));
            window.scrollBy(0, -200); 
            await new Promise(r => setTimeout(r, 400));
            window.scrollTo(0, document.body.scrollHeight);
            
            if(document.body.scrollHeight === lastH && i > 5) break;
            lastH = document.body.scrollHeight;
        }
    """

    raw_items = []
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config, js_code=js_scroll, wait_for=wait_for)
        if result.success:
            try:
                data = json.loads(result.extracted_content)
                if isinstance(data, list): raw_items.extend(data)
                elif isinstance(data, dict): raw_items.append(data)
            except: pass
            
            # Fallback for Missing/Partial JSON
            if not raw_items and result.html:
                soup = BeautifulSoup(result.html, 'html.parser')
                if source == "Daraz":
                    for card in soup.select("div[data-qa-locator='product-item']"):
                        try:
                            raw_items.append({
                                "name": card.select_one("a[title]")['title'],
                                "url": card.select_one("a[href]")['href'],
                                "price": card.select_one("span:not([style])").get_text(strip=True)
                            })
                        except: continue
                elif source == "StarTech":
                    for card in soup.select(".p-item"):
                        try:
                            # FIX: Use separator=" " to prevent merging "13,000" and "15,000"
                            price_txt = card.select_one(".p-item-price").get_text(separator=" ", strip=True)
                            raw_items.append({
                                "name": card.select_one("h4 a").get_text(strip=True),
                                "url": card.select_one("h4 a")['href'],
                                "price": price_txt
                            })
                        except: continue

    docs = []
    cat_name = urlparse(url).path.split('/')[-1] or "Scraped-Session"
    
    for item in raw_items:
        title = item.get('name', 'Unknown')
        raw_url = item.get('url', '')
        if raw_url.startswith("//"): raw_url = "https:" + raw_url
        elif raw_url.startswith("/"):
             base = "https://www.startech.com.bd" if source == "StarTech" else "https://www.daraz.com.bd"
             raw_url = urljoin(base, raw_url)
        
        price = parse_price(item.get('price', ''))
        doc_id = f"{source}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        raw_md = f"## {title}\n**DocID:** `{doc_id}`\n**Category:** {cat_name}\n**Price:** {price}\n**Source:** {source}\n**URL:** {raw_url}\n---"
        docs.append(ProductDoc(doc_id, title, source, cat_name, price, raw_url, raw_md))
        
    return docs

# ----------------------------
# 5. Hybrid Search Engine
# ----------------------------

class HybridSearchEngine:
    def __init__(self, products: List[ProductDoc]):
        self.products = products
        self.client = OpenAI()
        self.bm25 = None
        self.corpus_embeddings = None
        self.categories: Set[str] = set()
        
        # Init
        self.update_categories()
        self.rebuild_bm25()
        
        # Fresh embeddings for Live Scrape (small), Cache for Dataset (large)
        if len(products) < 500:
             self.generate_embeddings_fresh(use_cache=False)
        else:
             self.load_or_generate_embeddings()

    def update_categories(self):
        self.categories = sorted(list(set(p.category for p in self.products)))

    def rebuild_bm25(self):
        tokens = [simple_tokenize(p.title + " " + p.category) for p in self.products]
        self.bm25 = BM25Okapi(tokens)

    def load_or_generate_embeddings(self):
        if not self.products: return
        content_hash = hashlib.md5("".join([p.doc_id for p in self.products]).encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"emb_{content_hash}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.corpus_embeddings = pickle.load(f)
            for i, p in enumerate(self.products):
                if i < len(self.corpus_embeddings): p.embedding = self.corpus_embeddings[i]
        else:
            self.generate_embeddings_fresh(use_cache=True)

    def generate_embeddings_fresh(self, use_cache=True):
        if not self.products: return
        
        msg = "Vectorizing Scraped Data..." if len(self.products) < 500 else "Indexing Dataset..."
        if len(self.products) > 100:
            progress = st.progress(0, text=msg)
        
        texts = [f"{p.title} {p.category} Price: {p.price_value}" for p in self.products]
        all_embs = []
        batch_size = 200
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = self.client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                all_embs.extend([d.embedding for d in resp.data])
            except: 
                all_embs.extend([[0.0]*1536] * len(batch))
            if len(self.products) > 100:
                progress.progress(min((i+batch_size)/len(texts), 1.0))
        
        if len(self.products) > 100: progress.empty()
        
        self.corpus_embeddings = np.array(all_embs)
        
        for i, p in enumerate(self.products):
            p.embedding = all_embs[i]

        if use_cache:
            content_hash = hashlib.md5("".join([p.doc_id for p in self.products]).encode()).hexdigest()
            with open(os.path.join(CACHE_DIR, f"emb_{content_hash}.pkl"), "wb") as f:
                pickle.dump(self.corpus_embeddings, f)

    def search(self, query: str, filters: Dict, top_k: int = TOP_K) -> List[SearchResult]:
        # 1. Hard Filtering
        valid_indices = []
        for i, p in enumerate(self.products):
            # Price Filter
            if filters.get('max_price') and p.price_value > filters['max_price']: continue
            if filters.get('min_price') and p.price_value < filters['min_price']: continue
            # Category Filter
            if filters.get('category'):
                q_cat = filters['category'].lower()
                p_cat = p.category.lower()
                if q_cat not in p_cat and p_cat not in q_cat:
                    if "laptop" in q_cat and ("macbook" in p_cat or "notebook" in p_cat): pass
                    else: continue
            valid_indices.append(i)

        if not valid_indices: return []

        # 2. Vector Search
        q_emb = self.client.embeddings.create(input=query, model=EMBEDDING_MODEL).data[0].embedding
        valid_embs = self.corpus_embeddings[valid_indices]
        sem_scores = np.dot(valid_embs, np.array(q_emb))

        # 3. Keyword Search
        q_tok = simple_tokenize(query)
        bm25_full = self.bm25.get_scores(q_tok)
        kw_scores = np.array([bm25_full[i] for i in valid_indices])

        # 4. Fusion
        def norm(arr):
            if len(arr) == 0 or np.max(arr) == np.min(arr): return np.zeros_like(arr)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        final_scores = (0.7 * norm(sem_scores)) + (0.3 * norm(kw_scores))

        results = []
        for idx_in_valid, score in enumerate(final_scores):
            results.append(SearchResult(self.products[valid_indices[idx_in_valid]], score))

        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

# ----------------------------
# 6. Streamlit UI
# ----------------------------

st.set_page_config(page_title="ShopRAG Ultra", layout="wide", initial_sidebar_state="expanded")
st.title("🛍️ ShopRAG Ultra: Advanced Hybrid Search")

if "engine" not in st.session_state: st.session_state.engine = None
if "mode" not in st.session_state: st.session_state.mode = "Not Loaded"

with st.sidebar:
    st.header("1. Data Engine")
    
    # --- LOAD ---
    with st.expander("📥 Load Dataset", expanded=True):
        url_in = st.text_input("Dataset URL", value=DEFAULT_CORPUS_URL)
        
        st.write("Select Source Preference:")
        source_pref = st.radio(
            "Select Source",
            ["Both", "Daraz", "StarTech"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if st.button("Initialize Engine"):
            with st.spinner(f"Loading {source_pref} Data..."):
                try:
                    resp = requests.get(url_in)
                    if resp.status_code == 200:
                        docs = parse_corpus_text(resp.text, filter_source=source_pref)
                        if not docs:
                            st.warning(f"No data found for {source_pref}.")
                        else:
                            st.session_state.engine = HybridSearchEngine(docs)
                            st.session_state.mode = f"Dataset ({source_pref})"
                            st.success(f"Indexed {len(docs)} items from {source_pref}!")
                    else: st.error("Failed to fetch.")
                except Exception as e: st.error(str(e))

    # --- INFO ---
    if st.session_state.engine:
        st.divider()
        st.write(f"**Current Mode:** `{st.session_state.mode}`")
        st.write(f"**Items:** {len(st.session_state.engine.products)}")
        with st.expander("Active Categories"):
            st.write(st.session_state.engine.categories)

    # --- SCRAPE ---
    with st.expander("🕷️ Live Scraper"):
        st.info("Paste Daraz/StarTech category URL")
        scrape_url = st.text_input("URL")
        
        if st.button("Scrape & Use This Data"):
            source = "StarTech" if "startech" in scrape_url.lower() else "Daraz"
            with st.spinner(f"Deep scraping {source}..."):
                try:
                    new_docs = asyncio.run(crawl_category(scrape_url, source))
                    if new_docs:
                        st.session_state.engine = HybridSearchEngine(new_docs)
                        st.session_state.mode = f"Live Scrape ({len(new_docs)} items)"
                        st.success(f"Loaded {len(new_docs)} items! You can now chat about this link.")
                    else: st.warning("No items found.")
                except Exception as e: st.error(f"Error: {e}")

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am ready with Hybrid Search. Select data or scrape a link!"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ex: Best gaming laptop under 1 lakh..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not st.session_state.engine:
        st.error("Please load data or scrape a link first!")
    else:
        with st.chat_message("assistant"):
            # 1. PRE-PROCESS
            client = OpenAI()
            intent_prompt = (
                "Extract filters from user query.\n"
                "JSON: {\"query\": string, \"filters\": {\"max_price\": int, \"min_price\": int, \"category\": string}}\n"
                f"User: {prompt}"
            )
            
            with st.spinner("Processing..."):
                try:
                    raw = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system", "content":"JSON only."}, {"role":"user", "content":intent_prompt}],
                        response_format={"type": "json_object"}
                    )
                    intent = json.loads(raw.choices[0].message.content)
                    q_text = intent.get("query", prompt)
                    filters = intent.get("filters", {})
                except: q_text, filters = prompt, {}
            
            with st.expander(f"Debugging", expanded=False):
                st.json({"Mode": st.session_state.mode, "Query": q_text, "Filters": filters})

            # 2. SEARCH
            results = st.session_state.engine.search(q_text, filters, top_k=TOP_K)

            if not results:
                st.warning(f"No products found in '{st.session_state.mode}'.")
            else:
                # 3. GENERATE
                ctx = "\n".join([f"- {r.doc.title} | ৳{r.doc.price_value:,.0f} | {r.doc.source} | [DocID: {r.doc.doc_id}]" for r in results])
                
                gen_prompt = (
                    "You are a sales expert. Use the Product List to answer.\n"
                    "1. Recommend top 5 options.\n"
                    "2. State Price (৳) and Source, and Product link.\n"
                    "3. Cite [DocID].\n"
                    f"User: {prompt}\nList:\n{ctx}"
                )
                
                stream = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role":"user", "content": gen_prompt}],
                    stream=True
                )
                st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": "..."})