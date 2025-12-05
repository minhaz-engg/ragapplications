# try_lightrag_core.py
import os, re, asyncio, inspect, shutil
from dotenv import load_dotenv

load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./lightrag_cache"
COMBINED_MD = "./combined_corpus.md"  # put your file here (same folder as this script)
CLEAN_FIRST = False  # set True once if you want to wipe previous cache

# --- helpers ---------------------------------------------------------------

def read_items_from_md(path: str) -> list[str]:
    """
    Your corpus is a big markdown file with many items, separated by lines '---'.
    We keep each item block as a separate document so LightRAG can dedupe & link them well.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # split on lines that are only --- (with optional surrounding whitespace)
    parts = re.split(r"\n\s*---\s*\n", text)
    # keep only non-empty, minimally-sized items
    parts = [p.strip() for p in parts if p.strip()]
    return parts

async def print_stream(resp):
    if inspect.isasyncgen(resp):
        async for chunk in resp:
            if chunk:
                print(chunk, end="", flush=True)
        print()
    else:
        print(resp)

# --- lifecycle -------------------------------------------------------------

async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Uses OpenAI bindings the project ships with:
        llm_model_func=gpt_4o_mini_complete,   # for answers
        embedding_func=openai_embed,           # text-embedding-3-small (1536-d)
    )
    # REQUIRED in current versions (storage first, pipeline second):
    await rag.initialize_storages()
    await initialize_pipeline_status()  # <-- the missing piece in your run
    return rag

async def safe_finalize(rag: LightRAG | None):
    if rag is not None:
        try:
            await rag.finalize_storages()
        except Exception as e:
            print("Finalize warning:", e)

# --- main workflow ---------------------------------------------------------

async def main():
    # optional: clean storages to avoid “already exists / no new unique docs” confusion
    if CLEAN_FIRST and os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR, ignore_errors=True)
        os.makedirs(WORKING_DIR, exist_ok=True)

    if not os.path.exists(COMBINED_MD):
        raise FileNotFoundError(f"Cannot find {COMBINED_MD}; put your corpus here.")

    rag = None
    try:
        rag = await initialize_rag()

        items = read_items_from_md(COMBINED_MD)
        print(f"Found {len(items)} items from markdown. Inserting...")

        # Insert sequentially (simple & robust). You can parallelize if you want.
        for i, txt in enumerate(items, 1):
            await rag.ainsert(txt)
            if i % 100 == 0:
                print(f"  inserted {i} / {len(items)}")

        print("✅ Ingestion complete.\n")

        # Quick compatibility check: run the same question under different modes
        question = "Cheapest Lenovo LOQ with RTX 2050 and 16GB RAM?"
        for mode in ["naive", "local", "global", "hybrid"]:
            print(f"\n--- [{mode.upper()}] ---")
            resp = await rag.aquery(question, param=QueryParam(mode=mode, stream=True))
            await print_stream(resp)

    finally:
        await safe_finalize(rag)

if __name__ == "__main__":
    asyncio.run(main())
