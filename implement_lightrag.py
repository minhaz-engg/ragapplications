import os
import glob
import logging
from typing import List

# LightRAG Imports
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

# Markdown Processing Imports
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. Configuration & Logging
# ------------------------------------------------------------------------------
# Set your OpenAI API key in the environment variables: export OPENAI_API_KEY="sk-..."
WORKING_DIR = "./lightrag_storage"
INPUT_DIR = "./combined_corpus.md"  # Point this to your folder of.md files

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 2. Advanced Markdown Processing (Structure-Aware)
# ------------------------------------------------------------------------------
def read_and_chunk_markdown(directory_path: str) -> List[str]:
    """
    Reads Markdown files and splits them by headers to preserve semantic context.
    This solves the 'Context Fragmentation' issue before LightRAG even sees the data.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    processed_chunks = []
    
    # robust globbing for.md files
    files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
    logging.info(f"Found {len(files)} Markdown files in {directory_path}")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split the file based on headers
            docs = markdown_splitter.split_text(content)
            
            for doc in docs:
                # We reconstruct the chunk with its metadata (headers) to ensure
                # LightRAG's LLM sees the full context (e.g., "Header 1 > Header 2: Content")
                header_context = " > ".join([v for k, v in doc.metadata.items()])
                full_text = f"Context: {header_context}\nContent: {doc.page_content}"
                processed_chunks.append(full_text)
                
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            
    logging.info(f"Generated {len(processed_chunks)} structure-aware chunks.")
    return processed_chunks

# 3. LightRAG Initialization
# ------------------------------------------------------------------------------
# We use GPT-4o-mini for efficiency. For production extraction, GPT-4o is better.
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Used for generation & extraction
    # llm_model_func=gpt_4o_complete      # Uncomment for higher quality extraction (more $$)
)

# 4. Ingestion Pipeline
# ------------------------------------------------------------------------------
def ingest_data():
    chunks = read_and_chunk_markdown(INPUT_DIR)
    
    if chunks:
        logging.info("Starting LightRAG ingestion... This may take time as the Graph is built.")
        #.insert() handles the embedding, entity extraction, and graph construction
        rag.insert(chunks)
        logging.info("Ingestion Complete. Graph and Vector indexes updated.")
    else:
        logging.warning("No data found to ingest.")

# 5. Querying Interface
# ------------------------------------------------------------------------------
def run_queries():
    # Example 1: Low-Level (Specific Fact)
    # This uses the vector index to find specific entities.
    q1 = "What specific parameters are required for the authentication module?"
    print(f"\n[Low-Level Query]: {q1}")
    print(rag.query(q1, param=QueryParam(mode="local")))

    # Example 2: High-Level (Thematic Summary)
    # This uses the high-level keywords and global graph structure.
    q2 = "Summarize the security best practices outlined across the documentation."
    print(f"\n[High-Level Query]: {q2}")
    print(rag.query(q2, param=QueryParam(mode="global")))

    # Example 3: Mix (Hybrid) - The Recommended Default
    # Combines both for maximum accuracy.
    q3 = "How do the rate limits interact with the caching strategy?"
    print(f"\n[Mix Query]: {q3}")
    print(rag.query(q3, param=QueryParam(mode="mix")))

# Main Execution
if __name__ == "__main__":
    # Uncomment to ingest data (run once or when data changes)
    # ingest_data() 
    
    # Run queries
    run_queries()