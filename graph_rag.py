import networkx as nx
import re
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ======================================================
# 1. SETUP: LOAD MODELS & GRAPH
# ======================================================
print("⏳ Loading Embedding Model (This may take a minute)...")
# We use a small, fast model for creating vectors
encoder = SentenceTransformer('all-MiniLM-L6-v2') 

# Re-using your extraction logic (Simplified for brevity)
def extract_attributes(title, category):
    specs = {}
    title_lower = title.lower()
    
    # Brand Extraction
    brands = ['Lenovo', 'HP', 'ASUS', 'Gigabyte', 'MSI', 'Dell', 'Acer', 'Apple', 'Samsung', 'Xiaomi', 'Realme', 'OnePlus', 'Infinix', 'Tecno', 'Vivo', 'Oppo', 'Honor', 'Motorola', 'Nokia', 'Walton']
    for brand in brands:
        if brand.lower() in title_lower:
            specs['Brand'] = brand
            break
            
    # Spec Extraction
    if 'gb' in title_lower:
        ram = re.search(r'(\d+)\s?gb', title, re.IGNORECASE)
        if ram: specs['RAM'] = ram.group(1) + "GB"
    
    return specs

def build_graph_and_vectors(file_path):
    G = nx.Graph()
    products = []
    product_texts = [] # For Vector DB
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = content.split('---')
    print(f"🔄 Parsing {len(entries)} items...")

    for entry in entries:
        lines = [l.strip() for l in entry.strip().split('\n') if l.strip()]
        if not lines: continue
        
        # Parse basic info
        title = lines[0].replace('## ', '').strip()
        category = "General"
        price = "0"
        
        for line in lines:
            if '**Category:**' in line: category = line.split('**Category:**')[1].strip()
            if '**Price:**' in line: price = line.split('**Price:**')[1].strip()
        
        # Clean price
        try:
            price_num = float(re.sub(r'[^\d.]', '', price))
        except:
            price_num = 0

        # Add to Graph
        attrs = extract_attributes(title, category)
        prod_id = title[:40] # Short ID
        
        G.add_node(prod_id, type='Product', full_title=title, price=price_num, category=category)
        G.add_edge(prod_id, category, relation='IS_A')
        if 'Brand' in attrs: G.add_edge(prod_id, attrs['Brand'], relation='MADE_BY')
        if 'RAM' in attrs: G.add_edge(prod_id, attrs['RAM'], relation='HAS_RAM')

        # Add to Vector List
        # We embed the "Title + Category + Price" to capture semantic meaning
        text_representation = f"{title} | Category: {category} | Price: {price}"
        products.append({'id': prod_id, 'text': text_representation, 'price': price_num})
        product_texts.append(text_representation)

    return G, products, product_texts

# ======================================================
# 2. BUILDING THE PIPELINE
# ======================================================
filename = 'combined_corpus.md'
if os.path.exists(filename):
    # A. Build Graph
    KG, product_data, corpus_texts = build_graph_and_vectors(filename)
    
    # B. Build Vector Index (FAISS)
    print("ta-da! Creating Vector Embeddings (Math for AI)...")
    embeddings = encoder.encode(corpus_texts)
    
    # Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ Indexed {index.ntotal} products in Vector DB.")

    # ======================================================
    # 3. THE HYBRID SEARCH ENGINE (Graph + Vector)
    # ======================================================
    def graph_rag_search(user_query, max_budget=None):
        print(f"\n🔎 USER ASKS: '{user_query}' (Budget: {max_budget})")
        
        # STEP 1: Vector Retrieval (Semantic Search)
        # Find top 5 items that "sound like" the query
        query_vector = encoder.encode([user_query])
        D, I = index.search(query_vector, 5) # Distances, Indices
        
        vector_candidates = []
        print("\n[Step 1] Vector Search Results (Semantic Match):")
        for i in I[0]:
            item = product_data[i]
            print(f" - Found: {item['id']}...")
            vector_candidates.append(item['id'])
            
        # STEP 2: Graph Filtering (Logical Reasoning)
        # Now we use the Graph to validate these candidates against constraints
        print("\n[Step 2] Graph Reasoning Validation:")
        
        final_results = []
        
        for prod_id in vector_candidates:
            # Check Graph Constraints
            node_attrs = KG.nodes[prod_id]
            price = node_attrs.get('price', 0)
            
            # Logic: If user set a budget, strictly enforce it
            if max_budget and price > max_budget:
                print(f" ❌ Rejected '{prod_id}' (Price {price} > {max_budget})")
                continue
            
            # Graph Traversal: Check neighbors for context
            # Example: If query mentions 'Samsung', check if node is connected to 'Samsung'
            if "Samsung" in user_query and not KG.has_edge(prod_id, "Samsung"):
                 # Try 2 hops: Product -> Brand Node -> "Samsung"
                 # (Simplified for demo: just accepting if vector found it, but you see the potential)
                 pass

            print(f" ✅ Accepted '{prod_id}' (Price: {price})")
            final_results.append(node_attrs)

        return final_results

    # ======================================================
    # 4. TEST THE SYSTEM
    # ======================================================
    
    # Query 1: Vague semantic query
    results = graph_rag_search("I need a fast gaming phone", max_budget=45000)
    
    # Query 2: Specific brand query
    results = graph_rag_search("Best Apple Laptop", max_budget=200000)

else:
    print("Error: combined_corpus.md not found.")