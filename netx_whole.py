import networkx as nx
import matplotlib.pyplot as plt
import re
import os

# ======================================================
# 1. THE INTELLIGENCE CORE (Entity Extraction)
# ======================================================
def extract_attributes(title, category):
    """
    Extracts entities based on the category of the product.
    This makes the AI 'Context Aware'.
    """
    specs = {}
    title_lower = title.lower()

    # --- BRAND EXTRACTION (Works for all) ---
    # We define a known list of brands from your dataset
    known_brands = [
        'Lenovo', 'HP', 'ASUS', 'Gigabyte', 'MSI', 'Dell', 'Acer', 'Apple', 
        'Samsung', 'Xiaomi', 'Realme', 'OnePlus', 'Infinix', 'Tecno', 'Vivo', 
        'Oppo', 'Honor', 'Motorola', 'Nokia', 'Walton', 'Chuwi', 'ZTE', 
        'Sony', 'Haier', 'Singer', 'TCL', 'Dahua', 'Hikvision', 'TP-Link', 
        'Tenda', 'Netis', 'Mercusys', 'ZKTeco', 'DJI', 'GoPro'
    ]
    for brand in known_brands:
        if brand.lower() in title_lower:
            specs['Brand'] = brand
            break

    # --- LOGIC FOR ELECTRONICS (Laptops, Phones, Tablets) ---
    if category in ['gaming laptops', 'laptops', 'macbooks', 'smartphones', 'tablets']:
        # Extract RAM (e.g., 16GB, 8GB)
        ram_match = re.search(r'(\d+)\s?GB', title, re.IGNORECASE)
        if ram_match:
            specs['RAM'] = ram_match.group(1) + "GB"

        # Extract Storage (SSD/HDD)
        storage_match = re.search(r'(\d+)\s?(GB|TB)\s?(SSD|HDD|NVMe|ROM)', title, re.IGNORECASE)
        if storage_match:
            specs['Storage'] = f"{storage_match.group(1)}{storage_match.group(2)}"

        # Extract Processor Family
        if 'ryzen 3' in title_lower: specs['CPU'] = 'Ryzen 3'
        elif 'ryzen 5' in title_lower: specs['CPU'] = 'Ryzen 5'
        elif 'ryzen 7' in title_lower: specs['CPU'] = 'Ryzen 7'
        elif 'ryzen 9' in title_lower: specs['CPU'] = 'Ryzen 9'
        elif 'core i3' in title_lower or 'i3-' in title_lower: specs['CPU'] = 'Core i3'
        elif 'core i5' in title_lower or 'i5-' in title_lower: specs['CPU'] = 'Core i5'
        elif 'core i7' in title_lower or 'i7-' in title_lower: specs['CPU'] = 'Core i7'
        elif 'core i9' in title_lower or 'i9-' in title_lower: specs['CPU'] = 'Core i9'
        elif 'm1' in title_lower: specs['CPU'] = 'Apple M1'
        elif 'm2' in title_lower: specs['CPU'] = 'Apple M2'
        elif 'm3' in title_lower: specs['CPU'] = 'Apple M3'
        elif 'snapdragon' in title_lower: specs['CPU'] = 'Snapdragon'

    # --- LOGIC FOR FASHION (Sneakers, Polos) ---
    elif category in ['mens sneakers', 'mens polo shirts']:
        if 'cotton' in title_lower: specs['Material'] = 'Cotton'
        if 'leather' in title_lower: specs['Material'] = 'Leather'
        if 'mesh' in title_lower: specs['Material'] = 'Mesh'
        if 'canvas' in title_lower: specs['Material'] = 'Canvas'
        
        # Simple Color Extraction
        colors = ['black', 'white', 'blue', 'red', 'green', 'grey', 'yellow', 'navy']
        for color in colors:
            if color in title_lower:
                specs['Color'] = color.capitalize()
                break

    return specs

# ======================================================
# 2. FILE PARSER (The Ingestion Engine)
# ======================================================
def parse_corpus(file_path):
    products = []
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please create it first.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the horizontal rule separator
    entries = content.split('---')

    for entry in entries:
        lines = [l.strip() for l in entry.strip().split('\n') if l.strip()]
        if not lines: continue

        # Initialize product dict
        product = {'Title': None, 'Category': 'Uncategorized', 'Price': 'N/A', 'Source': 'Unknown'}

        # Extract Title (Lines starting with ##)
        for line in lines:
            if line.startswith('##'):
                product['Title'] = line.replace('##', '').strip()
                break
        
        if not product['Title']: continue # Skip if no title found

        # Extract Metadata (Lines starting with **)
        for line in lines:
            if '**Category:**' in line:
                product['Category'] = line.split('**Category:**')[1].strip().lower()
            elif '**Price:**' in line:
                product['Price'] = line.split('**Price:**')[1].strip()
            elif '**Source:**' in line:
                product['Source'] = line.split('**Source:**')[1].strip()

        # Extract Smart Attributes
        attributes = extract_attributes(product['Title'], product['Category'])
        product.update(attributes)
        
        products.append(product)
        
    return products

# ======================================================
# 3. GRAPH CONSTRUCTION
# ======================================================
def build_knowledge_graph(products):
    G = nx.Graph()
    
    print(f"Building Graph from {len(products)} products...")
    
    for p in products:
        # Node 1: The Product Itself
        # We limit title length to 30 chars for readability in graph
        prod_id = p['Title'][:30] + "..." 
        G.add_node(prod_id, type='Product', full_title=p['Title'], price=p['Price'])
        
        # Link to Category
        category_node = p['Category'].title() # e.g., "Gaming Laptops"
        G.add_edge(prod_id, category_node, relation='IS_A')
        
        # Link to Brand
        if 'Brand' in p:
            G.add_edge(prod_id, p['Brand'], relation='MADE_BY')
            
        # Link to Specs (The "Knowledge" part)
        if 'CPU' in p:
            G.add_edge(prod_id, p['CPU'], relation='POWERED_BY')
        if 'RAM' in p:
            G.add_edge(prod_id, p['RAM'], relation='HAS_MEMORY')
        if 'Storage' in p:
            G.add_edge(prod_id, p['Storage'], relation='HAS_STORAGE')
        if 'Material' in p:
            G.add_edge(prod_id, p['Material'], relation='MADE_OF')
        if 'Color' in p:
            G.add_edge(prod_id, p['Color'], relation='HAS_COLOR')

    return G

# ======================================================
# 4. EXECUTION MAIN
# ======================================================
# Step A: Load and Parse
filename = 'combined_corpus.md'
product_data = parse_corpus(filename)

if product_data:
    # Step B: Build Graph
    KG = build_knowledge_graph(product_data)
    
    print(f"\nGraph Stats:")
    print(f" - Nodes: {KG.number_of_nodes()}")
    print(f" - Edges: {KG.number_of_edges()}")

    # Step C: Visualization (Subset)
    # Visualizing ALL 1000+ nodes is messy in Python. We will visualize a random subgraph.
    subset_nodes = list(KG.nodes())[:100] # Take first 100 nodes for demo
    subgraph = KG.subgraph(subset_nodes)
    
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(subgraph, k=0.3)
    
    # Color coding nodes
    colors = []
    for node in subgraph.nodes():
        # Check if node exists in our product list's CPU/RAM/Brand list to color differently
        # Simplified: If it has neighbors > 2 it's likely a Hub (Brand/Spec)
        if subgraph.degree(node) > 3:
            colors.append('orange') # Hubs (Brand, CPU, Category)
        else:
            colors.append('lightblue') # Products

    nx.draw(subgraph, pos, 
            with_labels=True, 
            node_color=colors, 
            node_size=1500, 
            font_size=8, 
            edge_color='#D3D3D3')
    
    plt.title("Sample Knowledge Graph (First 100 Entities)", fontsize=20)
    plt.show()

    # Step D: Reasoning Example
    print("\n--- AI REASONING DEMO ---")
    
    # query 1: Find all i7 Laptops
    target_spec = "Core i7"
    if KG.has_node(target_spec):
        print(f"\n[Query]: Finding products connected to '{target_spec}':")
        # Get neighbors of 'Core i7' that are products
        connected_products = [n for n in KG.neighbors(target_spec) if KG.nodes[n].get('type') == 'Product']
        for prod in connected_products[:5]: # Show first 5
            print(f" - {prod}")
            
    # query 2: Cross-Sell based on Brand
    target_brand = "Samsung"
    if KG.has_node(target_brand):
        print(f"\n[Query]: Everything made by '{target_brand}':")
        items = list(KG.neighbors(target_brand))
        for item in items[:5]:
            print(f" - {item}")

    # Step E: Export for Professional Tools (Gephi)
    nx.write_gexf(KG, "ecommerce_graph.gexf")
    print("\n[Success] Full graph saved as 'ecommerce_graph.gexf'. You can open this in Gephi!")

else:
    print("No data found. Make sure 'combined_corpus.md' is in the folder.")
    


# ======================================================
# 5. THE REASONING ENGINE (Graph Search)
# ======================================================

def find_best_deal(graph, category_name, max_price=None, min_ram=None):
    print(f"\n--- 🧠 AI THINKING: Searching for '{category_name}' ---")
    
    # 1. Locate the Category Node
    # We capitalize it because our graph builder capitalized categories
    target_category = category_name.title() 
    
    if not graph.has_node(target_category):
        print(f"❌ Concept '{target_category}' not found in the Knowledge Graph.")
        return

    # 2. Get all neighbors (products linked to this category)
    # This is the "Traversal" step. Fast and efficient.
    candidates = list(graph.neighbors(target_category))
    print(f"-> Found {len(candidates)} items linked to concept '{target_category}'")
    
    results = []
    
    for product in candidates:
        # Get the internal attributes we saved in the node
        attrs = graph.nodes[product]
        
        # We only want nodes that are actual 'Products'
        if attrs.get('type') != 'Product':
            continue
            
        # Clean the price (Remove '৳', commas, and spaces to make it a number)
        try:
            raw_price = str(attrs.get('price', '0'))
            clean_price = float(re.sub(r'[^\d.]', '', raw_price))
        except:
            clean_price = 0
            
        # 3. Apply Filters (The Reasoning)
        
        # Filter A: Price Check
        if max_price and clean_price > max_price:
            continue # Too expensive, skip
            
        # Filter B: RAM Check (Graph-based filtering)
        # We check if the product is connected to a specific RAM node?
        # OR we check the extracted attributes. Let's check neighbors for RAM nodes.
        if min_ram:
            product_neighbors = list(graph.neighbors(product))
            # Check if any neighbor looks like "16GB" or "8GB"
            has_ram = False
            for n in product_neighbors:
                if 'GB' in n and n >= min_ram: # Simple string check logic
                    has_ram = True
                    break
            if not has_ram:
                continue

        results.append((product, clean_price))

    # 4. Sort by Price (Cheapest first)
    results.sort(key=lambda x: x[1])
    
    # 5. Output the Wisdom
    print(f"-> Filtered down to {len(results)} matches based on your constraints.")
    print("\nTOP RECOMMENDATIONS:")
    for i, (name, price) in enumerate(results[:5]): # Show top 5
        print(f"   {i+1}. {name} | Price: ৳{int(price)}")

# ======================================================
# 6. RUN EXPERIMENTS
# ======================================================

# Experiment A: Find a cheap Gaming Laptop
find_best_deal(KG, "Gaming Laptops", max_price=100000)

# Experiment B: Find a cheap Smartphone
find_best_deal(KG, "Smartphones", max_price=20000)

# Experiment C: Find Sneakers (Demonstrating Multi-modal capability)
find_best_deal(KG, "Mens Sneakers", max_price=3000)