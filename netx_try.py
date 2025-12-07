import networkx as nx
import matplotlib.pyplot as plt
import re
import pandas as pd

# ==========================================
# 1. THE DATA INGESTION (Simulating your file)
# ==========================================
raw_data = """
## Lenovo LOQ AI Powered Gaming (9) (83DV00K0LK) 13th Gen Core i7 16GB RAM 512GB SSD RTX 3050 Gaming Laptop
**Category:** gaming laptops
**Price:** 156,030

## HP VICTUS 15-FB1013 AMD Ryzen 5 7535HS 8GB RAM 512GB SSD 4GB RTX2050 15.6" FHD IPS Gaming Laptop
**Category:** gaming laptops
**Price:** 88,500

## ASUS TUF Gaming A15 FA506NFR Ryzen 7 7435HS 8GB RAM, 512GB SSD 15.6 Inch FHD Display Gaming Laptop
**Category:** gaming laptops
**Price:** 95,500

## Gigabyte B450M K AMD AM4 Micro ATX Motherboard
**Category:** motherboard
**Price:** 8,500

## AMD Ryzen 5 5600 Processor
**Category:** processor
**Price:** 12,000
"""

# ==========================================
# 2. THE INTELLIGENCE (Entity Extraction)
# ==========================================
def extract_attributes(title):
    """
    This function acts like a mini-brain. It looks at the Title
    and finds specific hardware specs using patterns (Regex).
    """
    specs = {}
    
    # Extract RAM (e.g., 16GB, 8GB)
    ram_search = re.search(r'(\d+)\s?GB\s?RAM', title, re.IGNORECASE)
    if ram_search:
        specs['RAM'] = ram_search.group(1) + "GB"

    # Extract Storage (e.g., 512GB SSD, 1TB SSD)
    storage_search = re.search(r'(\d+)\s?(GB|TB)\s?(SSD|HDD)', title, re.IGNORECASE)
    if storage_search:
        specs['Storage'] = f"{storage_search.group(1)}{storage_search.group(2)} {storage_search.group(3)}"

    # Extract Brand (Simple lookup list)
    brands = ['Lenovo', 'HP', 'ASUS', 'Gigabyte', 'AMD', 'Intel', 'Samsung', 'Walton']
    for brand in brands:
        if brand.lower() in title.lower():
            specs['Brand'] = brand
            break
            
    # Extract CPU Series (Simple logic)
    if 'Ryzen 5' in title: specs['CPU'] = 'Ryzen 5'
    elif 'Ryzen 7' in title: specs['CPU'] = 'Ryzen 7'
    elif 'Core i7' in title: specs['CPU'] = 'Core i7'
    elif 'Core i5' in title: specs['CPU'] = 'Core i5'

    return specs

# ==========================================
# 3. PARSING THE RAW DATA
# ==========================================
products = []
entries = raw_data.strip().split('\n\n')

for entry in entries:
    lines = entry.strip().split('\n')
    title = lines[0].replace('## ', '').strip()
    
    # Basic info
    category = "Unknown"
    price = "0"
    
    for line in lines[1:]:
        if "**Category:**" in line:
            category = line.split('**Category:**')[1].strip()
        if "**Price:**" in line:
            price = line.split('**Price:**')[1].strip()

    # Get the "Hidden" specs from the title
    extracted_specs = extract_attributes(title)
    
    products.append({
        'Title': title,
        'Category': category,
        'Price': price,
        **extracted_specs # Merge the extracted specs
    })

# ==========================================
# 4. BUILDING THE KNOWLEDGE GRAPH
# ==========================================
G = nx.Graph() # Initialize an empty graph

print("Constructing the Graph connections...\n")

for p in products:
    # 1. Create the Main Product Node
    # We use the Title as the unique ID for the node
    product_node = p['Title'][:20] + "..." # Shorten name for display
    G.add_node(product_node, type='Product', price=p['Price'])
    
    # 2. Connect to Category (The Classification)
    # If the category node doesn't exist, NetworkX creates it automatically
    G.add_edge(product_node, p['Category'], relation='IS_A')
    
    # 3. Connect to Extracted Attributes (The "Knowledge")
    if 'Brand' in p:
        G.add_edge(product_node, p['Brand'], relation='MADE_BY')
    
    if 'RAM' in p:
        # Connect product to a "16GB" node
        G.add_edge(product_node, p['RAM'], relation='HAS_MEMORY')
        
    if 'CPU' in p:
        # Connect product to a "Ryzen 5" node
        G.add_edge(product_node, p['CPU'], relation='POWERED_BY')

# ==========================================
# 5. VISUALIZATION (Seeing the Wall)
# ==========================================
plt.figure(figsize=(12, 10))

# Create layout (Physics simulation to space out nodes)
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Define node colors based on type (Logic to make it pretty)
node_colors = []
for node in G.nodes():
    if node in ['gaming laptops', 'motherboard', 'processor']:
        node_colors.append('lightgreen') # Categories
    elif 'GB' in node or 'Ryzen' in node or 'Core' in node:
        node_colors.append('lightblue') # Specs
    elif node in ['Lenovo', 'HP', 'ASUS', 'Gigabyte', 'AMD']:
        node_colors.append('orange') # Brands
    else:
        node_colors.append('#ffcccc') # Products

# Draw the graph
nx.draw(G, pos, 
        with_labels=True, 
        node_color=node_colors, 
        node_size=2000, 
        font_size=9, 
        font_weight='bold', 
        edge_color='gray')

# Draw edge labels (The "Strings" on the wall)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title("E-Commerce Knowledge Graph: Relationships & Specs", fontsize=15)
plt.show()

# ==========================================
# 6. GRAPH REASONING (The "Why")
# ==========================================
print("\n--- GRAPH REASONING ENGINE ---")

# Question: "What products use a Ryzen 5 Processor?"
target_node = "Ryzen 5"
if target_node in G:
    print(f"\n[Query]: Finding everything connected to '{target_node}'...")
    neighbors = G.neighbors(target_node)
    for n in neighbors:
        print(f" -> Found: {n}")
else:
    print(f"Node {target_node} not found.")

# Question: "Find me HP laptops."
brand_node = "HP"
if brand_node in G:
    print(f"\n[Query]: Finding products MADE_BY '{brand_node}'...")
    neighbors = G.neighbors(brand_node)
    for n in neighbors:
        print(f" -> Found: {n}")