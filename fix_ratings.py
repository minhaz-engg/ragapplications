import re
import random

def generate_dummy_rating():
    """
    Generates a realistic dummy rating between 3.5 and 5.0.
    Precision is set to 1 decimal place (e.g., 4.2/5).
    """
    # We use a triangular distribution to make ratings around 4.5 more likely than 3.5 or 5.0
    rating = random.triangular(3.5, 5.0, 4.5)
    return f"{rating:.1f}/5"

def process_corpus(input_file, output_file):
    """
    Parses the corpus line by line. Identifies StarTech entries and 
    injects a dummy rating if missing.
    """
    print(f"[Info] Reading data from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[Error] The file '{input_file}' was not found. Please check the name.")
        return

    new_lines = []
    # State variable to track if we are currently inside a StarTech product block
    is_startech_block = False
    
    # Counter for statistics
    startech_count = 0

    for i, line in enumerate(lines):
        # 1. Check Source to update state
        if "**Source:** StarTech" in line:
            is_startech_block = True
            startech_count += 1
        elif "**Source:**" in line and "StarTech" not in line:
            is_startech_block = False

        new_lines.append(line)

        # 2. If we are in a StarTech block, look for the Price line
        # The schema implies Rating usually follows Price.
        if is_startech_block and line.strip().startswith("**Price:**"):
            # Check if the NEXT line is already a Rating (to avoid duplication if run twice)
            if i + 1 < len(lines) and "**Rating:**" in lines[i+1]:
                continue
            
            # 3. Inject the dummy rating
            dummy_rating = generate_dummy_rating()
            injection = f"**Rating:** {dummy_rating}\n"
            new_lines.append(injection)
            
            # Reset state for this block to prevent multiple insertions if file format is weird
            is_startech_block = False 

    # 4. Write the transformed data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"[Success] Processed {len(lines)} lines.")
    print(f"[Success] Injected ratings into approximately {startech_count} StarTech entries.")
    print(f"[Info] Output saved to: {output_file}")

# --- Execution ---
if __name__ == "__main__":
    input_filename = "dataset/combined_corpus.md"
    output_filename = "./refined_dataset/combined_corpus_fixed.md"
    
    process_corpus(input_filename, output_filename)