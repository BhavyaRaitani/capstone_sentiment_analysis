import re
import string
import multiprocessing as mp
import os
import sys

# --- Configuration ---
# Set your file paths here
INPUT_FILE = r"D:\WISE\capstone\Dataset\train.ft.txt"
OUTPUT_FILE = r"D:\WISE\capstone\Dataset\train_minimal_clean_with_puctuation.ft.txt" # New output file

# Use all available cores for maximum speed
NUM_CORES = os.cpu_count() or 4
print(f"Using {NUM_CORES} CPU cores for parallel processing.")

# --- Global Punctuation List for Tokenization ---

# List of punctuation to treat as separate tokens (e.g., "word." becomes "word .")
# We include common sentence enders, strong sentiment indicators, and separators.
PUNCTUATION_TO_TOKENIZE = ['.', '!', '?', ',', ';', ':', '"', '(', ')', '[', ']', '{', '}', '/', '&', '*']

# Compile the regex pattern once for speed
# This pattern finds any character in the list and wraps it with spaces.
# It excludes the FastText label prefix "__" and the single quote/apostrophe.
TOKENIZATION_PATTERN = "|".join(re.escape(p) for p in PUNCTUATION_TO_TOKENIZE)
# Replace: any tokenization char | any repeated whitespace (for cleanup later)
# The full pattern is a bit complex for speed; simpler is:
# TOKEN_RE = re.compile(r'([' + re.escape(PUNCTUATION_TO_TOKENIZE) + r'])') 

# Define the pattern to add a space before and after the defined punctuation
TOKEN_RE = re.compile(r'(' + r'|'.join(re.escape(p) for p in PUNCTUATION_TO_TOKENIZE) + r')')

# Define pattern to compress multiple spaces into a single space
WHITESPACE_RE = re.compile(r'\s+')

def initialize_worker_minimal():
    """Worker initializer (empty, as resources are pre-compiled and simple)."""
    pass

# --- Core Cleaning Function (MAXIMUM SPEED, TOKENIZING SENTIMENT CUES) ---

def clean_and_format_line_tokenized(line: str) -> str | None:
    """
    Applies basic normalization: lowercasing, URL removal, and
    PUNCTUATION TOKENIZATION.
    """
    line = line.strip()
    if not line:
        return None

    # 1. Separate Label and Text
    try:
        label_end_index = line.find(' ')
        if label_end_index == -1: return None
            
        label = line[:label_end_index] 
        text = line[label_end_index+1:] 
        
    except Exception:
        return None
        
    # 2. String Cleaning Steps
    
    # Lowercasing
    text = text.lower() 
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Tokenize punctuation: Add space around critical punctuation marks
    text = TOKEN_RE.sub(r' \1 ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Clean up excess whitespace created by the tokenization step
    text = WHITESPACE_RE.sub(' ', text).strip()

    # 3. Tokenize and Basic Filtering
    tokens = text.split()
    
    # Filter for tokens that are not empty strings after cleanup
    cleaned_tokens = [
        word for word in tokens 
        if word 
    ]
        
    # 4. Recombine and return in FastText format
    if cleaned_tokens:
        return f"{label} {' '.join(cleaned_tokens)}\n"
        
    return None

def process_data_in_parallel(input_path: str, output_path: str, num_cores: int):
    """
    Manages the parallel reading, processing, and writing of the large file.
    """
    print(f"Starting minimal preprocessing with PUNCTUATION TOKENIZATION on {num_cores} cores.")
    print(f"Input file: {input_path}")

    CHUNKSIZE = 100000 
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            
            # Use initializer to load the translation table once per process
            with mp.Pool(processes=num_cores, initializer=initialize_worker_minimal) as pool:
                
                # imap_unordered streams data efficiently
                results_iterator = pool.imap_unordered(clean_and_format_line_tokenized, infile, chunksize=CHUNKSIZE)
                
                total_lines = 0
                cleaned_lines_count = 0
                
                # --- Write results iteratively to avoid memory overflow ---
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for cleaned_line in results_iterator:
                        total_lines += 1
                        if cleaned_line is not None:
                            outfile.write(cleaned_line)
                            cleaned_lines_count += 1
                            
                        # Print progress every 1 million lines
                        if total_lines % 1_000_000 == 0:
                            print(f"Processed {total_lines:,} lines so far...")

        print("-" * 50)
        print("Preprocessing complete! 🎉")
        print(f"Total lines read (approx): {total_lines:,}")
        print(f"Cleaned lines saved: {cleaned_lines_count:,}")
        print(f"Output saved to: {output_path}")
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"\n--- ERROR: Input file not found at '{input_path}' ---", file=sys.stderr)
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---", file=sys.stderr)


# --- Execution ---
if __name__ == "__main__":
    process_data_in_parallel(INPUT_FILE, OUTPUT_FILE, NUM_CORES)