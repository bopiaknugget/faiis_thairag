import os
import requests
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global set to store hashes of indexed documents
indexed_hashes = set()

def get_document_hash(content):
    """Generate a hash for the document content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def index_chunk(chunk, index_url):
    """Attempts to index a document chunk with retries."""
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(index_url, json={"text": chunk})
            if response.status_code == 201:
                logger.info("Chunk indexed successfully.")
                return True
            else:
                logger.error(f"Failed to index chunk. Status code: {response.status_code}, attempt {attempt + 1}")
                sleep(2 ** attempt)  # Exponential backoff
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}, attempt {attempt + 1}")
            sleep(2 ** attempt)
    return False

def read_and_index_docs(docs_dir='./docs', index_url='http://localhost:5000/index'):
    indexed_files, non_indexed_files = 0, 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for filename in os.listdir(docs_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(docs_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # Check for duplicates
                    doc_hash = get_document_hash(content)
                    if doc_hash in indexed_hashes:
                        logger.info(f"Skipping {filename}: duplicate document.")
                        continue
                    indexed_hashes.add(doc_hash)  # Mark this document as indexed
                    
                    # Split content into chunks by paragraphs for coherence
                    chunks = content.split("\n\n")  # Simple example split by paragraphs
                    for chunk in chunks:
                        if len(chunk) > 1000:
                            sub_chunks = [chunk[i:i+1000] for i in range(0, len(chunk), 1000)]
                            for sub_chunk in sub_chunks:
                                futures.append(executor.submit(index_chunk, sub_chunk, index_url))
                        else:
                            futures.append(executor.submit(index_chunk, chunk, index_url))
                    
                    indexed_files += 1
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    non_indexed_files += 1
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            if not future.result():
                non_indexed_files += 1
    
    logger.info(f"Indexing complete. Indexed files: {indexed_files}, Non-indexed files: {non_indexed_files}")

if __name__ == "__main__":
    read_and_index_docs()
