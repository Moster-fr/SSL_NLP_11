import os
import json
import chromadb
import time
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, logging
from huggingface_hub import login
from tqdm import tqdm

logging.set_verbosity_error()

DB_DIR = "./chroma_db"
ARTICLES_DIR = "./Articles/wiki-pages"
COLLECTION_NAME = "articles"
INDEX_ARTICLES_DIR = "./index_articles.json"
TEXT_ARTICLES_DIR = "./text_articles.jsonl"

CHUNK_SIZE = 5000  # Maximum size of chunks
CHUNK_OVERLAP = 200  # Overlap in characters between chunks

client = chromadb.PersistentClient(
    path=DB_DIR,
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def load_articles(max_articles=500):
    articles_counter = 0
    id = 0
    offsets = None
    existing_ids = collection.count()
    print(existing_ids)
    next_index = existing_ids

    for filename in os.listdir(ARTICLES_DIR):
        if filename.endswith(".jsonl"):
            path = os.path.join(ARTICLES_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                sentences, ids, giga_splits = [], [], []
                for line in f:
                    data = json.loads(line.strip())
                    content = data["text"]
                    #splits = text_splitter.split_text(content)
                    splits = [content]
                    for i in range(len(splits)):
                        if id >= existing_ids:
                            sentences.append(splits[i])
                            ids.append(str(id))
                            giga_splits.append(splits[i])
                        id += 1
                embeddings = model.encode(giga_splits, convert_to_tensor=False)
                if sentences:
                    next_index, offsets = append_chunks_to_jsonl_with_index(sentences, jsonl_path=TEXT_ARTICLES_DIR, index_path=INDEX_ARTICLES_DIR, start_index=next_index, offsets=offsets)
                    i = 0
                    while i < len(sentences):
                        collection.upsert(
                        embeddings=embeddings[i:i+5200],
                        ids=ids[i:i+5200]
                        )
                        i += 5200

            articles_counter += 1
            print(f"Inserted article {articles_counter} with {len(sentences)} chunks")
            if articles_counter >= max_articles:
                break
    print(f"Total articles loaded: {articles_counter}")

def append_chunks_to_jsonl_with_index(chunks, jsonl_path="documents.jsonl", index_path="index.json", start_index=0, offsets= None):
    # Load or initialize offset index

    if offsets is None:
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f_index:
                offsets = json.load(f_index)
        else:
            offsets = []

    with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        for i, chunk in enumerate(chunks, start=start_index):
            offset = f_jsonl.tell()
            #entry = {"id": str(i), "text": chunk}
            entry = chunk
            f_jsonl.write(json.dumps(entry) + "\n")
            if i == len(offsets): offsets.append(offset)
            else: offsets[i] = offset

    # Save updated index
    with open(index_path, "w", encoding="utf-8") as f_index:
        json.dump(offsets, f_index)
    
    return start_index + len(chunks), offsets

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
separators = [".", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=separators,
    keep_separator=False, 
)

if __name__ == "__main__":
    load_articles()
    existing_ids = collection.count()
    print(existing_ids)