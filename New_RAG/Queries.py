import json
import os
from minicheck.minicheck import MiniCheck
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import chromadb
import pickle

try:
    import nltk
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


DB_DIR = "./chroma_db"
COLLECTION_NAME = "articles"
INDEX_ARTICLES_DIR = "./index_articles.json"
TEXT_ARTICLES_DIR = "./text_articles.jsonl"

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
client = chromadb.PersistentClient(
    path=DB_DIR,
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

fact_check_model = MiniCheck(model_name='flan-t5-large', enable_prefix_caching=False, cache_dir='./ckpts')


def fact_check_batch(claims: list, evidences: list):
    """Run fact-checking on a batch of claims using corresponding evidences."""
    return fact_check_model.score(evidences, claims)

def fact_check_with_query_batch(claims: list) -> list:
    claim_embeddings = model.encode(claims)
    results = collection.query(
        query_embeddings=claim_embeddings,
        n_results=10,
    )
    all_evidences = []
    for ids in results["ids"]:
        # ids is a list of string ids for each claim
        chunk_ids = [int(i) for i in ids]
        evidences = get_chunks_by_id(chunk_ids, jsonl_path=TEXT_ARTICLES_DIR, index_path=INDEX_ARTICLES_DIR)
        # Join all evidence chunks for this claim
        all_evidences.append(" ".join(evidences))
    return fact_check_batch(claims, all_evidences)

def get_chunks_by_id(doc_ids, jsonl_path=TEXT_ARTICLES_DIR, index_path=INDEX_ARTICLES_DIR):
    with open(index_path, "r", encoding="utf-8") as f_index:
        offsets = json.load(f_index)

    if isinstance(doc_ids, int):
        doc_ids = [doc_ids]
    offsets_to_retrieve = [offsets[i] for i in doc_ids if i < len(offsets)]
    if not offsets_to_retrieve:
        return []

    elts = []
    with open(jsonl_path, "r", encoding="utf-8") as f_jsonl:
        for offset in offsets_to_retrieve:
            f_jsonl.seek(offset)
            line = f_jsonl.readline()
            elts.append(json.loads(line))
    return elts

if __name__ == "__main__":
    # Example usage
    # Load claims and labels from train.jsonl
    claims = []
    truth_labels = []
    label_map = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}
    with open("shared_task_dev.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            claims.append(item["claim"])
            truth_labels.append(label_map[item["label"]])

    labels, scores, _, _ = fact_check_with_query_batch(claims)

    with open("fact_check_results.pkl", "wb") as f:
        pickle.dump({
            "truth_labels": truth_labels,
            "labels": labels,
            "scores": scores
        }, f)

    accuracy = sum([l == t for l, t in zip(labels, truth_labels)]) / len(labels)
    print(f"Accuracy: {accuracy:.4f}")
