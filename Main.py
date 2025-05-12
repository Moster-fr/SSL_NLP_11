import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, logging
from huggingface_hub import login
from tqdm import tqdm

logging.set_verbosity_error()


DB_DIR = "./chroma_db"
ARTICLES_DIR = "./articles/train"
CLAIMS_DIR = "./train.json"
COLLECTION_NAME = "articles"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
INDEX_ARTICLES_DIR = "./index_articles.json"
TEXT_ARTICLES_DIR = "./text_articles.jsonl"


CHUNK_SIZE = 5000  # Maximum size of chunks
CHUNK_OVERLAP = 200  # Overlap in characters between chunks

def load_articles(max_articles=500):
    articles_counter = 0
    id = 0
    next_index = 0
    existing_ids = collection.count()
    print(existing_ids)
    
    for filename in os.listdir(ARTICLES_DIR):
        if filename.endswith(".json"):
            path = os.path.join(ARTICLES_DIR, filename)
            #print(f"Loading {path}...")
            with open(path, "r", encoding="utf-8") as f:
                sentences, embeddings, ids, metadatas = [], [], [], []
                for line in tqdm(f, desc="Processing {filename}", unit="articles"):
                    data = json.loads(line.strip())
                    content = " ".join(data["url2text"])
                    splits = text_splitter.split_text(content)
                    for i in range(len(splits)):
                        if id >= existing_ids:
                            sentences.append(splits[i])
                            ids.append(str(id))
                        id += 1
                    embeddings.extend(model.encode(splits, convert_to_tensor=False))
                if sentences:
                    next_index = append_chunks_to_jsonl_with_index(sentences, jsonl_path=TEXT_ARTICLES_DIR, index_path=INDEX_ARTICLES_DIR, start_index=next_index)
                    i = 0
                    while i < len(sentences):
                        collection.upsert(
                        embeddings=embeddings[i:i+5400],
                        ids=ids[i:i+5400]
                        )
                        i += 5400

            articles_counter += 1
            print(f"Inserted article {articles_counter} with {len(sentences)} chunks.")
            if articles_counter >= max_articles:
                break
    print(f"Total articles loaded: {articles_counter}")

def append_chunks_to_jsonl_with_index(chunks, jsonl_path="documents.jsonl", index_path="index.json", start_index=0):
    # Load or initialize offset index
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
    
    return start_index + len(chunks)

def load_claims(file_path,limit=0):
    claims = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load the JSON file as a list of dicts
        for item in data:
            claims.append(item["claim"])  # Extract the "claim" field from each dict
            labels.append([item["label"]])  # Extract the "label" field from each dict

    if limit > 0:
        claims = claims[:limit]
        labels = labels[:limit]
    
    return check_duplicates_claims(claims, labels)

def check_duplicates_claims(claims, labels):
    unique_claims = []
    unique_labels = []
    nb_duplicates = 0
    for claim, label in zip(claims, labels):
        if claim not in unique_claims:
            unique_claims.append(claim)
            unique_labels.append(label)
        else:
            nb_duplicates += 1
            index = unique_claims.index(claim)
            if label not in unique_labels[index]:
                unique_labels[index].extend(label)
                unique_labels[index] = list(set(unique_labels[index]))  # Remove duplicates in labels

    print(f"Found {nb_duplicates} duplicates in the claims.")

    return list(unique_claims), unique_labels

def answer_facts(context, question):
    return f"""[INST] <<SYS>> 
You're an expert in fact-checking. Using only the given context, do your best to answer to the question in two sentences. Be assertive and concise in your answers, and do not uses verbs like "it appears that" "it seems that" "it looks like". If the answer is not precised in the document, answer with nothing. <</SYS>>

Context:
{context}

Question:
{question}

[/INST]"""

def split_fact(fact):
    return f"""[INST] <<SYS>>
You are a smart assistant. Break down the claim into five questions about when / what / how / why. You sould be able to get the sense of the claim with the questions. 5 questions maximum. Do not use pronons, only the real names / places / nouns / subjects. <</SYS>>
Claim: {fact}
[INST]"""

def answer_claim(claim, questions, answers):

    couples = "\n".join(answers)

    for i in range(len(questions)):
        couples += f"""{questions[i]} - {answers[i]} \n\n"""

    return f"""[INST] <<SYS>>
You're an an expert in fact-checking. You have to fact check the Claim Using the context. and classify the claim into :
- "Supported"
- "Refuted"
- "Conflicting Evidence/Cherrypicking"
- "Not enough evidence"
Please only answer with [LABEL] where [LABEL] is one of : Supported, Refuted, Not Enough Evidence, Conflicting Evidence/Cherrypicking <</SYS>>
Claim : {claim}

Context : {couples} [/INST]"""

def answer_context(claim, context):
    return f"""[INST] <<SYS>>
You're an expert in fact-checking. You have to fact check the Claim Using the context. Classify the claim into the most fitting label :
- "Supported" : The elements about the claim in context are in favor of the Claim
- "Refuted" : The elements about the claim in context are in opposition of the Claim
- "Conflicting Evidence/Cherrypicking" : The elements about the claim in context are both clearly in favor and in opposition of the Claim
- "Not enough evidence" : There are no elements in favor or in opposition of the claim in the context
Please only answer with [LABEL] where [LABEL] is one of : Supported, Refuted, Not Enough Evidence, Conflicting Evidence/Cherrypicking
<</SYS>>
Claim : {claim}

Context {context} [/INST]"""

claims_list, label_list = load_claims(CLAIMS_DIR,171)
print(f"Loaded {len(claims_list)} claims.")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # Small and fast, using CUDA

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

separators = [".", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=separators,
    keep_separator=False, 
)

client = chromadb.PersistentClient(
    path=DB_DIR,
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# Query function
def search(claim, k=5):

    prompt = split_fact(claim)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del prompt, inputs, outputs
    list_of_questions = output_text.split("\n")
    list_of_questions = [question.strip("1234567890. ") for question in list_of_questions if question.strip() != ""]
    # list_of_questions = [claim]
    #print(list_of_questions)
    query_embeddings = [model.encode(question).tolist() for question in list_of_questions]
    torch.cuda.empty_cache()

    # query_embeddings = [model.encode(claim).tolist()]

    # print(list_of_questions)

    results = collection.query(query_embeddings= query_embeddings, n_results=k)





    # for i in range(k):
    #     print(f"Result {i+1}:")
    #     print("URL:", results["metadatas"][0][i]["url"])
    #     print("Snippet:", results["documents"][0][i], "\n")

    list_of_answers = []

    for i in range(len(list_of_questions)):
        prompt = answer_facts('\n\n'.join(results['documents'][i]),list_of_questions[i])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = llm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        list_of_answers.append(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        del prompt, inputs, outputs
        torch.cuda.empty_cache()
        #print(f"Question {i+1} processed")
        # print("Question:", list_of_questions[i])
        # print("Answer:", list_of_answers[i])

    #prompt = answer_claim(claim, list_of_questions, list_of_answers)


    # context = []

    # for i in results['documents']:
    #     if type(i) == list:
    #         for j in i:
    #             context.append(j)
    #     else: context.append(i)

    prompt = answer_context(claim, '\n'.join(list_of_answers))
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(**inputs, max_new_tokens=15, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    

    # all_contexts = []
    # for i in results['documents']:
    #     all_contexts.extend(i)

    
    # prompt = answer_facts('\n\n'.join(all_contexts),claim)
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # print(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
    # outputs = llm_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Example run
if __name__ == "__main__":
    load_articles()
    # while True:
    #     q = input("Enter search query (or 'exit'): ")
    #     if q.lower() == "exit":
    #         break
    #     text = search(q)
    #     print(text)

    # text = search(claims_list[0])
    # positives = 0
    # outputs = []

    # for i in range(len(claims_list)):
    #     print(f"Claim{i+1}/{len(claims_list)}")
    #     print(f"Claim {i+1} : {claims_list[i]}")
    #     print("Label(s) : ", label_list[i])
    #     text = search(claims_list[i]).strip("[]}{-LABEL .")
    #     print(text)
    #     outputs.append(text)
    #     if text in label_list[i]:
    #         print("Machine is correct")
    #         positives += 1
    #     print("\n")

    # print(f"Accuracy : {positives} / {len(claims_list)}")


##Minicheck
