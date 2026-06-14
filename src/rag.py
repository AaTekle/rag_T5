import faiss
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# external Hugging Face dataset (for rag)
dataset = load_dataset(
    "rag-datasets/rag-mini-wikipedia",
    "text-corpus",
    split="passages" #which portion of the dataset to load
)

docs = [row["passage"] for row in dataset]


# embedding documents
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu" #hardware used for inference
)

embeddings = embedder.encode(
    docs, #texts to embed
    convert_to_numpy=True, #return NumPy array instead of Torch tensor
    normalize_embeddings=True #L2-normalizes vectors to unit length, useful for cosine similarity search.
).astype("float32") #datatype for FAISS index (float32 is standard for FAISS)


# FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1]) #embedding dimension (number of features per vector, e.g. 384)
index.add(embeddings) #document vectors to store in FAISS


# google/flan-t5-base model pipeline
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=200 #max tokens generated in the answer
)


# chat loop
while True:
    query = input("\nAsk: ")

    if query.lower() in ["exit", "quit"]:
        break

    query_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(query_vec, 3)

    sources = [docs[i] for i in ids[0]]

    context = "\n\n".join(sources) #combine retrieved documents into a single context string for the LLM prompt

    prompt = f"""
Answer using only this context.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm(prompt)[0]["generated_text"]

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for score, i in zip(scores[0], ids[0]):
        print(f"- {score:.3f}: {docs[i][:300]}...")
        
'''
embeddings.shape[1] → embedding dimension (e.g. 384)
index.add(embeddings)
    embeddings → document vectors stored in FAISS

pipeline("text2text-generation", model="google/flan-t5-base", device=..., max_new_tokens=200)
    "text2text-generation" → task type
    model → model to load
    device → hardware (0 = GPU, -1 = CPU)
    max_new_tokens → maximum generated output length

input("\nAsk: ")
    prompt shown before user input

query.lower()
    converts query to lowercase

embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    [query] → text(s) to embed
    convert_to_numpy → return NumPy array
    normalize_embeddings → normalize vectors to unit length

.astype("float32")
    converts embeddings to float32

index.search(query_vec, 3)
    query_vec → query embedding
    3 → number of nearest documents to retrieve
    returns scores and ids

[docs[i] for i in ids[0]]
    docs → all documents
    i → retrieved document index
    ids[0] → indices from first query

"\n\n".join(sources)
    sources → retrieved document texts
    "\n\n" → separator between documents

llm(prompt)
    prompt → text sent to the model

llm(prompt)[0]["generated_text"]
    [0] → first result
    "generated_text" → generated answer

print("\nAnswer:")
    text displayed before answer

zip(scores[0], ids[0])
    scores[0] → similarity scores
    ids[0] → document indices
    pairs scores with document IDs

f"- {score:.3f}: {docs[i][:300]}..."
    score → similarity score
    .3f → 3 decimal places
    docs[i] → retrieved document
    [:300] → first 300 characters
    ... -> truncation indicator
'''
