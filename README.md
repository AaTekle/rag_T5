# Retrieval-Augmented Generation (RAG) with Hugging Face, Sentence Transformers, and FAISS

Lightweight Retrieval-Augmented Generation (RAG) model using:

* Hugging Face Datasets
* Sentence Transformers
* FAISS Vector Search
* FLAN-T5 Language Model

**Goal:** post-training algorithm that allows language models to answer questions using external knowledge retrieved at runtime rather than relying on information memorized during intial training phases.

RAG allows models (llms) to answer questions based within a specific knowledge domain (improving a models ability to recall specialized information)

---

# RAG

Retrieval-Augmented Generation (RAG) combines Information Retrieval & Large Language Models (LLMs).


Traditional LLM:

```
Question
    ↓
   LLM
    ↓
 Answer
```

RAG:

```
Question
    ↓
Embedding Model
    ↓
Vector Search
    ↓
Relevant Documents
    ↓
   LLM
    ↓
  Answer
```

The answer is based around retrieved evidence (external data/training).

---

# Why RAG?

**Language models have limits:**

* initial training knowledge becomes outdated
* facts could be made up (hallucinated)
* ability to recall domain-specific information may be missing
* traditional fine-tuning is expensive

RAG fixes these issues by providing external information during inference.

**RAG Benefits:**

* accurate responses
* reduced hallucinations
* no retraining required
* integration with up to date external data sources

---

# Project Dataset, Model, and VDB (vector database)

## Dataset

The project uses the rag-mini-wikipedia dataset via huggingface, dataset is the external knowledge base: (https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)

---

## Embedding Model

**Embedding Model:** 
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**Goal of Embedding Model:**

Convert text into dense numerical vectors (so computer can compute textual information).

Example:

```text
"What is relativity?"
```

becomes:

```text
[0.021, -0.114, 0.827, ...]
```

Each document in the dataset is converted into a vector.

The user's question is also converted into a vector.

---

## Vector Database (FAISS)

FAISS stores document vectors. (library for efficient similarity search and clustering of dense vectors) (https://faiss.ai/index.html)

When a query arrives:

```text
Question
    ↓
Embedding
    ↓
 Vector
    ↓
FAISS Search
```

FAISS finds the most similar vectors.

Instead of searching raw text, it searches vector space.

This makes retrieval extremely fast.

---

## Language Model

**Model:** google/flan-t5-base (https://huggingface.co/google/flan-t5-base)

The model receives:

* User question
* Retrieved context

Example:

```text
Context:
Albert Einstein developed the theory of relativity.

Question:
Who developed the theory of relativity?
```

The model generates:

```text
Albert Einstein developed the theory of relativity.
```

---

# Math Behind RAG

## Embeddings

A sentence is transformed into a high-dimensional vector.

Example:

```
Sentence A:
"What is photosynthesis?"

Vector A:
[0.23, 0.91, -0.12, ...]
```

```
Sentence B:
"Photosynthesis converts sunlight into energy."

Vector B:
[0.26, 0.87, -0.09, ...]
```

These vectors take up positions within memory as vector space.

Semantically similar sentences are located near each other.

---

## Cosine Similarity

The project uses normalized embeddings and inner product search.

Similarity is computed using:

```math
\cos(\theta)=
\frac{A \cdot B}
{|A||B|}
```


Where:

* $A$ = Query vector (the user's search or input)
* $B$ = Document vector (stored content being compared)
* $A·B$ = Dot product of vectors $A$ and $B$. Measures how aligned the vectors are.
* $cos$ = Cosine function. Produces a similarity score between -1 and 1.
* $θ$ (theta) = The angle between two vectors. Smaller angles mean higher similarity.
*  $|A||B|$ = Product of the magnitudes of $A$ and $B$. Used to normalize the score.



**Cosine Similarity Interpretations:**

| Score | Meaning             |
| ----- | ------------------- |
| 1.0   | Identical direction |
| 0.8   | Very similar        |
| 0.5   | Related             |
| 0.0   | Unrelated           |
| -1.0  | Opposite            |

Because embeddings are normalized:

```math
|A| = |B| = 1
```

the cosine similarity becomes:

```math
A \cdot B
```

which allows efficient FAISS retrieval.

---

# Retrieval Workflow

### Workflow 1: Query → Retrieval → Answer

* user submits a question.
* the question is converted into an embedding vector.
* FAISS searches for the most similar document vectors.
* top matching documents are retrieved.
* retrieved documents are combined into a context.
* the context and question are sent to the LLM.
* the LLM generates the final answer.

---

### Workflow 2: Embedding-Based Search

* convert the query into a vector representation.
* compare the query vector against stored document vectors.
* rank documents by similarity score.
* return the top *K* most relevant documents.

---

### Workflow 3: Context Construction

* retrieve the highest-scoring documents.
* merge document content into a single context block.
* keep the most relevant information for the query.
* pass the context to the language model.

---

### Workflow 4: Answer Generation

* Provide the LLM with:

  * User question
  * Retrieved context
* the model uses the retrieved evidence to generate an answer.
* the final response is returned to the user.


---

## Advantages of RAG

* Uses external knowledge
* Reduces hallucinations
* Fast retrieval
* No model retraining
* Simple architecture

---
##  Run Locally & Screenshots (Ouputs of RAG)

- **Run locally:** clone & run `rag.py` within local environment
- **Outputs:** look within outputs folder


