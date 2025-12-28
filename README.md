# RAG Project – Georgian Civil Code Q&A

This project is a **Retrieval-Augmented Generation (RAG)** system built on the **Georgian Civil Code**.

It allows a user to ask legal questions and receive answers **strictly based on the text of the Civil Code**, along with **clear source references** (articles / sections used).

The system **does not generate legal opinions** or new interpretations — its purpose is to **retrieve, analyze, and present relevant legal text** in a structured and understandable way.

---

## How the project works

1. The Civil Code document is cleaned and analyzed.
2. The text is split into meaningful chunks (articles / sections).
3. A retrieval index is built to enable semantic search.
4. When a user asks a question:

   * The most relevant legal sections are retrieved.
   * A large language model synthesizes an answer **only from those sections**.
5. The final response includes:

   * A clear answer in natural language
   * Explicit source references to the Civil Code text

If the answer cannot be found in the document, the system states that explicitly.

---

## Technologies used

* **Python**
* **BM25-based retrieval** (document search)
* **Large Language Model via Hugging Face Router**
* **OpenAI-compatible client**
* **Structured chunking and metadata indexing**

---

## Required Libraries

* **rank-bm25**
* **openai**
* **python-dotenv**

---

## Installing dependencies

Make sure you have **Python 3.10+** installed.

Install the required libraries using `pip`:

```bash
pip install rank-bm25 openai
```

---

## Running the project

1. Set required environment variables:

   ```bash
   HF_TOKEN=your_huggingface_token
   HF_MODEL=your_selected_model
   ```

2. Run the application:

   ```bash
   python app.py
   ```

---

