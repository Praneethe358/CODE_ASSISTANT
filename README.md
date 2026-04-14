# First RAG Pipeline with LangChain

This project is a beginner-friendly Retrieval-Augmented Generation (RAG) example.
It shows the full flow from a PDF to an answer:

1. Load a PDF
2. Split the text into chunks
3. Create embeddings for each chunk
4. Store the vectors in FAISS
5. Retrieve the most relevant chunks for a query
6. Ask an LLM to answer using only that context

## What RAG means

RAG combines two ideas:

- **Retrieval**: find the most relevant pieces of source text
- **Generation**: give those pieces to an LLM so it can answer with context

This is useful because the model does not need to memorize your PDF. It can look up the right parts first.

## Project Files

- `main.py` contains the full pipeline
- `requirements.txt` lists the Python packages
- `.env` stores environment variables and API settings
- `data/sample.pdf` is the example document

## Step-by-step walkthrough

### 1. Load the PDF

The script uses `PyPDFLoader` to read the PDF page by page.
Each page becomes a LangChain `Document` object.

Why this matters: page-level loading keeps the source metadata, which helps when you debug where an answer came from.

### 2. Split the text into chunks

Large pages are split with `RecursiveCharacterTextSplitter`.
The default chunk size is `800` characters with `120` characters of overlap.

Why this matters: embeddings work better on smaller, focused pieces of text than on huge pages.
Overlap helps keep sentences intact when a chunk boundary falls in the middle of an idea.

### 3. Create embeddings

`OpenAIEmbeddings` converts each chunk into a vector.
Vectors are numeric representations of meaning.

Why this matters: the retriever can compare the query vector with the chunk vectors to find similar text.

### 4. Store vectors in FAISS

FAISS is a fast local vector database.
The script saves the index in `data/faiss_index/` so you can reuse it later.

Why this matters: saving the index avoids rebuilding it every time you run the script.

### 5. Retrieve relevant chunks

The query is turned into a vector, and FAISS returns the most similar chunks.
The script prints those chunks so you can inspect what was retrieved.

Why this matters: if the answer is wrong, the first thing to check is whether retrieval found the right text.

### 6. Generate the answer

The retrieved chunks are put into a prompt and sent to `ChatOpenAI`.
The system prompt tells the model to answer only from the provided context.

Why this matters: grounded prompts reduce hallucinations.

## Installation

Create or activate your virtual environment, then install the packages:

```bash
pip install -r requirements.txt
```

If you do not already have a virtual environment, a common setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure environment variables

Open `.env` and replace `your-openai-api-key` with your real OpenAI API key.

You can also change these values:

- `OPENAI_CHAT_MODEL`: the chat model used for answering
- `OPENAI_EMBEDDING_MODEL`: the embedding model used for chunk vectors
- `RAG_PDF_PATH`: which PDF to load
- `RAG_QUERY`: the question to ask the document
- `RAG_CHUNK_SIZE`: chunk size in characters
- `RAG_CHUNK_OVERLAP`: how much text to overlap between chunks
- `RAG_TOP_K`: how many chunks to retrieve

## Run the project

```bash
python main.py
```

Expected flow:

- The PDF is loaded
- Chunks are created
- The FAISS index is built or loaded
- Retrieved chunks are printed
- The LLM generates a final answer

## Common errors and fixes

### 1. `OPENAI_API_KEY` is missing

Fix: set `OPENAI_API_KEY` in `.env`.

### 2. `ModuleNotFoundError`

Fix: install the dependencies from `requirements.txt` inside the active virtual environment.

### 3. `PDF not found`

Fix: check `RAG_PDF_PATH` and make sure the PDF really exists.

### 4. Retrieval returns irrelevant text

Fix: reduce chunk size, increase overlap, or test with a more specific query.

### 5. FAISS load errors

Fix: delete `data/faiss_index/` and rerun the script so the index is rebuilt.

### 6. OpenAI rate limit or authentication errors

Fix: confirm the API key is valid and your account has access to the chosen model.

## How to debug retrieval issues

If the answer is bad, do this in order:

1. Print the retrieved chunks and read them carefully.
2. Ask whether the answer is actually present in those chunks.
3. If the chunks are too broad, make `RAG_CHUNK_SIZE` smaller.
4. If chunks break up sentences too much, increase `RAG_CHUNK_OVERLAP`.
5. Try a more specific query.
6. If needed, switch FAISS search to MMR later for more diverse retrieval.

## How to improve answer quality

- Use a clearer PDF with clean text
- Lower the chunk size for dense documents
- Increase `RAG_TOP_K` to see more context
- Use `temperature=0` for more consistent answers
- Improve the system prompt so the model stays grounded
- Add metadata like page numbers when you expand this project

## Offline alternative

If you want a local embedding model instead of OpenAI, you can later swap `OpenAIEmbeddings` for a Hugging Face embedding model such as `sentence-transformers`.
That keeps the pipeline offline, but it adds one more dependency and setup step.

## Practice exercises

1. Change `RAG_CHUNK_SIZE` to `400` and compare retrieval results.
2. Ask three different questions and see how the printed chunks change.
3. Modify the prompt so the assistant answers in bullet points instead of paragraphs.

## Next steps

Once this works, the best upgrades are:

- add a Streamlit or FastAPI interface
- persist metadata more carefully
- add citations to the final answer
- support multiple PDFs instead of just one
