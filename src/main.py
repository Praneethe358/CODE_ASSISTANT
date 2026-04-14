"""Beginner-friendly Retrieval-Augmented Generation pipeline with LangChain."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = ROOT_DIR / "data" / "sample.pdf"
DEFAULT_INDEX_DIR = ROOT_DIR / "data" / "faiss_index"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4


def read_int_env(name: str, default: int) -> int:
    """Read an integer from the environment and fall back to a default value."""

    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load the PDF into LangChain Document objects, one per page."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Update RAG_PDF_PATH in .env or place a PDF there."
        )

    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split large pages into smaller overlapping chunks for better retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_embeddings() -> OpenAIEmbeddings:
    """Create the embedding model used to convert text chunks into vectors."""

    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))


def build_vector_store(chunks: list[Document], embeddings: OpenAIEmbeddings, index_dir: Path) -> FAISS:
    """Create a FAISS index, or load one from disk if it already exists."""

    index_file = index_dir / "index.faiss"
    pickle_file = index_dir / "index.pkl"

    if index_file.exists() and pickle_file.exists():
        # This flag is required by LangChain when loading local FAISS indexes.
        return FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(index_dir))
    return vector_store


def format_documents(documents: list[Document]) -> str:
    """Turn retrieved chunks into a readable context block for the LLM."""

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        page_number = document.metadata.get("page")
        page_label = page_number + 1 if isinstance(page_number, int) else "?"
        formatted_chunks.append(
            f"Chunk {index} | page {page_label}\n{document.page_content.strip()}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def retrieve_chunks(vector_store: FAISS, query: str, top_k: int) -> list[Document]:
    """Search FAISS for the most relevant chunks for a user question."""

    return vector_store.similarity_search(query, k=top_k)


def print_retrieved_chunks(documents: list[Document]) -> None:
    """Print the retrieved chunks so you can inspect what the retriever found."""

    print("\nRetrieved chunks:")
    print("=" * 80)
    for index, document in enumerate(documents, start=1):
        page_number = document.metadata.get("page")
        page_label = page_number + 1 if isinstance(page_number, int) else "?"
        print(f"Chunk {index} (page {page_label})")
        print(document.page_content.strip())
        print("-" * 80)


def answer_question(question: str, context: str, chat_model: str) -> str:
    """Send the retrieved context to the LLM and ask for a grounded answer."""

    llm = ChatOpenAI(model=chat_model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful RAG assistant. Answer only from the provided context. "
                "If the context does not contain the answer, say you do not know.",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


def main() -> None:
    """Run the full RAG pipeline from loading the PDF to printing the answer."""

    load_dotenv()

    pdf_path = Path(os.getenv("RAG_PDF_PATH", str(DEFAULT_PDF_PATH)))
    index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    query = os.getenv("RAG_QUERY", "What is this document about?")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    chunk_size = read_int_env("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = read_int_env("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    top_k = read_int_env("RAG_TOP_K", DEFAULT_TOP_K)

    print("Loading PDF...")
    documents = load_pdf(pdf_path)
    print(f"Loaded {len(documents)} PDF pages.")

    print("Splitting text into chunks...")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and FAISS index...")
    embeddings = build_embeddings()
    vector_store = build_vector_store(chunks, embeddings, index_dir)

    print(f"Retrieving the top {top_k} chunks for the query: {query!r}")
    retrieved_documents = retrieve_chunks(vector_store, query, top_k=top_k)
    print_retrieved_chunks(retrieved_documents)

    context = format_documents(retrieved_documents)
    print("\nGenerating answer with the LLM...")
    answer = answer_question(query, context, chat_model=chat_model)

    print("\nFinal answer:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
