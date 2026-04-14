from langchain.document_loaders import PyPDFLoader
import os

data_path = "data"
all_docs = []

files = os.listdir(data_path)

for file in files:
    if file.endswith(".pdf"):   # only PDFs
        file_path = os.path.join(data_path, file)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Add metadata (VERY IMPORTANT 🔥)
        for doc in docs:
            doc.metadata["source"] = file

        all_docs.extend(docs)

print(f"Total documents loaded: {len(all_docs)}")