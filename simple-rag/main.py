from utils import load_and_chunk_all
from retriever import Retriever
from generator import generate

# Accept multiple files
files = ['data/Semi-Annual Investor Call Transcript – March 19 2025.pdf','data/2024ltr.pdf']
docs = load_and_chunk_all(files)

retriever = Retriever(docs)

query = input("Ask a question: ")
relevant_docs = retriever.search(query)
answer = generate(query, relevant_docs)

print("\nAnswer:\n", answer)