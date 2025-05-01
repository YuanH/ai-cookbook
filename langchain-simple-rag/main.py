from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loaders = [
    PyPDFLoader('data/2024ltr.pdf'),
    PyPDFLoader('data/Semi-Annual Investor Call Transcript – March 19 2025.pdf')
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# Step 2: Chunk the text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 3: Embed & store in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Step 4: Setup OpenAI LLM
llm = ChatOpenAI(model="gpt-4", 
                 temperature=0,
                 openai_api_key="api-key")

# Step 5: RAG Chain with Retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Step 6: Ask a question
while True:
    query = input("Ask a question (or 'exit'): ")
    if query.lower() == 'exit':
        break
    answer = qa.run(query)
    print("\nAnswer:\n", answer, "\n")