# for the chunkings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# thats for the embedings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder


# 1) Load PDF as Documents (each page is a Document)
loader = PyPDFLoader("datamedia.pdf")
docs = loader.load()

# 2) Chunk the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)

print(chunks[0])

# 3) Quick check
# print("Pages:", len(docs))
# print("Chunks:", len(chunkos))
# print(chunkos[0].page_content[:500])


# 4) Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 5) Store in Chroma (local) without duplicate indexing
vectorstore = Chroma(
    collection_name="manuals",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

if vectorstore._collection.count() == 0:
    vectorstore.add_documents(chunks)
    print("Indexed chunks in ./chroma_db")
else:
    print("Chroma DB already has data, skipping re-indexing")



# 6) Retriever config
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 7) Retrieval test
query = "Qu'est-ce qu'un lecteur de microplaques ELISA ?"
docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content[:500])
    print("Metadata:", doc.metadata)

# 6) Retrieve a larger pool
query = "Comment fonctionne un lecteur de microplaques ELISA ?"
candidates = vectorstore.similarity_search(query, k=8)

# 7) Re-rank with a cross-encoder (multilingual)
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

pairs = [[query, doc.page_content] for doc in candidates] # list of two items arrray query and doc retrieved
scores = reranker.predict(pairs)

scored = list(zip(candidates, scores))
scored.sort(key=lambda x: x[1], reverse=True)

top_docs = [doc for doc, score in scored[:3]]

for i, doc in enumerate(top_docs, start=1):
    print(f"\n--- Reranked {i} ---")
    print(doc.page_content[:500])
    print("Metadata:", doc.metadata)

# 8) Build context from reranked docs
context = "\n\n".join(doc.page_content for doc in top_docs)

# --- CHAT HISTORY SETUP (add this once before your chat loop) ---
chat_history = []  # list of {"question": ..., "answer": ...}

def build_prompt(context, query, chat_history):
    # Format past turns
    history_text = ""
    for turn in chat_history[-3:]:  # keep last 3 turns to avoid token overflow
        history_text += f"Question précédente: {turn['question']}\n"
        history_text += f"Réponse précédente: {turn['answer']}\n\n"

    return f"""
Tu es un assistant technique biomédical.
Réponds UNIQUEMENT avec les informations du CONTEXTE.
Si l'information n'existe pas dans le contexte, dis: "Information non trouvée dans les documents."
Réponse courte, claire, actionnable.

CONTEXTE:
{context}

{f"HISTORIQUE DE CONVERSATION:{chr(10)}{history_text}" if history_text else ""}

QUESTION:
{query}
"""

# 9) Chat loop
llm = ChatOllama(model="mistral", temperature=0)

while True:
    query = input("\nVotre question (ou 'quit'): ")
    if query.lower() == "quit":
        break

    # Retrieve + rerank for each new question
    candidates = vectorstore.similarity_search(query, k=8)
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored[:3]]
    context = "\n\n".join(doc.page_content for doc in top_docs)

    # Build prompt with history
    prompt = build_prompt(context, query, chat_history)

    # Generate answer
    response = llm.invoke(prompt)
    answer = response.content

    print("\n=== ANSWER ===")
    print(answer)

    # Save this turn to history
    chat_history.append({"question": query, "answer": answer})