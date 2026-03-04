import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL   = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "manuals"
LLM_MODEL        = "mistral"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

reranker = CrossEncoder(RERANKER_MODEL)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

all_chunks  = []
bm25_index  = None

indexed_sources: set = set()

def index_pdf(pdf_path: str) -> dict:
    """
    Load a PDF, chunk it, and store in:
      - ChromaDB      (dense vector index, persisted on disk)
      - BM25Okapi     (sparse keyword index, in memory)
      - all_chunks    (accumulated list of all Document objects)
      - indexed_sources (set of already-indexed filenames)

    Supports multiple PDFs:
      - Checks if this specific PDF filename was already indexed
      - If yes: skips it and returns status
      - If no:  indexes it and rebuilds BM25 from ALL chunks

    Returns:
        {
            "status"  : "indexed" | "already_exists",
            "chunks"  : int (number of new chunks added),
            "source"  : str (filename)
        }
    """
    global all_chunks, bm25_index, indexed_sources

    import os
    source_name = os.path.basename(pdf_path)                        

    if source_name in indexed_sources:
        return {"status": "already_exists", "chunks": 0, "source": source_name}

    loader     = PyPDFLoader(pdf_path)
    docs       = loader.load()
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    new_chunks = splitter.split_documents(docs)

    vectorstore.add_documents(new_chunks)

    all_chunks.extend(new_chunks)

    tokenized_corpus = [chunk.page_content.split() for chunk in all_chunks]
    bm25_index       = BM25Okapi(tokenized_corpus)

    indexed_sources.add(source_name)

    return {"status": "indexed", "chunks": len(new_chunks), "source": source_name}

def get_indexed_sources() -> list:
    """Return the list of all indexed PDF filenames."""
    return sorted(indexed_sources)

def rewrite_query(query: str, chat_history: list) -> str:
    """
    If there is chat history, ask the LLM to rewrite the query
    into a fully self-contained question using the conversation context.
    This fixes pronoun references like 'le', 'it', 'this device', etc.
    If no history exists, return the original query unchanged.
    """
    if not chat_history:
        return query                                 

    history_text = ""
    for turn in chat_history[-3:]:
        history_text += f"User: {turn['question']}\n"
        history_text += f"Assistant: {turn['answer']}\n\n"

    rewrite_prompt = f"""Given this conversation history:
{history_text}
Rewrite the following question into a fully self-contained question \
without any pronouns or references that depend on the history.
Return ONLY the rewritten question, nothing else.

Question: {query}
Rewritten question:"""

    response = llm.invoke(rewrite_prompt)
    rewritten = response.content.strip()
    return rewritten if rewritten else query

def build_prompt(context: str, query: str, chat_history: list) -> str:
    """
    Assemble the full prompt:
    - Role instruction
    - Last 3 conversation turns (if any)
    - Retrieved context
    - Current question
    """
    history_text = ""
    for turn in chat_history[-3:]:
        history_text += f"Question précédente: {turn['question']}\n"
        history_text += f"Réponse précédente: {turn['answer']}\n\n"

    history_block = (
        f"HISTORIQUE DE CONVERSATION:\n{history_text}"
        if history_text else ""
    )

    return f"""Tu es un assistant technique biomédical.
Réponds UNIQUEMENT avec les informations du CONTEXTE.
Si l'information n'existe pas dans le contexte, dis: "Information non trouvée dans les documents."
Réponse courte, claire, actionnable.

CONTEXTE:
{context}

{history_block}
QUESTION:
{query}
"""

def ask(query: str, chat_history: list) -> dict:
    """
    Full RAG pipeline for one question.

    Args:
        query        : The user's question
        chat_history : List of past {"question": ..., "answer": ...} dicts

    Returns:
        {
            "answer"  : str,
            "sources" : list of {"source": str, "page": str}
        }
    """

    search_query = rewrite_query(query, chat_history)

    dense_results = vectorstore.similarity_search(search_query, k=8)

    sparse_results = []
    if bm25_index is not None:
        tokenized_query = search_query.split()                                            
        bm25_scores     = bm25_index.get_scores(tokenized_query)                         

        top_bm25_idx   = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:8]
        sparse_results = [all_chunks[i] for i in top_bm25_idx]

    seen_texts = set()
    candidates = []
    for doc in dense_results + sparse_results:
        if doc.page_content not in seen_texts:
            seen_texts.add(doc.page_content)
            candidates.append(doc)

    pairs  = [[search_query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored   = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:3]]

    context = "\n\n".join(doc.page_content for doc in top_docs)

    prompt   = build_prompt(context, query, chat_history)
    response = llm.invoke(prompt)
    answer   = response.content

    seen    = set()
    sources = []
    for doc in top_docs:
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        key  = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": src, "page": page})

    contexts = [doc.page_content for doc in top_docs]

    return {"answer": answer, "sources": sources, "contexts": contexts}
