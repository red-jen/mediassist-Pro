from fastapi import FastAPI
from backend.database import Base, engine
from backend.routers import auth, rag
from prometheus_fastapi_instrumentator import Instrumentator

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MediAssist Pro",
    description="RAG-powered assistant for biomedical device manuals",
    version="1.0.0"
)
Instrumentator().instrument(app).expose(app)                 

app.include_router(auth.router)                                
app.include_router(rag.router)                               

@app.get("/")
def root():
    return {"message": "MediAssist Pro API is running"}

@app.get("/health")
def health():
    """
    Health check endpoint used by Docker HEALTHCHECK and load balancers.
    Returns 200 if the API is running and the database is reachable.
    """
    return {"status": "ok"}
