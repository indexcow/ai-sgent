from fastapi import FastAPI, UploadFile, File
from rag_pipeline import ingest_file, query_rag

app = FastAPI()

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    result = ingest_file(file.filename, content)
    return {"message": result}

@app.get("/ask")
def ask(query: str):
    answer = query_rag(query)
    return {"answer": answer}
