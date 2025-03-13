from chromadb_functions import load_database_from_dir

def query_database(query_text, db_folder, k=5):
    """
    Queries the ChromaDB database to retrieve the k most similar chunks.
    """
    db = load_database_from_dir(db_folder)
    if db is None:
        return []
    results = db.similarity_search(query_text, k=k)
    if not results:
        return []
    extracted_texts = [doc.page_content for doc in results]
    return extracted_texts

query = ""
db_path = "./RAG/Database/Output"
max_docs = 5

result = query_database(query, db_path, max_docs)

print(result)