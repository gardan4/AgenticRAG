from chromadb_functions import load_database_from_dir

def query_database(query_text, db_folder, k=5, min_trust_score=0):
    """
    Queries the ChromaDB database to retrieve the k most similar chunks.
    Also considers metadata such as filename and trust_score.
    """
    db = load_database_from_dir(db_folder)
    if db is None:
        return []
    
    results = db.similarity_search(query_text, k=k)
    
    if not results:
        return []
    
    extracted_data = []
    
    for doc in results:
        metadata = doc.metadata
        trust_score = metadata.get("trust_score", 50)  # Default trust_score if missing
        filename = metadata.get("filename", "Unknown")  # Default filename if missing
        
        # Only return documents above a certain trust score
        if trust_score >= min_trust_score:
            extracted_data.append({
                "text": doc.page_content,
                "filename": filename,
                "trust_score": trust_score
            })

    return_string = []
    
    for doc in extracted_data:
        return_string.append(f"Filename: {doc['filename']}, Trust Score: {doc['trust_score']}, Text: {doc['text']}")
    
    return '\n'.join(return_string)


    


query = "deliverables"
db_path = "./RAG/Database/Output"
max_docs = 5
min_trust = 0

result = query_database(query, db_path, max_docs, min_trust)

print(result)