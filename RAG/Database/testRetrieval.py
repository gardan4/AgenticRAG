from chromadb_functions import load_database_from_dir
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

def query_database(query_text, db_folder, k=5):
    """
    Queries the ChromaDB database to retrieve the k most similar chunks.

    Args:
        query_text (str): The query/question to search for.
        db_folder (str): The folder where the database is stored.
        k (int): The number of most similar results to return.

    Returns:
        list: A list of the top k most relevant text chunks.
    """
    # Load the database
    db = load_database_from_dir(db_folder)
    
    if db is None:
        print("Error: Database could not be loaded.")
        return []

    # Query the database using similarity_search
    results = db.similarity_search(query_text, k=k)

    if not results:
        print("No relevant results found.")
        return []

    # Extract the relevant text chunks
    extracted_texts = [doc.page_content for doc in results]
    
    return extracted_texts

if __name__ == "__main__":
    db_folder = "./RAG/Database/Output"
    query_text = input("Enter your question: ")
    k = int(input("Enter the number of results to retrieve: "))

    retrieved_texts = query_database(query_text, db_folder, k)

    print("\nTop", k, "most relevant chunks:")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"{i}. {text}\n")
