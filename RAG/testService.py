import requests

# For local tests
url = "http://127.0.0.1:5000/process-txt"

# Path to the PDF file
file_path = "C:/Users/Stefan/Workspace/Thesis/AgenticRAG/RAG/DataScrape/Abstracts/Deep Drug Recommender_Ref_1.txt"


with open(file_path, 'rb') as file:
    files = {'file': file}
    
    response = requests.post(url, files=files)
    print(response)

    if response.status_code == 200:
        print("File uploaded successfully!")
        print("Extracted Text:", response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.json())