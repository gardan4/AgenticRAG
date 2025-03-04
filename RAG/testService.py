import requests

# For local tests
url = "http://127.0.0.1:5000/process-pdf"

# Path to the PDF file
file_path = "C:/Users/Stefan/Workspace/RAG Assistant/Input/test.pdf"


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