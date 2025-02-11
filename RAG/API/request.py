#This file is used to send requests to a docker container with all of the preprocessing logic that will extract information from a variety of input files

import requests

# URL of the API
# For local tests
url = "http://127.0.0.1:5000/process-pdf"
file_path = "D:/Downloads/test.pdf"

# Open the file in binary mode
with open(file_path, 'rb') as file:
    # Prepare the file to be uploaded using 'files' argument
    files = {'pdf': file}
    
    # Send the POST request with the file
    response = requests.post(url, files=files)
    print(response)

    # Print the response from the server
    if response.status_code == 200:
        print("File uploaded successfully!")
        print("Extracted Text:", response.json())

    else:
        print(f"Error: {response.status_code}")
        print(response.json())
