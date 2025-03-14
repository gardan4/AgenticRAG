from unstructured.documents.elements import Element, ElementMetadata, CoordinateSystem
from CustomElementMetadata import CustomElementMetadata
from CustomElement import CustomElement
from chromadb_functions import create_database,load_database_from_dir,add_documents_to_database
from dotenv import load_dotenv
import requests
import os
import json

def contains_sqlite3_file(folder_path):
    """
    Function for checking if a database was already created

    Args:
    folder_path (String): Path to the folder to check

    Returns:
    True if a sqlite3 file is found
    False if no sqlite3 file is found
    """
    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file ends with .sqlite3 extension
        if file_name.endswith('.sqlite3'):
            return True
    return False

def get_all_files(folder_path):
    """
    Function to get all files from a specified folder

    Args:
    folder_path (String): Path to the folder from which to retrieve files

    Returns:
    List of file names found in the folder
    """
    # Create an empty list to store file names
    file_list = []
    
    # List all items in the directory
    for item in os.listdir(folder_path):
        # Get the full path of the item
        item_path = os.path.join(folder_path, item)
        # Check if it's a file (not a directory)
        if os.path.isfile(item_path):
            # Add the file to the list
            file_list.append(item_path)
    
    return file_list

def jsonToElements(json_data,filename="Unknown",trust_score=50):
    # Check if json_data is a Response object and handle it
    if hasattr(json_data, 'json'):  # Check if it's a Response object
        json_data = json_data.json()  # Use the .json() method to directly parse it
    elif isinstance(json_data, bytes):
        json_data = json_data.decode('utf-8')  # Decode bytes to string
        json_data = json.loads(json_data)  # Convert JSON string to Python objects
    elif isinstance(json_data, str):
        json_data = json.loads(json_data)  # Parse the JSON string into Python objects

    elements = []
    for item in json_data:
        category = item['category']
        text = item['text']

        coordinate_system = CoordinateSystem(width=0, height=0)

        metadata = CustomElementMetadata(
            filename=filename,
            trust_score=trust_score,
            page_number=0,
            languages=['en']
        )

        # Initialize Element object
        element = CustomElement(
            element_id=None,
            coordinates=((0, 0), (0, 0)),
            coordinate_system=coordinate_system,
            metadata=metadata,
            detection_origin=category
        )

        # Assign category and text
        element.category = category
        element.text = text
        elements.append(element)

    return elements

load_dotenv()

#Change this field before running the file, depending on the trustworthiness of the documents located in the Input folder
#Trust score must always be between 0 and 100
batch_trust_score = 100

#Directory containing all input files
files_dir = "./RAG/Database/Input"

#Folder where the database should be created
db_folder = "./RAG/Database/Output"

#The api url
url = "http://127.0.0.1:5000/process-pdf-fast"
#url = "http://127.0.0.1:5000/process-pdf-yolox"
#url = "http://127.0.0.1:5000/process-docx"
#url = "http://127.0.0.1:5000/process-html"
#url = "http://127.0.0.1:5000/process-txt"

input_files_list = get_all_files(files_dir)
skip_file = ""

db_exists = contains_sqlite3_file(db_folder)
if(not db_exists):
    if os.path.isfile(input_files_list[0]) and (not skip_file==input_files_list[0]):
        skip_file=input_files_list[0]
        # Open the file in binary mode
        with open(input_files_list[0], 'rb') as file:
            # Prepare the file to be uploaded using 'files' argument
            files = {'file': file}
            
            # Send the POST request with the file
            response = requests.post(url, files=files)

            # Print the response from the server
            if response.status_code == 200:
                print("File uploaded successfully!")
                initial_elements = jsonToElements(json_data=response,filename=os.path.basename(input_files_list[0]),trust_score=batch_trust_score)
                db_exists = create_database(initial_elements,db_folder)
                skip_first_file=input_files_list[0]
            else:
                print(f"Error: {response.status_code}")
                print(response.json())
    

if db_exists:
    db = load_database_from_dir(db_folder)
    if not db is None:
        
        #Add all files in the folder (excluding the first if it was used to create the database)
        for file_path in input_files_list:
            if os.path.isfile(file_path) and (not skip_file==file_path):

                # Open the file in binary mode
                with open(file_path, 'rb') as file:
                    # Prepare the file to be uploaded using 'files' argument
                    files = {'file': file}
                    
                    # Send the POST request with the file
                    response = requests.post(url, files=files)

                    # Print the response from the server
                    if response.status_code == 200:
                        print("File uploaded successfully!")
                        new_pdf_elements = jsonToElements(json_data=response,filename=os.path.basename(file_path),trust_score=batch_trust_score)
                        add_documents_to_database(new_pdf_elements,db)
                    else:
                        print(f"Error: {response.status_code}")
                        print(response.json())


            else:
                print(f"Directory: {file_path}")