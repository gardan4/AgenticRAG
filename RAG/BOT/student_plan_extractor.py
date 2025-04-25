import requests
from pydantic_bot import PydanticAIBot
import json

def json_to_string(json_data):
    """
    Extracts and combines all text elements from a JSON list of dictionaries.

    Args:
        json_data (list): List of dictionaries containing 'text' keys.

    Returns:
        str: Combined text content as a single string.
    """
    return "\n".join(item["text"] for item in json_data if "text" in item)

# For local tests
url = "http://127.0.0.1:5000/process-pdf-yolox"

# Path to the PDF file
file_path = "RAG/Bot/InputFiles/group1 project plan proposal.pdf"

bot = PydanticAIBot(model="gpt-4o-mini", temperature=0.7)

with open(file_path, 'rb') as file:
    files = {'file': file}
    
    response = requests.post(url, files=files)
    print(response)

    if response.status_code == 200:
        print("File processed successfully!")
        plan_as_text = json_to_string(response.json()).replace("- ","")
        extracted_plan = bot.ask(question="Plan: "+plan_as_text, context="You are given the raw text extracted from a student plan report for their group research project. Analyze their plan and turn it into a step by step plan that can easily be checked by a supervisor. Make sure each step of the plan starts with its number and then a dot. Don't give any extra explanations or return any text other than the plan itself. For example: 1. Perform a deep research on topic relative software. 2. Download needed software. 3. Use software for... Make the planned steps as specific as possible or use as many steps as needed, given the fact that you have a lengthy explanation of the project plan.")
        #Write extracted plan to txt file
        with open("extracted_plan.txt", 'w') as output_file:
            output_file.write(extracted_plan)

    else:
        print(f"Error: {response.status_code}")
        print(response.json())