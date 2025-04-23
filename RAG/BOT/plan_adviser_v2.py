import requests
from pydantic_bot import PydanticAIBot
from plan_maker import make_plan
from rules_fetcher import get_rules_context
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
        project_title = str(bot.ask(question="From the provided context find and return only the title of the project plan. Do not give any other responses or extra words, since your response will be fed into another system and it needs to be just the title.", context=plan_as_text[0:1200]))

        task = "Create a project plan for my research project, called: "+ project_title

        #Have the bot make its own plan summary
        bot_plan_and_context = make_plan(db_path = "./RAG/Database/Output", task = task,max_docs = 3,temp=0.7,return_bot_context=True)
        bot_plan = bot_plan_and_context["plan"]

        #Get some context surrounding the actual project
        project_context = bot_plan_and_context["context"]

        #Get rules surrounding research projects
        rules = get_rules_context(file_folder="RAG/Bot/InputFiles/Rules")

        bot_advice = bot.ask(question="Provide pointers, tips and overall advice for improving the group project plan proposal. In the context you have the extracted text from the plan proposal as well as in the end you have a summarized steps only plan. You also have rules the team must abide by when creating their project and some context information about the project itself. Look at the whole provided context and focus on the actual plan proposal by the students. Give useful feedback to the students on what they need to improve based on the rest of the context you have.", context=plan_as_text+"    Plan Super Short Summary: "+bot_plan+"    Rules for projects: "+rules+"    Project info: "+project_context)

        print("--------------------------------------")
        print(bot_advice)

    else:
        print(f"Error: {response.status_code}")
        print(response.json())