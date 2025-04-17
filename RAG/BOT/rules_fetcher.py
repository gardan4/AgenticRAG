import sys
import os
from openai_bot import OpenAI_GPT_Bot
from pydantic_bot import PydanticAIBot
from context_prep_bot import prep_context
from removeRepeatingContext import remove_repeating_context
import requests

def json_to_string(json_data):
    """
    Extracts and combines all text elements from a JSON list of dictionaries.

    Args:
        json_data (list): List of dictionaries containing 'text' keys.

    Returns:
        str: Combined text content as a single string.
    """
    return "\n".join(item["text"] for item in json_data if "text" in item)

def get_rules_context(file_folder):

    bot = PydanticAIBot(model="gpt-4o-mini", temperature=0.7)

    url = "http://127.0.0.1:5000/process-pdf-fast"

    concat_rules = ""

    # Loop through all files in the folder
    for filename in os.listdir(file_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(file_folder, filename)
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.post(url, files=files)
                file_as_text = json_to_string(response.json()).replace("- ","")
                concat_rules=concat_rules+"   |   Information from "+filename+": "+file_as_text

    return str(bot.ask(question="From the provided context, please create logical rules for making a project plan and for things that need to be done when doing research projects. Make it as insightful as possible for students. Only provide the information in a way that would be ready to print, without giving additional words. Your answer will be printed and needs to look like it is written by someone and not returned by an AI chatbot.", context="In this context there is information about file names and the contents of those files. The files are about rules, tips and regulations surrounding handling research projects. |   "+concat_rules))