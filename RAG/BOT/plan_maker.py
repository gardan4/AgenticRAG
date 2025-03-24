import sys
import os
from openai_bot import OpenAI_GPT_Bot
from pydantic_bot import PydanticAIBot
from context_prep_bot import prep_context
from removeRepeatingContext import remove_repeating_context

# Change current dir to Database folder to access functions there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Database')))
from queryDB import query_database

#bot = OpenAI_GPT_Bot(model="gpt-4o-mini",temperature=0.7)
bot = PydanticAIBot(model="gpt-4o-mini",temperature=0.7)

db_path = "./RAG/Database/Output"

max_docs = 3

#Set your task here:
task = "Create a project plan for my research project, called: A Thousand Games A Day"
initial_context = query_database(task,db_path,max_docs)
task = "Task: "+task

template_context = """You are part of an AI plan maker for achieving a specified task. You are given the task and some initial context information that could be useful for completing the task or creating a plan for it. Extract the useful keywords and phrases from the context and task in order to further query a database and get even more useful information, that will be fed into the next part of the system responsible for making the actual plan. Don't add any explanations or extra words, simply provide the query for the database so that your answer can be used without further preprocessing."

Context:
=========
{context}
=========
"""

combined_context = template_context.format(context=initial_context)

combined_context = remove_repeating_context(combined_context)

bot_response = str(bot.ask(question=task, context=combined_context))

#Actual answer with final db query

post_keywords_query = query_database(bot_response,db_path,max_docs+2)

template_context2 = """You are a plan builder. You are given a task that needs to be completed and some information context about the problem at hand, extracted by querying a vector database for text that is similar to the task. Use all provided information to build a step by step plan for an autonomous agent to follow in order to complete the task. Provide small enough and actionable steps. Don't add any explanations or extra words, simply provide the plan steps so that your answer can be used without further preprocessing."

Information Context:
=========
{context}
=========
"""

combined_context2 = template_context2.format(context=(post_keywords_query))
combined_context2 = remove_repeating_context(combined_context2)
final_bot_response = str(bot.ask(question=task, context=combined_context2))

print("Final Response:\n\n\n-------------------------------------\n\n")
print(task)
print("\n\n")
print(final_bot_response)