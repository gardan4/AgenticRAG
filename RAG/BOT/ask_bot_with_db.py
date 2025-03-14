import sys
import os
from openai_bot import OpenAI_GPT_Bot
from pydantic_bot import PydanticAIBot
from context_prep_bot import prep_context

# Change current dir to Database folder to access functions there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Database')))
from queryDB import query_database

#bot = OpenAI_GPT_Bot(model="gpt-4o-mini",temperature=0.7)
bot = PydanticAIBot(model="gpt-4o-mini",temperature=0.7)

db_path = "./RAG/Database/Output"

max_docs = 3

#Ask your question here:
question = "Help me with my research topic about training data shuffling and whether or not that is good. What should I learn before I start with my project? What literature can be useful? For the literature you propose, give a short explanation of what it is."

initial_context = "\n".join(query_database(question,db_path,max_docs))

information_context = prep_context(db_path, 5, question, initial_context)

template_context = """You are an assistant that answers questions based strictly on the context provided below.
If the question is not directly answerable from the provided context, simply respond with "I don't know." 
Do not make up answers or use your pre-trained knowledge to answer the question.
Do not ask follow-up questions, only provide answers.
Answer the question directly, without any introductory phrases or explanations. 

Context:
=========
{context}
=========
"""

combined_context = template_context.format(context=information_context)

bot_response = str(bot.ask(question=question, context=combined_context))

print("Final Response:\n\n\n-------------------------------------\n\n")
print(bot_response)