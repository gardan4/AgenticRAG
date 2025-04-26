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

#Ask your question here:
#question = "I need help with finding information about topics that are referenced as useful under my project topic. My project topic is called To shuffle, or not to shuffle, this is the question, and it is related to training data in neural networks."
question = "My project topic is called Deep Drug Recommender. Please find an abstract of information about one of the references of that topic and return that abstract plus the reference from which you found it."
initial_context = query_database(question,db_path,max_docs)
question="Question: "+question

information_context = prep_context(db_path, 5, question, initial_context)

template_context = """You are an assistant that answers questions based strictly on the context provided below.
If the question is not directly answerable from the provided context, simply respond with "I don't know." 
Do not make up answers or use your pre-trained knowledge to answer the question.
Do not ask follow-up questions, only provide answers.
Answer the question directly, without any introductory phrases or explanations. 

In the following context make sure to prioritize text that is marked with higher trust score. In case of conflicting information, always use the one with higher trust score or if the trust score is the same, make an educated decision on which one is more plausible.
Also use the information for file name to consider the type of source giving you the information and whether or not that should be trusted and used more or less.
Also consider any dates that you can find in the filenames or the text itself. You should prioritize more recent information if that is possible.
Context:
=========
{context}
=========
"""

combined_context = template_context.format(context=information_context)

combined_context = remove_repeating_context(combined_context)

bot_response = str(bot.ask(question=question, context=combined_context))

#Additional check if a better answer can be given.

if(not "I don't know." in bot_response):
    post_answer_check = query_database(bot_response,db_path,max_docs+2)

    template_context2 = """You are an assistant that answers questions based strictly on the context provided below.
    If the question is not directly answerable from the provided context, simply respond with "I don't know." 
    Do not make up answers or use your pre-trained knowledge to answer the question.
    Do not ask follow-up questions, only provide answers.
    Answer the question directly, without any introductory phrases or explanations. 
    In the context bellow you have the previous attempt to answer the question (initial answer) plus results from querying a vector database with that previous answer. Provide the final answer to the user, if the intial answer is not good enough or more precise answer can be given now, do so.

    Context:
    =========
    {context}
    =========
    """

    combined_context2 = template_context2.format(context=(information_context + "\n" +post_answer_check+"\n\nInitial Answer:\n"+bot_response))
    combined_context2 = remove_repeating_context(combined_context2)
    final_bot_response = str(bot.ask(question=question, context=combined_context2))
else:
    final_bot_response = bot_response
print("Final Response:\n\n\n-------------------------------------\n\n")
print(final_bot_response)