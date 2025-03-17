import sys
import os
from openai_bot import OpenAI_GPT_Bot
from pydantic_bot import PydanticAIBot

# Change current dir to Database folder to access functions there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Database')))
from queryDB import query_database

#bot = OpenAI_GPT_Bot(model="gpt-4o-mini",temperature=0.7)
bot = PydanticAIBot(model="gpt-4o-mini",temperature=0.7)

def prep_context(db_path, max_docs, user_question, initial_context):
    template_context = """You are a part of a system for question answering. You are not supposed to answer the question but assist another part in doing so. You are given a question or a task and partial context for answering it. If the context given is not enough to be able to provide a complete answer, give keywords or questions that should be used to query a vector database in order to pull relevant context. Never provide the actual answer, if the context is sufficient, just reply with END OF CONTEXT. If you do provide keywords, only provide those and no other extra text, since your whole response is going to be used for the next query! Only provide a few most important keywords or questions, to get better results. Make sure you are making good use of the context you already have. If you see there is something in it you would like to learn more about, simply pass those words or sentences along so more context is extracted for that part of the previous context.

    Context:
    =========
    {context}
    =========
    """

    combined_context = template_context.format(context=initial_context)

    bot_keywords = str(bot.ask(question=user_question, context=combined_context)).replace("END OF CONTEXT","").replace("\n","")

    additional_context = query_database(bot_keywords,db_path,max_docs)

    final_context = initial_context + "\n" + additional_context

    return final_context