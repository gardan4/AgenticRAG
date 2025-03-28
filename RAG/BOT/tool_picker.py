from plan_maker import make_plan
from pydantic_bot import PydanticAIBot

def pick_tool_for_task(task):
    """
    This function takes a task as input and returns the best tool for completing it.
    """
    # Initialize the PydanticAIBot with the desired model and temperature
    bot = PydanticAIBot(model="gpt-4o-mini", temperature=0.7)

    # Ask the bot for the best tool to complete the task
    tool = bot.ask(question="Task: "+step, context="You are an assistant that is tasked with picking the best tool for completing a task. Only respond with the tool, without any extra text or explanations. Be specific with the tool. If you are aware of multiple tools that work, pick one. Keep in mind the tasks are for an AI agent to complete. If it is a programming task simply choose the best language for completing it.")
    
    return tool