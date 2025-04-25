from pydantic_bot import PydanticAIBot

def code_solution(problem,language):
    """
    This function takes a problem and a language as input and returns the best code solution for the problem in the specified language.
    """
    # Initialize the PydanticAIBot with the desired model and temperature
    bot = PydanticAIBot(model="gpt-4o-mini", temperature=0.7)

    # Ask the bot for the best code solution to the problem in the specified language
    code = bot.ask(question="Problem: "+problem, context="You are an assistant that is tasked with writing code to solve a problem. Write the code in "+language+" only, without any extra text or explanations, since your response will be executed and it needs to be a valid script file. Be specific with the code. If you are aware of multiple solutions that work, pick one. Your code must solve the problem upon execution. Make sure everything your code needs is in that file. No extra files can be done. You can use external libraries and import them. You dont need to specify them or install them, simply import them in the start of the file.")
    
    return code