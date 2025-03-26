from plan_maker import make_plan
from pydantic_bot import PydanticAIBot

#plan = make_plan(task = "Create a project plan for my research project, called: A Thousand Games A Day")
plan='''1. Define project objectives and goals for "A Thousand Games A Day."
2. Conduct a literature review on General Game Playing (GGP) and AI methodologies.
3. Identify and select the 1,000 Ludii games to be included in the project.
4. Design the architecture for the AI agents using Monte Carlo Tree Search (MCTS).
5. Develop a prototype MCTS agent in Java for testing.
6. Implement the AI's move planning strategies focusing on efficiency.
7. Create a time management system to limit thinking time to 1 minute per agent per trial.
8. Set up the testing environment for self-play trials between AI agents.
9. Run preliminary tests to evaluate the performance of the MCTS agent.
10. Analyze test results and refine AI strategies based on findings.
11. Document the development process and results.
12. Prepare a final report summarizing project outcomes and contributions to GGP.
13. Submit the project for evaluation, including potential entry into the Ludii AI Kilothon competition.'''

bot = PydanticAIBot(model="gpt-4o-mini",temperature=0.7)

#Split the plan into steps in a list
steps = plan.split("\n")

#For each step ask the bot to pick a tool to be used for completing the task
print("\n\n")
for step in steps:
    tool = bot.ask(question="Task: "+step, context="You are an assistant that is tasked with picking the best tool for completing a task. Only respond with the tool, without any extra text or explanations. Be specific with the tool. If you are aware of multiple tools that work, pick one. Keep in mind the tasks are for an AI agent to complete. If it is a programming task simply choose the best language for completing it.")
    print(f"Step: {step}\nTool: {tool}\n\n")