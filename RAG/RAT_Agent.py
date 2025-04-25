from Bot.plan_maker import make_plan
from Bot.tool_picker import pick_tool_for_task
from Bot.bot_programmer import code_solution

plan = make_plan(task = "Create a project plan for my research project, called: A Thousand Games A Day")

#Split the plan into steps in a list
steps = plan.split("\n")

#For each step ask the bot to pick a tool to be used for completing the task
print("\n\n")
code_counter = 0
for step in steps:
    tool = pick_tool_for_task(step)
    print(f"Step: {step}\nTool: {tool}\n\n")
    if("python" in tool.lower()):
        code_counter+=1
        solution_code = code_solution(step, "Python")

        if solution_code.startswith("'''python") and solution_code.endswith("'''"):
            clean_code = solution_code[9:-3].strip()
        else:
            clean_code = solution_code 

        # Save the string to a .py file
        file_name = f"temp_solution_script_{code_counter}.py"
        with open(file_name, "w") as file:
            file.write(clean_code)