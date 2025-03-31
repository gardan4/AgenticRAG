# src/agentic_framework/agent.py

import re
import torch

from src.agentic_framework.tools import CalculatorTool, SearchTool
from src.utils.model_loader import load_model


class AgenticModel:
    def __init__(
        self,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quantize=None,
        max_length=128
    ):
        self.tokenizer, self.model = load_model(model_name, quantize=quantize)
        self.max_length = max_length

        # A simple pattern to detect when the LLM wants to call a tool
        # For example, the model might output: "<tool> calculator: 3+7"
        # You can define your own format or use existing frameworks.
        self.tool_call_pattern = re.compile(r"<tool>\s*(\w+):\s*(.*)")

    def _format_conversation(self, conversation):
        # 1) Start with a system prompt that enumerates tools
        system_message = (
            "SYSTEM: You are an AI assistant with access to the following tools. "
            "Use them when needed:\n"
            " - calculator: for math expressions\n"
            " - search: for general queries\n"
            "To use a tool: <tool> tool_name: tool_input\n"
            "You will receive the tool's output as a separate message from 'tool'."
        )
        # 2) Add a few-shot demonstration
        few_shot_example = """
        USER: What is 5 * 6?
        ASSISTANT: <tool> calculator: 5*6
        TOOL: 30
        ASSISTANT: The answer is 30.
        """

        # 3) Then your conversation so far
        lines = [system_message, few_shot_example]
        for speaker, text in conversation:
            lines.append(f"{speaker.upper()}: {text}")
        return "\n".join(lines) + "\nASSISTANT:"


    def _parse_tool_call(self, text):
        """
        Check if the generated text contains a request to use a tool.
        Returns (tool_name, tool_input) or (None, None) if not found.
        """
        match = self.tool_call_pattern.search(text)
        if match:
            tool_name = match.group(1).strip()
            tool_input = match.group(2).strip()
            return tool_name, tool_input
        return None, None

    def run(self, user_query: str, max_steps=5, max_length=2048):
        """
        Conduct a conversation with the user query. 
        The agent can use a tool if it outputs a special command.
        """
        conversation = [
            ("user", user_query),
        ]

        # We’ll do a short loop for demonstration
        for step in range(max_steps):
            prompt = self._format_conversation(conversation)

            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # The chunk after "ASSISTANT:" is the actual new text
            # Since we appended "ASSISTANT:" at the end of prompt, let's isolate the new generation
            # We can do a more robust approach to parse partial text.
            assistant_reply = generated_text.split("ASSISTANT:")[-1].strip()

            # Check if the model wants to use a tool
            tool_name, tool_input = self._parse_tool_call(assistant_reply)
            if tool_name and tool_name in self.tools:
                # Call the tool
                tool_result = self.tools[tool_name].run(tool_input)
                # Add the tool result as if the model "saw" it
                conversation.append(("assistant", assistant_reply))
                conversation.append(("tool", tool_result))

            else:
                # No tool call detected or unknown tool
                conversation.append(("assistant", assistant_reply))
                # If the agent has concluded an answer, we break
                # You might define a special token or condition to end the loop
                if "<tool>" not in assistant_reply:
                    # Let's say if there's no further tool call, we end
                    break

        # Return the last assistant message as the final answer
        return conversation[-1][1]


def demo_agentic_interaction():
    agent = AgenticModel(quantize="8bit")
    user_input = "What is the square root of 144?"
    print("START--------------------------")
    print("User:", user_input)
    answer = agent.run(user_input)
    print("Agent:", answer)


if __name__ == "__main__":
    demo_agentic_interaction()
