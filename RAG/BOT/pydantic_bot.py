from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel

class PydanticAIBot:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Creates a PydanticAI bot object that replies to the provided prompt.

        Args:
        model (str): Choose a chat model (e.g., "gpt-3.5-turbo").
        temperature (float): Controls randomness in responses. Range: 0.0 to 1.0.
        
        Returns:
        None
        """
        # Define model settings with the specified temperature
        self.model_settings = ModelSettings(temperature=temperature)
        # Initialize the AI agent with the specified model and settings
        self.agent = Agent(model=model, model_settings=self.model_settings)

    def ask(self, question: str, context: str) -> str:
        """
        Sends the question and context to the chatbot and returns its response.
        
        Args:
        question (string): The question to be answered by the bot.
        context (string): The context or background information to help answer the question.
        
        Returns:
        string: The bot's response to the prompt.
        """
        # Combine the context and question to form the prompt
        prompt = f"{context}\n{question}"
        response = self.agent.run_sync(prompt)
        return response.data.strip()