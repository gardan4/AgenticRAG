from pydantic_bot import PydanticAIBot
from typing import List, Tuple

class ContextChatBot:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Creates a context-aware chatbot that responds to questions based on provided contexts.
        
        Args:
            model (str): The model to use for generating responses.
            temperature (float): Controls randomness in responses. Range: 0.0 to 1.0.
        """
        self.bot = PydanticAIBot(model=model, temperature=temperature)
        
    def chat(self, 
             question: str, 
             information_context: str = "", 
             role_context: str = "",
             chat_history: List[Tuple[str, str]] = None) -> str:
        """
        Generates a response to the user's question based on the provided contexts and chat history.
        
        Args:
            question (str): The user's question or prompt.
            information_context (str): Background information relevant to answering the question.
            role_context (str): Instructions about how the bot should behave or respond.
            chat_history (List[Tuple[str, str]]): List of previous exchanges as (user, assistant) tuples.
            
        Returns:
            str: The bot's response to the question.
        """
        # Format chat history if provided
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_text = "Chat History:\n"
            for user_msg, assistant_msg in chat_history:
                history_text += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        
        # Combine contexts, history, and question into a single prompt
        prompt = f"""
        Role and Instructions:
        {role_context}

        Information Context:
        {information_context}

        {history_text}
        Current User Question:
        {question}

        Please respond to the user's question based on the provided information context, chat history, and following the role instructions.
        """
        
        # Get response from the bot
        response = self.bot.ask(question=prompt, context="")
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = ContextChatBot(model="gpt-4o-mini", temperature=0.7)
    
    # Example contexts
    info_context = "The Earth is the third planet from the Sun and the only astronomical object known to harbor life."
    role_context = "You are a helpful astronomy expert. Provide accurate, concise information about space and astronomy."
    
    # Example chat history
    chat_history = [
        ("Tell me about planets in our solar system.", "Our solar system has eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. They orbit the Sun at different distances and have varying characteristics."),
        ("Which is the largest?", "Jupiter is the largest planet in our solar system. It's a gas giant with a mass more than 300 times that of Earth.")
    ]
    
    # Example question
    question = "What makes Earth special compared to other planets?"
    
    # Get response
    response = chatbot.chat(
        question=question,
        information_context=info_context,
        role_context=role_context,
        chat_history=chat_history
    )
    
    print(f"Question: {question}")
    print(f"Response: {response}")