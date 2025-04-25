from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

class OpenAI_GPT_Bot:
    """
    Creates a bot object that simply replies to the provided prompt.

    Args:
    model (String): Choose a chat GPT model (e.g., "gpt-3.5-turbo").
    temperature (int): Controls randomness of the model's output (0 for minimal randomness).
    template (String): Instructions for the bot to follow, if any.
    
    Returns:
    string: The bot's response to the prompt.
    """
    
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.template = PromptTemplate(input_variables=["question"], template="{question}")

        # Initialize the model
        self.llm = ChatOpenAI(temperature=temperature, model=model)

        # Use LLMChain to connect the prompt to the model
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.template)

    def ask(self, question, context):
        """
        Sends the question and context to the chatbot and returns its response.
        
        Args:
        question (string): The question to be answered by the bot.
        context (string): The context or background information to help answer the question.
        
        Returns:
        string: The bot's response to the prompt.
        """
        prompt = context+" \n "+question
        response = self.llm_chain.run({
            "question": prompt
        })
        
        return response
