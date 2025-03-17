from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from CustomElement import CustomElement
from CustomElementMetadata import CustomElementMetadata

def get_str_after(text, split_str):
    """
    Split on the first occurrence of `split_str` and take everything after it

    Args:
        text (str): The string that the function is going to do the extraction on
        split_str (str): The string or character that the function is going to split the text on

    Returns:
        A string that is part of the text that was taken as input. Taking only the part after the initial split_str found in the text.
    """
    split_index = text.find(split_str)
    
    if split_index != -1:
        # Extract everything after the `split_str`
        return text[split_index + len(split_str):].strip()
    else:
        return text  # Return the whole input if the split_str is not found in it

def custom_chunk_by_title(input_elements, characterLimit=1000, overlap=0, last_symbol=".", last_overlap_symbol=" ", min_chunk_length=150):
    """
    Chunks elements into groups of text chunks, with each chunk containing up to the specified character limit. 
    Each chunk starts with the title associated with it.
    Allows optional overlap between chunks to repeat characters from the end of one chunk at the start of the next.
    Optionally, chunks end at the last occurrence of a specified symbol before reaching the character limit.

    Args:
        input_elements (list[CustomElement]): A list of CustomElement objects to be chunked based on character limit.
        characterLimit (int): The maximum cumulative character count for each chunk.
        overlap (int, optional): Number of characters to overlap between consecutive chunks.
        last_symbol (str, optional): Character to end chunk on before reaching character limit. 
        last_overlap_symbol (str, optional): Character to make sure the overlap that is copied to the next chunk is not starting from a cut off word
        min_chunk_length (int, optional): Minimum length of chunks. Chunks smaller than this will be discarded or combined.

    Returns:
        list[CustomElement]: A list of CustomElement objects representing chunks of text that respect the character limit 
                       (with optional overlap applied).
    """
    output_chunks = []
    current_chunk_text = ""
    title_text = ""  # Store the last encountered title

    last_filename = input_elements[0].metadata.filename   
    last_trustscore = input_elements[0].metadata.trust_score
    
    for element in input_elements:
        # Check if the element is a title
        if element.category.lower() == "title":
            # If there's any existing content in the current chunk, save it before starting a new one
            if current_chunk_text:
                if len(current_chunk_text) >= min_chunk_length:
                    custom_metadata = CustomElementMetadata(filename=last_filename, trust_score=last_trustscore)
                    output_chunks.append(CustomElement(text=current_chunk_text, category="Chunk", metadata=custom_metadata))
                else:
                    # If the chunk is too small, combine with next chunk (skip saving this one)
                    pass
                current_chunk_text = ""  # Reset for next chunk

            # Begin a new chunk with the title element's text
            title_text = element.text.replace("\xa0", " ")
            current_chunk_text = title_text + ": "

        # Process the element's text
        text_to_process = element.text.replace("\xa0", " ")

        while text_to_process:
            # Calculate remaining space
            remaining_space = characterLimit - len(current_chunk_text)
            
            if len(text_to_process) <= remaining_space:
                # Add text if it fits in the current chunk
                current_chunk_text += text_to_process
                text_to_process = ""
            else:
                # Find last occurrence of `last_symbol` before `remaining_space`
                chunk_text = text_to_process[:remaining_space]
                if last_symbol:
                    last_index = chunk_text.rfind(last_symbol)
                    if last_index != -1:
                        chunk_text = chunk_text[:last_index+1]  # Include the last symbol
                        current_chunk_text += chunk_text
                        text_to_process = text_to_process[len(chunk_text):]
                else:
                    current_chunk_text += chunk_text
                    text_to_process = text_to_process[len(chunk_text):]

                # Save the chunk if it's big enough or combine it with the next chunk
                if len(current_chunk_text) >= min_chunk_length:
                    custom_metadata = CustomElementMetadata(filename=last_filename, trust_score=last_trustscore)
                    output_chunks.append(CustomElement(text=current_chunk_text, category="Chunk", metadata=custom_metadata))
                else:
                    # If chunk is too small, combine it with the next one (don't save it yet)
                    pass

                # Reset for the next chunk, applying overlap if specified
                if overlap > 0:
                    current_chunk_text = title_text + ": " + get_str_after(current_chunk_text[-overlap:], last_overlap_symbol)
                else:
                    current_chunk_text = title_text + ": "

    # Append any remaining text in the current chunk, only if it's large enough
    if current_chunk_text:
        if len(current_chunk_text) >= min_chunk_length:
            custom_metadata = CustomElementMetadata(filename=last_filename, trust_score=last_trustscore)
            output_chunks.append(CustomElement(text=current_chunk_text, category="Chunk", metadata=custom_metadata))

    return output_chunks



def create_database(input_elements, output_db_path):
    """
    Creates a chromadb database with the input elements and saves it to the desired location.
    
    Args:
    input_elements (list(Unstructured Element)): The list of extracted elements from the input files that should be saved in the vectorstore
    output_db_path (String): The path to the folder where the vectorstore should be created
    
    Returns:
    True if it successfully created the vectorstore in the desired location.
    False if there was an issue.
    """
    try:
        elements = custom_chunk_by_title(input_elements=input_elements,overlap=140)

        # Create documents from the elements
        documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            metadata["source"] = metadata["filename"]
            # Remove metadata items that are lists
            keys_to_remove = [key for key, value in metadata.items() if isinstance(value, list)]
            for key in keys_to_remove:
                del metadata[key]

            documents.append(Document(page_content=element.text, metadata=metadata))

        embeddings = OpenAIEmbeddings()

        # Create a vector db from the documents and specify the directory for persistence
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=output_db_path)
        
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False
    
def load_database_from_dir(db_path):
    """
    Loads the vectorstore to a variable by providing the path to the folder containing it.
    
    Args:
    db_path (String): Path to the folder, where the vectorstore was saved.
    
    Returns:
    A reference to the vectorstore found and the specified location.
    """
    try:
        embeddings = OpenAIEmbeddings()
        # Initialize Chroma with the directory where the vector store is persisted
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        return vectorstore
    except Exception as e:
        print(f"Error loading database: {e}")
        return None
    
def add_documents_to_database(input_elements, vectorstore):
    """
    Adds new elements to an already existing vectorstore
    
    Args:
    input_elements (list(Unstructured Element)): The list of extracted elements from the input files that should be saved in the vectorstore
    vectorstore (Chroma): A reference to the chromadb vectorstore that should aquire the new elements.
    
    Returns:
    True if success
    False if failed
    """
    try:
        # Chunk the input elements and create documents
        elements = custom_chunk_by_title(input_elements=input_elements,overlap=140)
        new_documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            metadata["source"] = metadata["filename"]
            new_documents.append(Document(page_content=element.text, metadata=metadata))
            # Remove metadata items that are lists
            keys_to_remove = [key for key, value in metadata.items() if isinstance(value, list)]
            for key in keys_to_remove:
                del metadata[key]
        # Add new documents to the existing vector store
        vectorstore.add_documents(new_documents)
        
        print("Documents added and database successfully updated.")
        return True
    except Exception as e:
        print(f"Error adding documents to database: {e}")
        return False