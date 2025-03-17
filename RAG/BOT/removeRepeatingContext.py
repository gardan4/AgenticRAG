import difflib

def remove_repeating_context(doc_text, threshold=0.9):
    # Split the document into chunks using "Filename:" as the separator
    paragraphs = doc_text.split("Filename:")
    
    unique_paragraphs = []
    initial_answers = []
    
    first_paragraph = True
    for para in paragraphs:
        if not first_paragraph:
            para = "Filename:" + para
        first_paragraph = False
        
        # Check if there is an "Initial Answer:" section in the paragraph
        parts = para.split("Initial Answer:", 1)
        main_text = parts[0].strip()
        initial_answer = parts[1].strip() if len(parts) > 1 else None

        is_duplicate = False

        for unique_para in unique_paragraphs:
            similarity = difflib.SequenceMatcher(None, main_text, unique_para).ratio()
            if similarity > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_paragraphs.append(main_text)

        if initial_answer:
            initial_answers.append(f"Initial Answer:\n{initial_answer}")
    
    # Merge the cleaned paragraphs and their respective initial answers
    final_text = "\n\n".join(unique_paragraphs)
    if initial_answers:
        final_text += "\n\n" + "\n\n".join(initial_answers)

    return final_text