#To run this you need to install the packages listed in ./ParsingService/parsing_requirements.txt
#You also need to have tessaract-ocr and poppler utils installed locally

from ParsingService.PDF.advanced_pdf import pdf_to_elements_advanced
from ParsingService.PDF.fast_pdf import pdf_to_elements_fast

def parse_and_print_pdf(file_path):
    elements = pdf_to_elements_advanced(file_path)
    for element in elements:
        print(element)

if __name__ == "__main__":
    file_path = "C:/Users/Stefan/Workspace/Thesis/AgenticRAG/RAG/Database/Input/test.pdf"
    parse_and_print_pdf(file_path)