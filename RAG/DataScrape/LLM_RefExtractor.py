import os
import openai
from dotenv import load_dotenv
import PyPDF2
import json
import requests
from bs4 import BeautifulSoup
import re
import io
import glob

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
            #return text after removing special characters that could break the json.
    return text.replace(":","/").replace("\"","").replace("\\","").replace("“","").replace("”","").replace("‘","").replace("’","")

def extract_text_from_online_pdf(pdf_url):
    """Fetch a PDF from a URL and extract its text."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with io.BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        return text if text.strip() else ""
    except Exception as e:
        return f"Error fetching PDF: {e}"


def extract_references_from_text(text):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": '''I need you to provide me in JSON format information from the text in the prompt. 
                                          The text consists of research topics, and for each topic, I want you to extract the title, 
                                          the (references)/(extra reading material) written as text and the URLs as well if those are present. Seperate the name of the reference and the found url with |. If only url is found it must be on the right of |, and if only text reference is found it must be on the left of |. 
                                          The extracted references should be stored in a JSON array with the fields 'Title' and 'References'.
                                          Only write information that is present in the prompt (Copy paste the relevant stuff). Don't generate any new text.
                                          Reply with just the json, with no extra text.
                                          Here is an example json structure: 
                                          {
                                            ResearchTopics:[
                                                {
                                                    "Title": "Research Topic 1",
                                                    "References": [
                                                    "Paper on examples written by Dr. John Doe | https://www.example.com",
                                                    "Paper on examples2 written by Dr. Smith | http://www.example2.com"
                                                    ]
                                                },
                                                {
                                                    "Title": "Research Topic 2",
                                                    "References": [
                                                    "Other paper on examples written by Dr. John Doe |",
                                                    "Paper on stuff written by Prof. Karl |"
                                                    ]
                                                },
                                                {
                                                    "Title": "Research Topic 3",
                                                    "References": [
                                                    "|http://www.exampleOnlyUrlFound.com",
                                                    "Some document |"http://www.normalexampleagain.com""
                                                    ]
                                                }
                                            ]
                                          }
                                          Always return just the json structure with no extra words and spaces. Make sure the whole json is on 1 line with no tabs or spaces, unless they are in the field values. If there are any special characters in the text, that make break the json, replace them, otherwise leave the text as is.
                                          '''
    },
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

def extract_html_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"
    
def clean_html(html_content, base_url=None):
    soup = BeautifulSoup(html_content, "html.parser")

    # Check for meta refresh redirect
    meta_refresh = soup.find("meta", attrs={"http-equiv": "REFRESH"})
    if meta_refresh and "url=" in meta_refresh["content"]:
        redirect_url = meta_refresh["content"].split("url=")[-1]
        if base_url and not redirect_url.startswith("http"):
            redirect_url = base_url + redirect_url  # Convert relative URL to absolute
        print(f"Redirect detected! Fetching: {redirect_url}")
        try:
            redir_content = extract_text_from_online_pdf(redirect_url)
            return redir_content
        except Exception as e:
            print(f"Error fetching redirected URL: {e}")
            return ''
    
    # Extract text from meta tags if they contain useful content
    meta_abstract = soup.find("meta", attrs={"name": "citation_abstract"})
    abstract_text = meta_abstract["content"].strip() if meta_abstract else ""

    # Remove script, style, and unwanted tags
    for tag in soup(["script", "style", "footer", "nav", "aside"]):
        tag.extract()

    # Extract text from relevant tags
    main_text = " ".join(p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3", "article", "section"]))

    # Normalize spaces
    main_text = " ".join(main_text.split())

    # Combine extracted text and abstract
    final_text = f"Abstract: {abstract_text}\n\n{main_text}" if abstract_text else main_text

    return final_text if final_text.strip() else ""

def extract_abstract_from_html(html):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "If the following html/text contains an abstract for a research paper, extract the abstract word for word and provide it in the response without any extra explanations. "
            "If the html/text does not contain an abstract, it might contain information that is considered the abstract. If the information feels irrelevant or there is nothing of substance found in the text, simply respond only with 'No abstract found'."},
            {"role": "user", "content": html}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

def save_abstract_to_file(title, ref_title, abstract, dir, counter):
    os.makedirs(dir, exist_ok=True)

    # Sanitize filename (limit length and remove invalid characters)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in ref_title)

    # Limit filename length to avoid Windows path limits - Truncate to 100 chars
    safe_title = safe_title[:100]

    file_name = f"{safe_title}_Ref_{counter}.txt"
    file_path = os.path.join(dir, file_name)

    # Normalize path to avoid issues with mixed separators
    file_path = os.path.normpath(file_path)

    # Write the title and abstract to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Reference to {title}\n\n{abstract}")

    return file_path
    
def preprocess_json_string(json_str):
    try:
        # Remove unnecessary artifacts
        json_str = json_str.replace("\\n", "").replace("\n", "").replace("```", "")
        json_str = json_str.replace("json", "").replace("json:", "")

        # Fix incorrectly escaped backslashes before quotes
        json_str = re.sub(r'\\+"', '"', json_str)

        # Remove unwanted spaces in hyphenated words
        json_str = re.sub(r"\b(\w+)\s-\s(\w+)\b", r"\1-\2", json_str)

        # Fix incorrectly escaped quotes in Titles and References
        json_str = re.sub(r'\\"', '"', json_str)

        # Fix Titles with extra quotes or backslashes
        json_str = re.sub(r'(?<="Title":)"([^"]*?)"', r'"\1"', json_str)

        # Fix References with extra quotes or backslashes
        json_str = re.sub(r'(?<="References":\[)"([^"]*?)"', r'"\1"', json_str)

        # Attempt to parse the JSON
        return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

def save_abstracts_from_refs(json_data,output_path):
    data = preprocess_json_string(json_data)
    
    #If json was broken, return None
    if not data:
        print("Failed to parse JSON.")
        return None
    
    for topic in data.get("ResearchTopics", []):
        title = topic.get("Title", "Unknown Title")
        references = topic.get("References", [])
        
        refCounter = 0
        for ref in references:
            refCounter += 1
            ref = ref.strip()  # Ensure there are no leading/trailing spaces

            if "|" in ref:  # Check if the reference contains '|'
                ref_title, ref_url = ref.split('|', 1)
                ref_title = ref_title.strip()
                ref_url = ref_url.strip()
                
                # Sometimes there is no title and the url is placed in the title field
                if ref_url=="" and "http" in ref_title:
                    ref_url = ref_title

                if "http" in ref_url:
                    ref_url = ref_url.replace("///", "://")
                    
                    if ".pdf" in ref_url:
                        cleaned_text = extract_text_from_online_pdf(ref_url)
                    else:
                        html_content = extract_html_from_url(ref_url)
                        cleaned_text = clean_html(html_content, ref_url)

                    if cleaned_text:
                        abstract = extract_abstract_from_html(cleaned_text)
                        if "no abstract found" not in abstract.lower():
                            if ref_title=="":
                                ref_title = title+"_NoNameReference_"+str(refCounter)
                            save_abstract_to_file(title, ref_title, abstract, output_path, refCounter)
                else:
                    print("Invalid URL reference:", ref_url)
            else:
                print("Skipping reference without a URL:", ref)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    #Get all ref abstracts from project topic proposals
    input_folder = "./RAG/DataScrape/Input"
    output_folder = "./RAG/DataScrape/Abstracts"

    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")

        input_txt = extract_text_from_pdf(pdf_file)

        json_string = extract_references_from_text(input_txt)

        save_abstracts_from_refs(json_string, output_folder)