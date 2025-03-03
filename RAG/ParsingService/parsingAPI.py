from PDF.advanced_pdf import pdf_to_elements_advanced
from PDF.fast_pdf import pdf_to_elements_fast
from DOCX.extractDOCX import docx_to_elements
from HTML.extractHTML import html_to_elements
from flask import Flask, request, jsonify
import json
import uuid
import os

app = Flask(__name__)

tempFolder = "/app/tmp"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

def elements_to_json(elements):
    # Create a list of dictionaries for each element
    elements_list = [{"category": element.category, "text": element.text} for element in elements]
    
    # Convert the list to a JSON string
    return json.dumps(elements_list, indent=4)

@app.route("/process-pdf-fast", methods=["POST"])
def extract_pdf_fast():
    
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    pdf_file = request.files['file']

    # Check if the user has uploaded a file
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary directory
    unique_filename = f"{uuid.uuid4()}_{pdf_file.filename}"
    pdf_path = os.path.join(tempFolder, unique_filename)
    pdf_file.save(pdf_path)

    try:
        print("Test")
        extracted_json = elements_to_json(pdf_to_elements_fast(pdf_path))
        os.remove(pdf_path)
    except Exception as e:
        os.remove(pdf_path)
        return jsonify({"error": str(e)}), 500

    # Return the extracted text as json
    return extracted_json, 200

@app.route("/process-pdf-yolox", methods=["POST"])
def extract_pdf_yolox():
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    pdf_file = request.files['file']

    # Check if the user has uploaded a file
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary directory
    unique_filename = f"{uuid.uuid4()}_{pdf_file.filename}"
    pdf_path = os.path.join(tempFolder, unique_filename)
    pdf_file.save(pdf_path)

    try:
        extracted_json = elements_to_json(pdf_to_elements_advanced(pdf_path))
        os.remove(pdf_path)
    except Exception as e:
        os.remove(pdf_path)
        return jsonify({"error": str(e)}), 500

    # Return the extracted text as json
    return extracted_json, 200

@app.route("/process-docx", methods=["POST"])
def extract_docx():
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    docx_file = request.files['file']

    # Check if the user has uploaded a file
    if docx_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary directory
    unique_filename = f"{uuid.uuid4()}_{docx_file.filename}"
    docx_path = os.path.join(tempFolder, unique_filename)
    docx_file.save(docx_path)

    try:
        extracted_json = elements_to_json(docx_to_elements(docx_path))
        os.remove(docx_path)
    except Exception as e:
        os.remove(docx_path)
        return jsonify({"error": str(e)}), 500

    # Return the extracted text as json
    return extracted_json, 200

@app.route("/process-html", methods=["POST"])
def extract_html():
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    html_file = request.files['file']

    # Check if the user has uploaded a file
    if html_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary directory
    unique_filename = f"{uuid.uuid4()}_{html_file.filename}"
    html_path = os.path.join(tempFolder, unique_filename)
    html_file.save(html_path)

    try:
        extracted_json = elements_to_json(html_to_elements(html_path))
        os.remove(html_path)
    except Exception as e:
        os.remove(html_path)
        return jsonify({"error": str(e)}), 500

    # Return the extracted text as json
    return extracted_json, 200


app.run(host="0.0.0.0", port=5000, debug=True)
