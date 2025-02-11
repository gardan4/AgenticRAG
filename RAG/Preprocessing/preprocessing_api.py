from PDF.extract_pdf import pdf_to_elements
from flask import Flask, request, jsonify
import json
import uuid
import os

app = Flask(__name__)

tempFolder = "/app/tmp"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"message": "Welcome to this API test"}), 200

def elements_to_json(elements):
    # Create a list of dictionaries for each element
    elements_list = [{"category": element.category, "text": element.text} for element in elements]
    
    # Convert the list to a JSON string
    return json.dumps(elements_list, indent=4)

@app.route("/process-pdf", methods=["POST"])
def extract_pdf_yolox():
    # Check if a file is in the request
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400

    pdf_file = request.files['pdf']

    # Check if the user has uploaded a file
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary directory
    unique_filename = f"{uuid.uuid4()}_{pdf_file.filename}"
    pdf_path = os.path.join(tempFolder, unique_filename)
    pdf_file.save(pdf_path)

    try:
        extracted_json = elements_to_json(pdf_to_elements(pdf_path))
        #extracted_json = jsonify({"Text": pdf_to_elements(pdf_path).element[40].text})
        os.remove(pdf_path)  # Clean up
    except Exception as e:
        os.remove(pdf_path)
        return jsonify({"error": str(e)}), 500

    # Return the extracted text as json
    return extracted_json, 200


app.run(host="0.0.0.0", port=5000, debug=True)
