from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import easyocr
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from flask_cors import CORS
import numpy as np
import cv2
import pickle

load_dotenv()

app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

# Load the model once at the start
model = YOLO("./crop_best.pt")

@app.route("/api/v2/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/v2/predict-image", methods=["POST"])
def predict1():
    image = request.files.get("image")

    if not image:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Read the image directly from the request without saving
        image_np = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Process image with the preloaded YOLO model
        results = model.predict(img, imgsz=320, conf=0.5)

        if results:
            detected_text = results[0].names[int(results[0].boxes[0].cls)] if results[0].boxes else None
            
            if detected_text:
                # Directly generate the text using the detected text
                return generate_text(detected_text)
            else:
                return jsonify({"error": "No text detected in the image"}), 400
        else:
            return jsonify({"error": "No objects detected in the image"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/v2/generate-text", methods=["POST"])
def generate_text(detected_text=None):
    if detected_text is None:
        data = request.get_json()
        detected_text = data.get('detected_text', None)
    
    if not detected_text:
        return jsonify({"error": "No detected text provided"}), 400

    get_api = os.getenv('API_KEY')
    genai.configure(api_key=get_api)

    def answer(text, model):
        prompt_template = PromptTemplate(
            input_variables=['plants_disease'],
            template="""
            if the plant disease is unclear make the most probable as an answer
            you are a plant disease specialist. your job is to answer how to aid the plant when it is suffering from
            {plants_disease}. give me the answer in the following format:
            your plant is suffering from the {plants_disease}
            the following are the ways to aid the plant's disease.
            make sure that the response does not have "**" or "*" symbols in it and the response should be in JSON Format. And the JSON Format should contain two keys "diagnosis" and "treatment". And the value of the key "diagnosis" should be string and the value of the key "treatment" should be a array of strings.
            JSON Format must follow the above conditions.
            """
        )
        prompt = prompt_template.format(plants_disease=text)
        response = model.generate_content(prompt)
        return response

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = answer(detected_text, model)
        clean_text = response.text.replace("```json", "").replace("```", "")
        return clean_text
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/v2/yield", methods=["POST"])
def yield_main():
    try:
        with open("model_pickle.pkl", "rb") as f:
            mp = pickle.load(f)

        crop = request.form["crop"]
        crop_year = request.form["crop_year"]
        season = request.form["season"]
        state = request.form["state"]
        area = request.form["area"]
        rainfall = request.form["rainfall"]
        fertilizer = request.form["fertilizer"]
        pesticide = request.form["pesticide"]

        x = [[crop, crop_year, season, state, area, rainfall, fertilizer, pesticide]]
        print(x)
        y_pred = mp.predict(x)

        print(y_pred)
        return str(y_pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
