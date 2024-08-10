from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import easyocr
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle
from flask_cors import CORS
load_dotenv()

app = Flask(__name__)

# Enable CORS
CORS(app)

@app.route("/api/v2/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/v2/predict-image", methods=["POST"])
def predict1():
    image = request.files["image"]
    image_name = image.filename
    try:
        image.save("./images/" + image_name)
        print("Image saved successfully")
    except Exception as e:
        return jsonify({"error": str(e)})
    model = YOLO("./crop_best.pt")
    model.predict("./images/" + image_name, save=True, imgsz=320, conf=0.5)[0]
    
    path = 'runs/detect'
    # Sort directories by modification time
    dir_list = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
    
    last_dirpath = os.path.join(path, dir_list[0])
    dir_list = os.listdir(last_dirpath)
    reader = easyocr.Reader(['en'])
    IMAGE_PATH = os.path.join(last_dirpath, dir_list[0])
    result = reader.readtext(IMAGE_PATH, paragraph="False")
    
    if result:
        print(result[0][-1])
        return generate_text()
    else:
        return "No text detected in the image"

@app.route("/api/v2/generate-text", methods=["GET"])
def generate_text():
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
    
    path = './runs/detect'
    # Sort directories by modification time
    dir_list = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
    
    last_dirpath = os.path.join(path, dir_list[0])
    dir_list = os.listdir(last_dirpath)
    IMAGE_PATH = os.path.join(last_dirpath, dir_list[0])
    reader = easyocr.Reader(['en'])
    result = reader.readtext(IMAGE_PATH, paragraph="False")

    if result:
        plant_disease = result[0][-1]
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = answer(plant_disease, model)
        print(response.text)
        clean_text=response.text.replace("```json","").replace("```","")
        return clean_text
    else:
        return "No text detected in the image"
    
@app.route("/api/v2/yield", methods=["POST"])
def yield_main():
    with open("model_pickle.pkl","rb") as f:
        mp= pickle.load(f)
    
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

if __name__ == "__main__":
    app.run(debug=True)
