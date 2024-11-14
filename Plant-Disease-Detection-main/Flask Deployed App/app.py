import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN  # Ensure CNN.py file is present and correctly implemented
import numpy as np
import torch
import pandas as pd

# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the model
model = CNN.CNN(39)
model_file_path = r"plant_disease_model_1_latest.pt"  # Adjust path as needed
try:
    model.load_state_dict(torch.load(model_file_path))
except FileNotFoundError:
    print(f"Model file not found: {model_file_path}")
    raise  # Stop the program if the model file cannot be found
model.eval()

def prediction(image_path):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        
        # Create uploads directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image.save(file_path)
        print(f"Saved image to: {file_path}")

        pred = prediction(file_path)
        if pred is None:  # Check if prediction failed
            return render_template('error.html', message="Failed to process the image.")

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template('submit.html',
                               title=title,
                               desc=description,
                               prevent=prevent,
                               image_url=image_url,
                               pred=pred,
                               sname=supplement_name,
                               simage=supplement_image_url,
                               buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

