from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import os
import tensorflow as tf
import numpy as np
import folium
from PIL import Image
import requests
from werkzeug.utils import secure_filename
from fpdf import FPDF
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Secret key for session management
app.secret_key = os.getenv('SECRET_KEY')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the model
model = tf.keras.models.load_model('pneumonia_detection_model.h5')

# API Keys from environment variables
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocess the X-ray image before prediction
def preprocess_image(filepath):
    try:
        img = Image.open(filepath).convert('RGB')  # Convert to RGB
        img = img.resize((150, 150))  # Resize to match model input shape
        img = np.array(img) / 255.0  # Normalize pixel values
        img = img.reshape(1, 150, 150, 3)  # Reshape for the model
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Generate a report using GPT-Neo
def generate_report_with_gpt_neo(prompt):
    headers = {
        'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'inputs': prompt,
        'parameters': {
            'max_length': 1500,  # Increase the length for a longer report
            'temperature': 0.7,  # Controls creativity/randomness
            'top_p': 0.9,        # Adjusts diversity of choices
            'top_k': 50          # Reduces repetitive responses
        }
    }
    try:
        response = requests.post('https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B', headers=headers, json=data)
        response.raise_for_status()
        generated_text = response.json()[0]['generated_text']
        generated_text = generated_text.replace(prompt, '')  # Remove the prompt
        return generated_text
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Error generating report."

# Find the nearest hospital using Overpass API
def find_nearest_hospital(latitude, longitude):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node
      ["amenity"="hospital"]
      (around:10000,{latitude},{longitude});
    out body;
    """
    try:
        response = requests.get(overpass_url, params={'data': query})
        response.raise_for_status()
        data = response.json()

        if data['elements']:
            hospital = data['elements'][0]
            name = hospital.get('tags', {}).get('name', 'No hospital found')
            street = hospital.get('tags', {}).get('addr:street', 'Street not available')
            city = hospital.get('tags', {}).get('addr:city', 'City not available')
            state = hospital.get('tags', {}).get('addr:state', 'State not available')
            country = hospital.get('tags', {}).get('addr:country', 'Country not available')

            address = f"{street}, {city}, {state}, {country}"
            hospital_lat = hospital['lat']
            hospital_lon = hospital['lon']
            return name, address.strip(), hospital_lat, hospital_lon
        return "No hospital found", "Address not available", latitude, longitude
    except Exception as e:
        print(f"Error fetching hospital info: {e}")
        return "Error", "Error fetching hospital info", latitude, longitude

# Use OpenCage to get location from address
def get_location_from_address(address):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={OPENCAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"OpenCage API Response: {data}")  # Debugging
        if data['results']:
            latitude = data['results'][0]['geometry']['lat']
            longitude = data['results'][0]['geometry']['lng']
            return latitude, longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None, None

# Get current location using IP-based geolocation
def get_current_location():
    ip_geolocation_url = 'https://ipinfo.io/json'
    response = requests.get(ip_geolocation_url)

    if response.status_code == 200:
        data = response.json()
        loc = data['loc'].split(',')
        latitude = loc[0]
        longitude = loc[1]
        return latitude, longitude
    else:
        return None, None

# Function to create a PDF report
def create_pdf_report(patient_name, patient_age, patient_gender, pneumonia_status, symptoms, report_text, hospital_name, hospital_address):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Age: {patient_age}", ln=True)
    pdf.cell(0, 10, f"Gender: {patient_gender}", ln=True)
    pdf.cell(0, 10, f"Pneumonia Status: {pneumonia_status}", ln=True)
    pdf.cell(0, 10, f"Symptoms: {symptoms}", ln=True)
    pdf.ln(10)
    
    pdf.multi_cell(0, 10, "Detailed Report:\n" + report_text)
    pdf.ln(10)
    
    pdf.cell(0, 10, "Nearest Hospital Information", ln=True)
    pdf.cell(0, 10, f"Name: {hospital_name}", ln=True)
    pdf.cell(0, 10, f"Address: {hospital_address}", ln=True)

    pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{patient_name}_pneumonia_report.pdf")
    pdf.output(pdf_file_path)
    return pdf_file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report', methods=['POST'])
def report():
    if 'xray_image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['xray_image']
    user_location = request.form.get('user_location')
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_gender = request.form.get('patient_gender')
    symptoms = request.form.get('symptoms')

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image and make prediction
    image = preprocess_image(filepath)
    if image is None:
        flash('Error processing the image')
        return redirect(url_for('index'))

    try:
        prediction = model.predict(image)
        has_pneumonia = prediction[0][0] > 0.5  # Simple binary classification
    except Exception as e:
        flash('Error making prediction')
        print(f"Prediction error: {e}")
        return redirect(url_for('index'))

    # Basic report information
    pneumonia_status = 'Positive' if has_pneumonia else 'Negative'
    report_info = f"Patient Name: {patient_name}\nAge: {patient_age}\nGender: {patient_gender}\nPneumonia: {pneumonia_status}\nPatient Symptoms: {symptoms}\n"

    # Generate report text
    prompt = (
        "Provide a detailed explanation of the different types of pneumonia, including their causes and symptoms. "
        "Outline the available treatments for each type, and explain what a diagnosis of pneumonia means for treatment options and prognosis. "
        "Ensure the response is informative and suitable for patients, doctors, and family members, while remaining concise."
    )

    report = "Pneumonia detected. Please consult a doctor immediately.\n\n" + generate_report_with_gpt_neo(prompt) if has_pneumonia else generate_report_with_gpt_neo(prompt)

    # Process location if available
    if user_location:
        latitude, longitude = get_location_from_address(user_location)
    else:
        latitude, longitude = get_current_location()

    # Find nearest hospital
    hospital_name, hospital_address, lat, lon = find_nearest_hospital(latitude, longitude)

    # Create a PDF report
    pdf_file_path = create_pdf_report(patient_name, patient_age, patient_gender, pneumonia_status, symptoms, report, hospital_name, hospital_address)

    # Render the report page with the generated report and map
    return render_template('report.html', 
                         report=report, 
                         pneumonia_status=pneumonia_status, 
                         patient_name=patient_name, 
                         patient_age=patient_age, 
                         patient_gender=patient_gender, 
                         symptoms=symptoms, 
                         hospital_name=hospital_name, 
                         hospital_address=hospital_address, 
                         latitude=lat, 
                         longitude=lon, 
                         report_id=filename)  # Ensure report_id is correct here

@app.route('/download_report/<report_id>')
def download_report(report_id):
    # Generate or fetch your PDF report based on the report_id
    pdf_path = generate_pdf_report(report_id)  # Your function to generate the PDF
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
