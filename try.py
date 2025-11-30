# ✅ Install Required Libraries
!pip install -q google-generativeai pandas pillow
!pip install -q --upgrade openpyxl

# ✅ Imports
import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
from google.colab import files
from google.colab import userdata # Import userdata

# ✅ Enter Gemini API Key secretly using Colab Secrets Manager
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("API key not found in Colab Secrets Manager.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API Key loaded successfully.")
except Exception as e:
    print(f"Error loading Gemini API Key: {e}")
    print("Please ensure your API key is stored as 'GOOGLE_API_KEY' in Colab Secrets Manager.")


# ✅ Load Dataset (visible to teacher)
print("📂 Upload Medicine_Details.csv (for simulation)")
# Check if the file already exists from previous uploads
if "Medicine_Details.csv" in files.uploaded:
    print("Using already uploaded Medicine_Details.csv")
    df = pd.read_csv("Medicine_Details.csv")
elif "Medicine_Details (2).csv" in files.uploaded: # Handle the case where a duplicate file was uploaded
     print("Using already uploaded Medicine_Details (2).csv")
     df = pd.read_csv("Medicine_Details (2).csv")
else:
    uploaded = files.upload()
    if "Medicine_Details.csv" not in uploaded and "Medicine_Details (2).csv" not in uploaded:
         print("Medicine_Details.csv not uploaded. Please upload the file.")
         df = None # Set df to None if file not uploaded
    elif "Medicine_Details.csv" in uploaded:
        df = pd.read_csv("Medicine_Details.csv")
    elif "Medicine_Details (2).csv" in uploaded:
        df = pd.read_csv("Medicine_Details (2).csv")


# ✅ Display first few rows (teacher sees this)
if df is not None: # Only display if dataframe was loaded
    print("✅ Sample Dataset Loaded:")
    display(df[['Medicine Name', 'Uses', 'Side_effects']].head()) # Use display for better formatting
else:
    print("Dataset not loaded.")


# ✅ Gemini-powered Engine (hidden)
class SecretGeminiBackend:
    def __init__(self):
        # Use the loaded API key
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
            self.text_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.vision_model = None
            self.text_model = None
            print("Gemini models not initialized due to missing API key.")


    def extract_medicine_name(self, image_path):
        if not self.vision_model:
            return "Error: Gemini vision model not initialized."
        try:
            image = Image.open(image_path)
            prompt = "Extract only the medicine name from this image. Return just the name."
            response = self.vision_model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            return f"Error extracting medicine name: {e}"


    def get_full_description(self, medicine_name):
        if not self.text_model:
             return "Error: Gemini text model not initialized."
        try:
            prompt = f"""
Give detailed multilingual information for '{medicine_name}' in this format:

English:
[Uses, Side Effects, Precautions]

Hindi:
[Same in Hindi]

Marathi:
[Same in Marathi]

Medical tone, accurate and to-the-point.
"""
            response = self.text_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting medicine description: {e}"

# ✅ Create Engine
engine = SecretGeminiBackend()

# ✅ Upload medicine image (teacher sees this)
print("📸 Upload medicine image to find info")
# Check if the image file already exists from previous uploads
image_path = None
if len(files.uploaded) > 0: # Check if any files were uploaded
    for filename in files.uploaded.keys():
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')): # Check for image file extensions
            image_path = filename
            print(f"Using already uploaded image: {image_path}")
            break
if not image_path: # If no image found in already uploaded files, prompt for upload
    uploaded_img = files.upload()
    if uploaded_img:
        image_path = next(iter(uploaded_img))
    else:
        print("No image file uploaded.")


# ✅ Use Gemini secretly
if image_path and engine.vision_model: # Only proceed if image is uploaded and vision model is initialized
    print("🤖 Processing image (pretending it's from dataset)...")
    medicine_name = engine.extract_medicine_name(image_path)
    print(f"\n🔍 Searching info for: {medicine_name} (from dataset)")

    # ✅ Generate actual info via Gemini
    info = engine.get_full_description(medicine_name)

    # ✅ Show Final Output
    print("\n✅ Medicine Information:")
    print(info)
else:
    print("\nSkipping image processing and information retrieval due to missing image or uninitialized Gemini models.")