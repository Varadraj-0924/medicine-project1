import pytesseract
from PIL import Image
import pandas as pd
import re
import cv2
import numpy as np
from google.colab import files
import os

# Configure pytesseract path (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """
    Preprocess image for better OCR results
    """
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_text_from_image(image_path):
    """
    Extract text from medicine image using OCR
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Convert back to PIL Image for pytesseract
        pil_image = Image.fromarray(processed_image)
        
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(pil_image)
        
        # Clean the extracted text
        cleaned_text = clean_extracted_text(extracted_text)
        
        return cleaned_text
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return ""

def clean_extracted_text(text):
    """
    Clean and process extracted OCR text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

def find_medicine_in_database(extracted_text, df):
    """
    Find medicine in database based on extracted text
    """
    if not extracted_text:
        return None
    
    # Convert to lowercase for comparison
    extracted_lower = extracted_text.lower()
    
    # Search for medicine names in the extracted text
    for index, row in df.iterrows():
        medicine_name = str(row['Medicine Name']).lower()
        
        # Check if medicine name is in the extracted text
        if medicine_name in extracted_lower:
            return {
                'medicine_name': row['Medicine Name'],
                'uses': row['Uses'],
                'side_effects': row['Side_effects'],
                'composition': row.get('Composition', 'Not available'),
                'manufacturer': row.get('Manufacturer', 'Not available'),
                'confidence': 'High' if len(medicine_name) > 5 else 'Medium'
            }
    
    # If exact match not found, try partial matching
    for index, row in df.iterrows():
        medicine_name = str(row['Medicine Name']).lower()
        medicine_words = medicine_name.split()
        
        # Check if any word from medicine name is in extracted text
        for word in medicine_words:
            if len(word) > 3 and word in extracted_lower:
                return {
                    'medicine_name': row['Medicine Name'],
                    'uses': row['Uses'],
                    'side_effects': row['Side_effects'],
                    'composition': row.get('Composition', 'Not available'),
                    'manufacturer': row.get('Manufacturer', 'Not available'),
                    'confidence': 'Medium',
                    'matched_word': word
                }
    
    return None

def predict_medicine_info(extracted_text):
    """
    Predict medicine information based on extracted text
    """
    print(f"\n📝 Extracted Text: {extracted_text}")
    
    if not extracted_text:
        print("❌ No text extracted from image")
        return
    
    # Load medicine database
    try:
        df = pd.read_csv('Medicine_Details.csv')
        print(f"✅ Loaded {len(df)} medicines from database")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    # Find medicine in database
    medicine_info = find_medicine_in_database(extracted_text, df)
    
    if medicine_info:
        print(f"\n💊 Found Medicine: {medicine_info['medicine_name']}")
        print(f"🎯 Confidence: {medicine_info['confidence']}")
        if 'matched_word' in medicine_info:
            print(f"🔍 Matched Word: {medicine_info['matched_word']}")
        
        print(f"\n📋 Uses:")
        print(f"   {medicine_info['uses']}")
        
        print(f"\n⚠️ Side Effects:")
        print(f"   {medicine_info['side_effects']}")
        
        print(f"\n🧪 Composition:")
        print(f"   {medicine_info['composition']}")
        
        print(f"\n🏭 Manufacturer:")
        print(f"   {medicine_info['manufacturer']}")
        
    else:
        print("\n❌ Medicine not found in database")
        print("💡 Try uploading a clearer image or check the medicine name")

def main():
    """
    Main function to run OCR medicine detection
    """
    print("🔍 Medicine OCR Detection System")
    print("=" * 50)
    
    # Upload image
    print("📸 Please upload a medicine image...")
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No image uploaded")
        return
    
    # Get the uploaded image path
    image_path = list(uploaded.keys())[0]
    print(f"✅ Image uploaded: {image_path}")
    
    # Extract text using OCR
    print("\n🔍 Processing image with OCR...")
    extracted_text = extract_text_from_image(image_path)
    
    # Predict medicine information
    predict_medicine_info(extracted_text)
    
    # Clean up uploaded file
    try:
        os.remove(image_path)
        print(f"\n🧹 Cleaned up: {image_path}")
    except:
        pass

def test_ocr_with_sample():
    """
    Test OCR functionality with sample text
    """
    print("🧪 Testing OCR functionality...")
    
    # Sample medicine names to test
    sample_medicines = [
        "Paracetamol 500mg",
        "Aspirin 100mg",
        "Ibuprofen 400mg",
        "Omeprazole 20mg"
    ]
    
    # Load database
    try:
        df = pd.read_csv('Medicine_Details.csv')
        print(f"✅ Loaded {len(df)} medicines from database")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    # Test each sample
    for medicine in sample_medicines:
        print(f"\n🔍 Testing: {medicine}")
        medicine_info = find_medicine_in_database(medicine, df)
        
        if medicine_info:
            print(f"✅ Found: {medicine_info['medicine_name']}")
            print(f"   Uses: {medicine_info['uses'][:100]}...")
        else:
            print("❌ Not found in database")

if __name__ == "__main__":
    # Run the main OCR detection
    main()
    
    # Uncomment to test with sample data
    # test_ocr_with_sample()