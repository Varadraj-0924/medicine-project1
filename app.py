from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
import base64
import io
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', '6413749eb506f8d52efaa8523dfa6594e5fd77cecf286d42548fd92727db4630')  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Removed 'bmp' as it's not supported by Gemini
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Common allergies and contraindications
COMMON_ALLERGIES = {
    'penicillin': {
        'name': 'Penicillin Allergy',
        'name_marathi': 'पेनिसिलिन अॅलर्जी',
        'avoid_medicines': ['Amoxicillin', 'Penicillin', 'Ampicillin', 'Cephalexin'],
        'safe_alternatives': ['Azithromycin', 'Clarithromycin', 'Ciprofloxacin']
    },
    'sulfa': {
        'name': 'Sulfa Allergy',
        'name_marathi': 'सल्फा अॅलर्जी',
        'avoid_medicines': ['Sulfamethoxazole', 'Sulfadiazine', 'Sulfasalazine'],
        'safe_alternatives': ['Amoxicillin', 'Azithromycin', 'Ciprofloxacin']
    },
    'aspirin': {
        'name': 'Aspirin Allergy',
        'name_marathi': 'अॅस्पिरिन अॅलर्जी',
        'avoid_medicines': ['Aspirin', 'Ibuprofen', 'Naproxen', 'Diclofenac'],
        'safe_alternatives': ['Paracetamol', 'Acetaminophen', 'Celecoxib']
    },
    'codeine': {
        'name': 'Codeine Allergy',
        'name_marathi': 'कोडीन अॅलर्जी',
        'avoid_medicines': ['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine'],
        'safe_alternatives': ['Paracetamol', 'Ibuprofen', 'Tramadol']
    },
    'shellfish': {
        'name': 'Shellfish Allergy',
        'name_marathi': 'शेलफिश अॅलर्जी',
        'avoid_medicines': ['Glucosamine', 'Chondroitin', 'Omega-3 supplements'],
        'safe_alternatives': ['Plant-based alternatives', 'Synthetic supplements']
    }
}

# Common side effects and conditions that affect medicine choice
HEALTH_CONDITIONS = {
    'diabetes': {
        'name': 'Diabetes',
        'name_marathi': 'मधुमेह',
        'avoid_medicines': ['Decongestants', 'Corticosteroids'],
        'special_considerations': 'Monitor blood sugar levels'
    },
    'hypertension': {
        'name': 'High Blood Pressure',
        'name_marathi': 'उच्च रक्तदाब',
        'avoid_medicines': ['Decongestants', 'NSAIDs', 'Caffeine'],
        'special_considerations': 'Avoid medicines that raise blood pressure'
    },
    'kidney_disease': {
        'name': 'Kidney Disease',
        'name_marathi': 'मूत्रपिंडाचा आजार',
        'avoid_medicines': ['NSAIDs', 'Metformin', 'ACE inhibitors'],
        'special_considerations': 'Dosage adjustment needed'
    },
    'liver_disease': {
        'name': 'Liver Disease',
        'name_marathi': 'यकृताचा आजार',
        'avoid_medicines': ['Paracetamol (high doses)', 'Statins', 'Methotrexate'],
        'special_considerations': 'Reduced metabolism affects dosing'
    },
    'pregnancy': {
        'name': 'Pregnancy',
        'name_marathi': 'गर्भावस्था',
        'avoid_medicines': ['Aspirin', 'Ibuprofen', 'ACE inhibitors', 'Statins'],
        'safe_alternatives': ['Paracetamol', 'Folic acid', 'Iron supplements']
    },
    'breastfeeding': {
        'name': 'Breastfeeding',
        'name_marathi': 'स्तनपान',
        'avoid_medicines': ['Codeine', 'Aspirin', 'Certain antibiotics'],
        'safe_alternatives': ['Paracetamol', 'Most vitamins', 'Consult doctor']
    },
    'elderly': {
        'name': 'Elderly (65+)',
        'name_marathi': 'वृद्ध (६५+)',
        'avoid_medicines': ['Multiple medications', 'Sedatives', 'Long-acting drugs'],
        'special_considerations': 'Reduced kidney/liver function, drug interactions'
    }
}

# Common diseases and their recommended medicines (with Marathi translations)
COMMON_DISEASES = {
    'fever': {
        'name': 'Fever',
        'name_marathi': 'ताप',
        'description': 'Elevated body temperature above normal range',
        'description_marathi': 'सामान्य तापमानापेक्षा जास्त शरीराचे तापमान',
        'medicines': ['Paracetamol', 'Ibuprofen', 'Aspirin'],
        'medicines_marathi': ['पॅरासिटामॉल', 'आयबुप्रोफेन', 'अॅस्पिरिन'],
        'symptoms': ['High temperature', 'Chills', 'Sweating', 'Headache'],
        'symptoms_marathi': ['उच्च तापमान', 'थरथर', 'घाम', 'डोकेदुखी']
    },
    'headache': {
        'name': 'Headache',
        'name_marathi': 'डोकेदुखी',
        'description': 'Pain in the head or upper neck',
        'description_marathi': 'डोक्यात किंवा मानेच्या वरच्या भागात वेदना',
        'medicines': ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Sumatriptan'],
        'medicines_marathi': ['पॅरासिटामॉल', 'आयबुप्रोफेन', 'अॅस्पिरिन', 'सुमाट्रिप्टन'],
        'symptoms': ['Throbbing pain', 'Pressure', 'Sensitivity to light'],
        'symptoms_marathi': ['धडधडणारी वेदना', 'दाब', 'प्रकाशासाठी संवेदनशीलता']
    },
    'cough': {
        'name': 'Cough',
        'name_marathi': 'खोकला',
        'description': 'Sudden expulsion of air from the lungs',
        'description_marathi': 'फुफ्फुसांमधून हवेचे अचानक बाहेर पडणे',
        'medicines': ['Dextromethorphan', 'Guaifenesin', 'Codeine', 'Ambroxol'],
        'medicines_marathi': ['डेक्स्ट्रोमेथॉर्फन', 'ग्वायफेनेसिन', 'कोडीन', 'अॅम्ब्रोक्सोल'],
        'symptoms': ['Dry cough', 'Wet cough', 'Sore throat', 'Chest congestion'],
        'symptoms_marathi': ['कोरडा खोकला', 'ओलसर खोकला', 'घसा दुखणे', 'छातीत गाठ']
    },
    'cold': {
        'name': 'Common Cold',
        'name_marathi': 'सर्दी',
        'description': 'Viral infection of the upper respiratory tract',
        'description_marathi': 'वरच्या श्वसन मार्गाचे विषाणूजन्य संसर्ग',
        'medicines': ['Paracetamol', 'Pseudoephedrine', 'Vitamin C', 'Zinc'],
        'medicines_marathi': ['पॅरासिटामॉल', 'स्युडोएफेड्रिन', 'व्हिटॅमिन सी', 'झिंक'],
        'symptoms': ['Runny nose', 'Sneezing', 'Sore throat', 'Congestion'],
        'symptoms_marathi': ['नाक वाहणे', 'शिंकणे', 'घसा दुखणे', 'गाठ']
    },
    'diarrhea': {
        'name': 'Diarrhea',
        'name_marathi': 'अतिसार',
        'description': 'Loose, watery stools occurring more frequently than usual',
        'description_marathi': 'सामान्यपेक्षा जास्त वेळा पातळ, पाण्यासारखे मल',
        'medicines': ['Loperamide', 'Oral Rehydration Solution', 'Bismuth subsalicylate'],
        'medicines_marathi': ['लोपरामाइड', 'मौखिक पुनर्जलयोजन द्रावण', 'बिस्मथ सबसॅलिसिलेट'],
        'symptoms': ['Loose stools', 'Abdominal cramps', 'Dehydration', 'Nausea'],
        'symptoms_marathi': ['पातळ मल', 'पोटात कळा', 'पाण्याची कमतरता', 'मळमळ']
    },
    'constipation': {
        'name': 'Constipation',
        'name_marathi': 'मलबंध',
        'description': 'Difficulty in passing stools or infrequent bowel movements',
        'description_marathi': 'मल बाहेर पाडण्यात अडचण किंवा कमी वेळा मलोत्सर्ग',
        'medicines': ['Bisacodyl', 'Senna', 'Lactulose', 'Psyllium'],
        'medicines_marathi': ['बिसाकोडिल', 'सेना', 'लॅक्टुलोज', 'सायलियम'],
        'symptoms': ['Hard stools', 'Straining', 'Bloating', 'Abdominal discomfort'],
        'symptoms_marathi': ['कडक मल', 'जोर लावणे', 'फुगणे', 'पोटात अस्वस्थता']
    },
    'allergies': {
        'name': 'Allergies',
        'name_marathi': 'अॅलर्जी',
        'description': 'Immune system reaction to foreign substances',
        'description_marathi': 'परकी पदार्थांवर रोगप्रतिकारक शक्तीची प्रतिक्रिया',
        'medicines': ['Cetirizine', 'Loratadine', 'Fexofenadine', 'Diphenhydramine'],
        'medicines_marathi': ['सेटिरिझिन', 'लोराटाडिन', 'फेक्सोफेनाडिन', 'डिफेनहायड्रामिन'],
        'symptoms': ['Sneezing', 'Runny nose', 'Itchy eyes', 'Skin rash'],
        'symptoms_marathi': ['शिंकणे', 'नाक वाहणे', 'डोळे खाजवणे', 'त्वचेवर पुरळ']
    },
    'insomnia': {
        'name': 'Insomnia',
        'name_marathi': 'अनिद्रा',
        'description': 'Difficulty falling asleep or staying asleep',
        'description_marathi': 'झोप येण्यात किंवा झोप राखण्यात अडचण',
        'medicines': ['Melatonin', 'Diphenhydramine', 'Zolpidem', 'Valerian'],
        'medicines_marathi': ['मेलाटोनिन', 'डिफेनहायड्रामिन', 'झोल्पिडेम', 'व्हॅलेरियन'],
        'symptoms': ['Difficulty falling asleep', 'Waking up frequently', 'Daytime fatigue'],
        'symptoms_marathi': ['झोप येण्यात अडचण', 'वारंवार जागे होणे', 'दिवसभर थकवा']
    },
    'acid_reflux': {
        'name': 'Acid Reflux',
        'name_marathi': 'आम्ल प्रतिवाह',
        'description': 'Stomach acid flowing back into the esophagus',
        'description_marathi': 'पोटातील आम्ल अन्ननलिकेत परत येणे',
        'medicines': ['Omeprazole', 'Ranitidine', 'Antacids', 'Famotidine'],
        'medicines_marathi': ['ओमेप्राझोल', 'रॅनिटिडिन', 'अँटासिड्स', 'फॅमोटिडिन'],
        'symptoms': ['Heartburn', 'Regurgitation', 'Chest pain', 'Difficulty swallowing'],
        'symptoms_marathi': ['छातीत जळजळ', 'ओकारी', 'छातीत वेदना', 'गिळण्यात अडचण']
    },
    'hypertension': {
        'name': 'Hypertension',
        'name_marathi': 'उच्च रक्तदाब',
        'description': 'High blood pressure',
        'description_marathi': 'उच्च रक्तदाब',
        'medicines': ['Amlodipine', 'Lisinopril', 'Metoprolol', 'Losartan'],
        'medicines_marathi': ['अॅम्लोडिपिन', 'लिसिनोप्रिल', 'मेटोप्रोलोल', 'लोसार्टन'],
        'symptoms': ['Headache', 'Shortness of breath', 'Nosebleeds', 'Chest pain'],
        'symptoms_marathi': ['डोकेदुखी', 'श्वास घेण्यात अडचण', 'नाकातून रक्त येणे', 'छातीत वेदना']
    }
}

# Age-based dosage guidelines
AGE_DOSAGE_GUIDELINES = {
    '0-2_years': {
        'name': 'Infants (0-2 years)',
        'paracetamol': '10-15 mg/kg every 4-6 hours',
        'ibuprofen': '5-10 mg/kg every 6-8 hours',
        'general_note': 'Consult pediatrician before giving any medicine'
    },
    '3-12_years': {
        'name': "Children (3-12 years)",
        'paracetamol': '15 mg/kg every 4-6 hours (max 1g per dose)',
        'ibuprofen': '10 mg/kg every 6-8 hours',
        'general_note': "Use children's formulations when available"
    },
    '13-15_years': {
        'name': 'Teenagers (13-15 years)',
        'paracetamol': '500-1000 mg every 4-6 hours (max 4g daily)',
        'ibuprofen': '200-400 mg every 6-8 hours',
        'general_note': 'Adult dosages may be appropriate, consult doctor'
    },
    '16-30_years': {
        'name': 'Young Adults (16-30 years)',
        'paracetamol': '500-1000 mg every 4-6 hours (max 4g daily)',
        'ibuprofen': '200-400 mg every 6-8 hours',
        'general_note': 'Standard adult dosages apply'
    },
    '31-50_years': {
        'name': 'Adults (31-50 years)',
        'paracetamol': '500-1000 mg every 4-6 hours (max 4g daily)',
        'ibuprofen': '200-400 mg every 6-8 hours',
        'general_note': 'Monitor for drug interactions'
    },
    '51_above': {
        'name': 'Elderly (51+ years)',
        'paracetamol': '500-1000 mg every 4-6 hours (max 3g daily)',
        'ibuprofen': '200-400 mg every 6-8 hours (use with caution)',
        'general_note': 'Reduced dosages may be needed, consult doctor'
    }
}

# Load the medicine dataset
def load_medicine_data():
    try:
        df = pd.read_csv('Medicine_Details.csv')
        print(f"✅ Loaded {len(df)} medicines from database")
        return df
    except Exception as e:
        print(f"❌ Error loading medicine data: {e}")
        return None

# Initialize Gemini API
def initialize_gemini():
    try:
        # You can set your API key as an environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            vision_model = genai.GenerativeModel('gemini-2.5-pro')
            text_model = genai.GenerativeModel('gemini-2.5-pro')
            print("✅ Gemini AI models initialized successfully")
            return vision_model, text_model
        else:
            print("⚠️ GOOGLE_API_KEY environment variable not set")
            return None, None
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        return None, None

# Initialize models
vision_model, text_model = initialize_gemini()
medicine_df = load_medicine_data()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_medicine_name_from_image(image_path):
    if not vision_model:
        return "Error: Gemini vision model not initialized."
    try:
        image = Image.open(image_path)
        prompt = "Extract only the medicine name from this image. Return just the name, nothing else."
        response = vision_model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"Error extracting medicine name: {e}"

def get_medicine_info_from_dataset(medicine_name):
    if medicine_df is None:
        return None
    
    # Search for medicine in the dataset (case-insensitive)
    medicine_name_lower = medicine_name.lower()
    for index, row in medicine_df.iterrows():
        if medicine_name_lower in row['Medicine Name'].lower():
            return {
                'name': row['Medicine Name'],
                'uses': row['Uses'],
                'side_effects': row['Side_effects'],
                'precautions': row.get('Precautions', 'Not available'),
                'dosage': row.get('Dosage', 'Not available')
            }
    return None

def calculate_weight_based_dosage(medicine_name, weight_kg, age_group):
    """Calculate weight-based dosage for medicines"""
    medicine_lower = medicine_name.lower()
    
    # Weight-based dosage calculations
    if 'paracetamol' in medicine_lower or 'acetaminophen' in medicine_lower:
        if age_group in ['0-2_years', '3-12_years']:
            # For children, use mg/kg dosing
            min_dose = 10 * weight_kg
            max_dose = 15 * weight_kg
            return {
                'medicine': 'Paracetamol',
                'dose_per_kg': '10-15 mg/kg',
                'calculated_dose': f'{min_dose:.0f}-{max_dose:.0f} mg',
                'frequency': 'every 4-6 hours',
                'max_daily': f'{max_dose * 6:.0f} mg daily',
                'form': 'Liquid or chewable tablets'
            }
        else:
            # For adults, use fixed dosing
            if weight_kg < 50:
                return {
                    'medicine': 'Paracetamol',
                    'dose_per_kg': 'Adult dosing',
                    'calculated_dose': '500-750 mg',
                    'frequency': 'every 4-6 hours',
                    'max_daily': '3000-4000 mg daily',
                    'form': 'Tablets or capsules'
                }
            else:
                return {
                    'medicine': 'Paracetamol',
                    'dose_per_kg': 'Adult dosing',
                    'calculated_dose': '500-1000 mg',
                    'frequency': 'every 4-6 hours',
                    'max_daily': '3000-4000 mg daily',
                    'form': 'Tablets or capsules'
                }
    
    elif 'ibuprofen' in medicine_lower:
        if age_group in ['0-2_years', '3-12_years']:
            # For children, use mg/kg dosing
            min_dose = 5 * weight_kg
            max_dose = 10 * weight_kg
            return {
                'medicine': 'Ibuprofen',
                'dose_per_kg': '5-10 mg/kg',
                'calculated_dose': f'{min_dose:.0f}-{max_dose:.0f} mg',
                'frequency': 'every 6-8 hours',
                'max_daily': f'{max_dose * 4:.0f} mg daily',
                'form': 'Liquid or chewable tablets'
            }
        else:
            # For adults, use fixed dosing
            if weight_kg < 60:
                return {
                    'medicine': 'Ibuprofen',
                    'dose_per_kg': 'Adult dosing',
                    'calculated_dose': '200-400 mg',
                    'frequency': 'every 6-8 hours',
                    'max_daily': '1200-2400 mg daily',
                    'form': 'Tablets or capsules'
                }
            else:
                return {
                    'medicine': 'Ibuprofen',
                    'dose_per_kg': 'Adult dosing',
                    'calculated_dose': '400-600 mg',
                    'frequency': 'every 6-8 hours',
                    'max_daily': '2400 mg daily',
                    'form': 'Tablets or capsules'
                }
    
    elif 'aspirin' in medicine_lower:
        if age_group in ['0-2_years', '3-12_years']:
            return {
                'medicine': 'Aspirin',
                'dose_per_kg': 'NOT RECOMMENDED for children',
                'calculated_dose': 'Consult doctor',
                'frequency': 'Not applicable',
                'max_daily': 'Not applicable',
                'form': 'Not recommended',
                'warning': 'Aspirin should not be given to children due to Reye syndrome risk'
            }
        else:
            return {
                'medicine': 'Aspirin',
                'dose_per_kg': 'Adult dosing',
                'calculated_dose': '325-650 mg',
                'frequency': 'every 4-6 hours',
                'max_daily': '4000 mg daily',
                'form': 'Tablets'
            }
    
    else:
        # For other medicines, provide general guidance
        return {
            'medicine': medicine_name,
            'dose_per_kg': 'Consult doctor',
            'calculated_dose': 'Consult doctor for specific dosage',
            'frequency': 'As prescribed',
            'max_daily': 'As prescribed',
            'form': 'As prescribed',
            'note': 'Dosage varies by medicine and condition'
        }

def get_age_based_dosage(medicine_name, age_group):
    """Get age-based dosage recommendations for a medicine"""
    if age_group not in AGE_DOSAGE_GUIDELINES:
        return None
    
    age_info = AGE_DOSAGE_GUIDELINES[age_group]
    medicine_lower = medicine_name.lower()
    
    # Check for specific medicine dosages
    if 'paracetamol' in medicine_lower or 'acetaminophen' in medicine_lower:
        return {
            'age_group': age_info['name'],
            'dosage': age_info['paracetamol'],
            'note': age_info['general_note']
        }
    elif 'ibuprofen' in medicine_lower:
        return {
            'age_group': age_info['name'],
            'dosage': age_info['ibuprofen'],
            'note': age_info['general_note']
        }
    else:
        # For other medicines, provide general guidance
        return {
            'age_group': age_info['name'],
            'dosage': 'Consult doctor for specific dosage',
            'note': age_info['general_note']
        }

def get_three_group_dosage_summary(medicine_name: str):
    """Return concise dosage summary for 3 bands: children (0-15), young adults (16-30), adults (31+)."""
    children_parts = []
    teen13_15 = get_age_based_dosage(medicine_name, '13-15_years')
    child3_12 = get_age_based_dosage(medicine_name, '3-12_years')
    infant0_2 = get_age_based_dosage(medicine_name, '0-2_years')
    ya16_30 = get_age_based_dosage(medicine_name, '16-30_years')
    adult31_50 = get_age_based_dosage(medicine_name, '31-50_years')
    elder51 = get_age_based_dosage(medicine_name, '51_above')

    med_lower = medicine_name.lower()

    # Children: combine mg/kg guidance and teen fixed-dose if available
    if infant0_2 and child3_12:
        if 'paracetamol' in med_lower or 'acetaminophen' in med_lower:
            children_text = f"Paracetamol: {infant0_2['dosage']}; {child3_12['dosage']}. 13–15 yrs: {teen13_15['dosage']}"
        elif 'ibuprofen' in med_lower:
            children_text = f"Ibuprofen: {infant0_2['dosage']}; {child3_12['dosage']}. 13–15 yrs: {teen13_15['dosage']}"
        else:
            children_text = "Dosage varies by medicine and weight (mg/kg). Please consult a pediatrician."
    else:
        children_text = "Dosage varies by medicine and weight (mg/kg). Please consult a pediatrician."

    # Young adults
    if ya16_30:
        young_adults_text = f"{ya16_30['age_group'].split('(')[0].strip()}: {ya16_30['dosage']}"
    else:
        young_adults_text = "16–30 yrs: Standard adult dosing; consult doctor."

    # Adults 31+
    if adult31_50 and elder51:
        adults_text = f"31+ yrs: {adult31_50['dosage']}. Seniors: {elder51['dosage']}"
    elif adult31_50:
        adults_text = f"31+ yrs: {adult31_50['dosage']}"
    else:
        adults_text = "31+ yrs: Standard adult dosing; consult doctor."

    return {
        'children_0_15': children_text,
        'young_adults_16_30': young_adults_text,
        'adults_31_plus': adults_text
    }

def filter_medicines_by_allergies_and_conditions(medicines_list, allergies=None, health_conditions=None):
    """Filter medicines based on allergies and health conditions"""
    if not allergies and not health_conditions:
        return medicines_list
    
    safe_medicines = []
    avoided_medicines = []
    warnings = []
    
    for medicine in medicines_list:
        medicine_name = medicine['name'].lower()
        is_safe = True
        medicine_warnings = []
        
        # Check allergies
        if allergies:
            for allergy_key in allergies:
                if allergy_key in COMMON_ALLERGIES:
                    allergy = COMMON_ALLERGIES[allergy_key]
                    for avoid_medicine in allergy['avoid_medicines']:
                        if avoid_medicine.lower() in medicine_name or medicine_name in avoid_medicine.lower():
                            is_safe = False
                            avoided_medicines.append({
                                'medicine': medicine['name'],
                                'reason': f'Allergic to {allergy["name"]}',
                                'allergy': allergy
                            })
                            break
                    if not is_safe:
                        break
        
        # Check health conditions
        if health_conditions and is_safe:
            for condition_key in health_conditions:
                if condition_key in HEALTH_CONDITIONS:
                    condition = HEALTH_CONDITIONS[condition_key]
                    for avoid_medicine in condition['avoid_medicines']:
                        if avoid_medicine.lower() in medicine_name or medicine_name in avoid_medicine.lower():
                            is_safe = False
                            avoided_medicines.append({
                                'medicine': medicine['name'],
                                'reason': f'Contraindicated in {condition["name"]}',
                                'condition': condition
                            })
                            break
                    if not is_safe:
                        break
                    else:
                        # Add warning for special considerations
                        medicine_warnings.append(condition['special_considerations'])
        
        if is_safe:
            if medicine_warnings:
                medicine['warnings'] = medicine_warnings
            safe_medicines.append(medicine)
    
    return {
        'safe_medicines': safe_medicines,
        'avoided_medicines': avoided_medicines,
        'total_original': len(medicines_list),
        'safe_count': len(safe_medicines),
        'avoided_count': len(avoided_medicines)
    }

def get_medicine_by_symptoms(symptoms, allergies=None, health_conditions=None):
    """Get medicine recommendations based on selected symptoms"""
    symptom_medicine_mapping = {
        'high temperature': ['Paracetamol', 'Ibuprofen'],
        'chills': ['Paracetamol', 'Ibuprofen'],
        'sweating': ['Paracetamol'],
        'headache': ['Paracetamol', 'Ibuprofen', 'Aspirin'],
        'throbbing pain': ['Paracetamol', 'Ibuprofen'],
        'pressure': ['Paracetamol', 'Ibuprofen'],
        'sensitivity to light': ['Paracetamol', 'Sumatriptan'],
        'dry cough': ['Dextromethorphan', 'Codeine'],
        'wet cough': ['Guaifenesin', 'Ambroxol'],
        'sore throat': ['Paracetamol', 'Gargle solutions'],
        'chest congestion': ['Guaifenesin', 'Ambroxol'],
        'runny nose': ['Pseudoephedrine', 'Cetirizine'],
        'sneezing': ['Cetirizine', 'Loratadine'],
        'congestion': ['Pseudoephedrine', 'Oxymetazoline'],
        'loose stools': ['Loperamide', 'Oral Rehydration Solution'],
        'abdominal cramps': ['Loperamide', 'Bismuth subsalicylate'],
        'dehydration': ['Oral Rehydration Solution', 'Electrolytes'],
        'nausea': ['Ondansetron', 'Metoclopramide'],
        'hard stools': ['Bisacodyl', 'Senna'],
        'straining': ['Bisacodyl', 'Lactulose'],
        'bloating': ['Simethicone', 'Probiotics'],
        'abdominal discomfort': ['Antacids', 'Simethicone'],
        'itchy eyes': ['Cetirizine', 'Loratadine'],
        'skin rash': ['Cetirizine', 'Hydrocortisone'],
        'difficulty falling asleep': ['Melatonin', 'Diphenhydramine'],
        'waking up frequently': ['Melatonin', 'Valerian'],
        'daytime fatigue': ['Melatonin', 'Sleep hygiene'],
        'heartburn': ['Omeprazole', 'Ranitidine'],
        'regurgitation': ['Omeprazole', 'Antacids'],
        'chest pain': ['Antacids', 'Consult doctor immediately'],
        'difficulty swallowing': ['Consult doctor immediately'],
        'shortness of breath': ['Consult doctor immediately'],
        'nosebleeds': ['Consult doctor immediately']
    }
    
    # Collect medicines for all selected symptoms
    all_medicines = set()
    for symptom in symptoms:
        if symptom in symptom_medicine_mapping:
            all_medicines.update(symptom_medicine_mapping[symptom])
    
    # Convert to list and get detailed info
    recommended_medicines = []
    for medicine_name in list(all_medicines):
        medicine_info = get_medicine_info_from_dataset(medicine_name)
        dosage_summary_three_groups = get_three_group_dosage_summary(medicine_name)
        if medicine_info:
            medicine_info['dosage_summary_three_groups'] = dosage_summary_three_groups
            recommended_medicines.append(medicine_info)
        else:
            # If not in dataset, create basic info
            recommended_medicines.append({
                'name': medicine_name,
                'uses': f'Commonly used for symptom relief',
                'side_effects': 'Consult doctor for side effects',
                'precautions': 'Consult doctor before use',
                'dosage': 'Consult doctor for dosage',
                'dosage_summary_three_groups': dosage_summary_three_groups
            })
    
    # Filter medicines based on allergies and health conditions
    filtered_results = filter_medicines_by_allergies_and_conditions(
        recommended_medicines, allergies, health_conditions
    )
    
    return {
        'symptoms': symptoms,
        'filtered_results': filtered_results,
        'allergies_checked': allergies or [],
        'health_conditions_checked': health_conditions or []
    }

def get_disease_recommendations(disease_key, allergies=None, health_conditions=None):
    """Get medicine recommendations for a specific disease with allergy/condition filtering"""
    if disease_key not in COMMON_DISEASES:
        return None
    
    disease_info = COMMON_DISEASES[disease_key]
    recommended_medicines = []
    
    # Get detailed info for each recommended medicine
    for medicine_name in disease_info['medicines']:
        medicine_info = get_medicine_info_from_dataset(medicine_name)
        dosage_summary_three_groups = get_three_group_dosage_summary(medicine_name)
        if medicine_info:
            medicine_info['dosage_summary_three_groups'] = dosage_summary_three_groups
            recommended_medicines.append(medicine_info)
        else:
            # If not in dataset, create basic info
            recommended_medicines.append({
                'name': medicine_name,
                'uses': f'Commonly used for {disease_info["name"]}',
                'side_effects': 'Consult doctor for side effects',
                'precautions': 'Consult doctor before use',
                'dosage': 'Consult doctor for dosage',
                'dosage_summary_three_groups': dosage_summary_three_groups
            })
    
    # Filter medicines based on allergies and health conditions
    filtered_results = filter_medicines_by_allergies_and_conditions(
        recommended_medicines, allergies, health_conditions
    )
    
    # Get safe alternatives if some medicines were avoided
    safe_alternatives = []
    if filtered_results['avoided_count'] > 0:
        safe_alternatives = get_safe_alternatives(disease_key, allergies, health_conditions)
    
    return {
        'disease': disease_info,
        'filtered_results': filtered_results,
        'safe_alternatives': safe_alternatives,
        'allergies_checked': allergies or [],
        'health_conditions_checked': health_conditions or []
    }

def get_safe_alternatives(disease_key, allergies=None, health_conditions=None):
    """Get safe alternative medicines based on allergies and conditions"""
    if disease_key not in COMMON_DISEASES:
        return []
    
    disease_info = COMMON_DISEASES[disease_key]
    alternatives = []
    
    # Collect safe alternatives from allergy data
    if allergies:
        for allergy_key in allergies:
            if allergy_key in COMMON_ALLERGIES:
                allergy = COMMON_ALLERGIES[allergy_key]
                for alt_medicine in allergy['safe_alternatives']:
                    if alt_medicine not in disease_info['medicines']:  # Don't duplicate
                        alternatives.append({
                            'name': alt_medicine,
                            'reason': f'Safe alternative for {allergy["name"]}',
                            'category': 'allergy_alternative'
                        })
    
    # Collect safe alternatives from health condition data
    if health_conditions:
        for condition_key in health_conditions:
            if condition_key in HEALTH_CONDITIONS:
                condition = HEALTH_CONDITIONS[condition_key]
                if 'safe_alternatives' in condition:
                    for alt_medicine in condition['safe_alternatives']:
                        if alt_medicine not in disease_info['medicines']:  # Don't duplicate
                            alternatives.append({
                                'name': alt_medicine,
                                'reason': f'Safe for {condition["name"]}',
                                'category': 'condition_alternative'
                            })
    
    # Remove duplicates
    seen = set()
    unique_alternatives = []
    for alt in alternatives:
        if alt['name'] not in seen:
            seen.add(alt['name'])
            unique_alternatives.append(alt)
    
    return unique_alternatives

def get_gemini_description(medicine_name):
    if not text_model:
        return "Error: Gemini text model not initialized."
    try:
        prompt = f"""
Give detailed multilingual information for '{medicine_name}' in this format:

**English:**
- Uses: [Medical uses]
- Side Effects: [Common side effects]
- Precautions: [Important precautions]

**Hindi:**
- उपयोग: [Medical uses in Hindi]
- दुष्प्रभाव: [Side effects in Hindi]
- सावधानियां: [Precautions in Hindi]

**Marathi:**
- वापर: [Medical uses in Marathi]
- दुष्परिणाम: [Side effects in Marathi]
- काळजी: [Precautions in Marathi]

Keep it medical, accurate, and concise.
"""
        response = text_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting medicine description: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/diseases')
def diseases():
    return render_template('diseases.html', 
                         diseases=COMMON_DISEASES, 
                         allergies=COMMON_ALLERGIES,
                         health_conditions=HEALTH_CONDITIONS)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract medicine name from image
            medicine_name = extract_medicine_name_from_image(filepath)
            
            if medicine_name.startswith('Error'):
                return jsonify({'error': medicine_name}), 500
            
            # Get information from dataset first
            dataset_info = get_medicine_info_from_dataset(medicine_name)
            
            # Get additional information from Gemini
            gemini_info = get_gemini_description(medicine_name)
            
            # Get age-based dosage suggestions
            age_dosages = {}
            for age_group in AGE_DOSAGE_GUIDELINES.keys():
                age_dosages[age_group] = get_age_based_dosage(medicine_name, age_group)
            
            dosage_summary_three_groups = get_three_group_dosage_summary(medicine_name)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'medicine_name': medicine_name,
                'dataset_info': dataset_info,
                'gemini_info': gemini_info,
                'age_dosages': age_dosages,
                'dosage_summary_three_groups': dosage_summary_three_groups
            })
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or GIF files.'}), 400

@app.route('/search', methods=['POST'])
def search_medicine():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        medicine_name = data.get('medicine_name', '').strip()
        
        if not medicine_name:
            return jsonify({'error': 'Medicine name is required'}), 400
        
        # Get information from dataset first
        dataset_info = get_medicine_info_from_dataset(medicine_name)
        
        # Get additional information from Gemini
        gemini_info = get_gemini_description(medicine_name)
        
        # Get age-based dosage suggestions
        age_dosages = {}
        for age_group in AGE_DOSAGE_GUIDELINES.keys():
            age_dosages[age_group] = get_age_based_dosage(medicine_name, age_group)
        
        dosage_summary_three_groups = get_three_group_dosage_summary(medicine_name)
        
        return jsonify({
            'medicine_name': medicine_name,
            'dataset_info': dataset_info,
            'gemini_info': gemini_info,
            'age_dosages': age_dosages,
            'dosage_summary_three_groups': dosage_summary_three_groups
        })
    except Exception as e:
        return jsonify({'error': f'Error processing search: {str(e)}'}), 500

@app.route('/disease_recommendations', methods=['POST'])
def disease_recommendations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        disease_key = data.get('disease', '').strip()
        allergies = data.get('allergies', [])  # List of allergy keys
        health_conditions = data.get('health_conditions', [])  # List of health condition keys
        
        if not disease_key:
            return jsonify({'error': 'Disease is required'}), 400
        
        recommendations = get_disease_recommendations(disease_key, allergies, health_conditions)
        
        if not recommendations:
            return jsonify({'error': 'Disease not found'}), 404
        
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/symptom_recommendations', methods=['POST'])
def symptom_recommendations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        symptoms = data.get('symptoms', [])  # List of symptom keys
        allergies = data.get('allergies', [])  # List of allergy keys
        health_conditions = data.get('health_conditions', [])  # List of health condition keys
        
        if not symptoms:
            return jsonify({'error': 'At least one symptom is required'}), 400
        
        recommendations = get_medicine_by_symptoms(symptoms, allergies, health_conditions)
        
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/api/medicines')
def get_medicines():
    if medicine_df is None:
        return jsonify({'error': 'Medicine database not available'}), 500
    
    try:
        # Return list of all medicines for autocomplete
        medicines = medicine_df['Medicine Name'].tolist()
        return jsonify({'medicines': medicines})
    except Exception as e:
        return jsonify({'error': f'Error retrieving medicines: {str(e)}'}), 500

@app.route('/api/diseases')
def get_diseases():
    try:
        return jsonify({'diseases': COMMON_DISEASES})
    except Exception as e:
        return jsonify({'error': f'Error retrieving diseases: {str(e)}'}), 500

@app.route('/api/allergies')
def get_allergies():
    try:
        return jsonify({'allergies': COMMON_ALLERGIES})
    except Exception as e:
        return jsonify({'error': f'Error retrieving allergies: {str(e)}'}), 500

@app.route('/api/health_conditions')
def get_health_conditions():
    try:
        return jsonify({'health_conditions': HEALTH_CONDITIONS})
    except Exception as e:
        return jsonify({'error': f'Error retrieving health conditions: {str(e)}'}), 500

@app.route('/api/symptoms')
def get_symptoms():
    try:
        # Get all symptoms from diseases
        all_symptoms = []
        for disease_key, disease_info in COMMON_DISEASES.items():
            for symptom in disease_info['symptoms']:
                all_symptoms.append({
                    'key': symptom.lower().replace(' ', '_'),
                    'name': symptom,
                    'name_marathi': disease_info['symptoms_marathi'][disease_info['symptoms'].index(symptom)] if disease_info['symptoms'].index(symptom) < len(disease_info['symptoms_marathi']) else symptom,
                    'disease': disease_key
                })
        
        # Remove duplicates
        unique_symptoms = []
        seen = set()
        for symptom in all_symptoms:
            if symptom['key'] not in seen:
                seen.add(symptom['key'])
                unique_symptoms.append(symptom)
        
        return jsonify({'symptoms': unique_symptoms})
    except Exception as e:
        return jsonify({'error': f'Error retrieving symptoms: {str(e)}'}), 500

@app.route('/weight_based_dosage', methods=['POST'])
def weight_based_dosage():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        medicine_name = data.get('medicine_name', '').strip()
        weight_kg = data.get('weight_kg')
        age_group = data.get('age_group', '').strip()
        
        if not medicine_name:
            return jsonify({'error': 'Medicine name is required'}), 400
        
        if weight_kg is None or weight_kg <= 0:
            return jsonify({'error': 'Valid weight in kg is required'}), 400
        
        if not age_group:
            return jsonify({'error': 'Age group is required'}), 400
        
        # Calculate weight-based dosage
        dosage_info = calculate_weight_based_dosage(medicine_name, weight_kg, age_group)
        
        # Get additional medicine information
        medicine_info = get_medicine_info_from_dataset(medicine_name)
        
        # Get age-based dosage for comparison
        age_dosage = get_age_based_dosage(medicine_name, age_group)
        
        return jsonify({
            'medicine_name': medicine_name,
            'weight_kg': weight_kg,
            'age_group': age_group,
            'weight_based_dosage': dosage_info,
            'age_based_dosage': age_dosage,
            'medicine_info': medicine_info,
            'recommendations': {
                'for_children': 'Use liquid formulations for easier administration',
                'for_adults': 'Tablets or capsules are preferred',
                'general': 'Always consult with a healthcare provider before taking any medication'
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error calculating dosage: {str(e)}'}), 500

@app.route('/api/age_groups')
def get_age_groups():
    try:
        age_groups = []
        for key, info in AGE_DOSAGE_GUIDELINES.items():
            age_groups.append({
                'key': key,
                'name': info['name'],
                'description': f'Age group: {key.replace("_", " ")}'
            })
        return jsonify({'age_groups': age_groups})
    except Exception as e:
        return jsonify({'error': f'Error retrieving age groups: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    print("🚀 Starting MedDetect Flask Application...")
    print(f"📊 Medicine database: {'✅ Loaded' if medicine_df is not None else '❌ Not available'}")
    print(f"🤖 Gemini AI: {'✅ Initialized' if vision_model and text_model else '❌ Not available'}")
    print("🌐 Server starting at http://localhost:5000")
    app.run()