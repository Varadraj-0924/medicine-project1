# Medicine Detection System

A comprehensive AI-powered medicine detection and information system that helps users identify medicines, get detailed information, and receive personalized dosage recommendations based on age groups and disease conditions.

## 🚀 Features

### Core Features
- **Image-based Medicine Detection**: Upload medicine images for instant identification using AI
- **Comprehensive Medicine Database**: Access detailed information about 1000+ medicines
- **Multilingual Support**: Information available in English, Hindi, and Marathi
- **AI-Powered Analysis**: Powered by Google Gemini AI for accurate medicine identification

### New Features (Latest Update)

#### 1. Age-Based Dosage Recommendations
- **Personalized Dosage Guidelines**: Get dosage recommendations based on age groups:
  - Infants (0-2 years)
  - Children (3-12 years)
  - Teenagers (13-15 years)
  - Young Adults (16-30 years)
  - Adults (31-50 years)
  - Elderly (51+ years)
- **Safety Notes**: Each age group includes specific safety precautions and warnings
- **Medicine-Specific Dosages**: Detailed dosage information for common medicines like Paracetamol and Ibuprofen

#### 2. Disease-Based Medicine Recommendations
- **Common Diseases Coverage**: 10 major disease categories including:
  - Fever
  - Headache
  - Cough & Cold
  - Diarrhea & Constipation
  - Allergies
  - Insomnia
  - Acid Reflux
  - Hypertension
- **Symptom Information**: Detailed symptoms for each disease
- **Recommended Medicines**: Curated list of medicines for each condition
- **Interactive Disease Selection**: User-friendly interface to browse and select diseases

#### 3. Enhanced User Interface
- **New Diseases Page**: Dedicated page for disease-based medicine recommendations
- **Improved Navigation**: Updated navigation menu with diseases section
- **Responsive Design**: Mobile-friendly interface for all devices
- **Visual Enhancements**: Modern UI with icons and intuitive design

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **AI/ML**: Google Gemini AI (Vision & Text models)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Database**: CSV-based medicine database
- **Image Processing**: PIL (Python Imaging Library)

## 📋 Prerequisites

- Python 3.7+
- Google Gemini API key
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Medicine-Detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and go to `http://localhost:5000`

## 📱 Usage

### Medicine Detection
1. Go to the "Detect" page
2. Upload an image of the medicine or search by name
3. View comprehensive information including:
   - Uses and indications
   - Side effects and precautions
   - Age-based dosage recommendations
   - AI-generated multilingual information

### Disease-Based Recommendations
1. Navigate to the "Diseases" page
2. Select a disease from the available options
3. View recommended medicines with detailed information
4. Get symptom descriptions and treatment guidance

### Age-Based Dosage
- Automatically displayed with medicine detection results
- Shows dosage recommendations for all age groups
- Includes safety notes and precautions
- Provides specific dosage information for common medicines

## 🏗️ Project Structure

```
Medicine Detection/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Homepage
│   ├── detect.html       # Medicine detection page
│   ├── diseases.html     # Disease recommendations page
│   ├── about.html        # About page
│   ├── contact.html      # Contact page
│   └── 404.html          # Error page
├── static/               # Static files
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── uploads/         # Uploaded images (temporary)
├── Medicine_Details.csv  # Medicine database
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `FLASK_SECRET_KEY`: Secret key for Flask sessions

### File Upload Settings
- Maximum file size: 10MB
- Supported formats: PNG, JPG, JPEG, GIF
- Upload folder: `static/uploads/`

## 📊 Database Structure

The medicine database (`Medicine_Details.csv`) contains:
- Medicine Name
- Composition
- Uses
- Side Effects
- Precautions
- Dosage Information
- Manufacturer Details
- Review Ratings

## 🎯 Key Features Explained

### Age-Based Dosage System
The system provides age-specific dosage recommendations based on medical guidelines:

- **Infants (0-2 years)**: Conservative dosages with pediatrician consultation required
- **Children (3-12 years)**: Weight-based dosages with maximum limits
- **Teenagers (13-15 years)**: Transition to adult dosages with supervision
- **Young Adults (16-30 years)**: Standard adult dosages
- **Adults (31-50 years)**: Standard dosages with drug interaction monitoring
- **Elderly (51+ years)**: Reduced dosages with caution notes

### Disease Recommendation System
Comprehensive coverage of common diseases with:
- Detailed disease descriptions
- Common symptoms
- Recommended medicine lists
- Safety precautions
- Treatment guidelines

## 🔒 Security Features

- Secure file upload handling
- Input validation and sanitization
- Environment variable protection
- Temporary file cleanup
- Error handling and logging

## 🌐 API Endpoints

- `GET /`: Homepage
- `GET /detect`: Medicine detection page
- `GET /diseases`: Disease recommendations page
- `POST /upload`: Image upload and medicine detection
- `POST /search`: Text-based medicine search
- `POST /disease_recommendations`: Disease-based medicine recommendations
- `GET /api/medicines`: Medicine list for autocomplete
- `GET /api/diseases`: Disease list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider before taking any medication.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔄 Updates

### Latest Version (v2.0)
- Added age-based dosage recommendations
- Implemented disease-based medicine suggestions
- Enhanced user interface with new diseases page
- Improved navigation and user experience
- Added comprehensive documentation

---

**Note**: This system is designed to assist users in making informed decisions about medicines but should not replace professional medical consultation. 