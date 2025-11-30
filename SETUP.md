# 🚀 Quick Setup Guide - MedDetect

## Prerequisites
- Python 3.8 or higher
- Google Gemini API key

## 🛠️ Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxx
FLASK_SECRET_KEY=6413749eb506f8d52efaa8523dfa6594e5fd77cecf286d42548fd92727db4630
```

### 3. Get Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Add it to your `.env` file

### 4. Run the Application

**Option 1: Using the startup script (Recommended)**
```bash
python run.py
```

**Option 2: Direct Flask run**
```bash
python app.py
```

### 5. Access the Application
Open your browser and go to: `http://localhost:5000`

## 🎯 Features Available

### ✅ Without API Key
- Landing page with information
- About and Contact pages
- Search functionality (limited to database)
- Modern, responsive UI

### ✅ With API Key
- Image upload and medicine detection
- AI-powered medicine identification
- Multilingual information (English, Hindi, Marathi)
- Comprehensive medicine details

## 🔧 Troubleshooting

### Common Issues

1. **"Medicine database not available"**
   - Ensure `Medicine_Details.csv` is in the root directory

2. **"Gemini AI not initialized"**
   - Check your `.env` file has the correct API key
   - Verify the API key is valid

3. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Ensure you're in the correct virtual environment

4. **Port already in use**
   - Change the port in `app.py` or `run.py`
   - Kill the process using the port

### File Structure Check
Ensure you have these files and directories:
```
Medicine-Detection/
├── app.py
├── run.py
├── requirements.txt
├── Medicine_Details.csv
├── .env (create this)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── detect.html
│   ├── about.html
│   ├── contact.html
│   └── 404.html
└── static/
    ├── uploads/
    ├── css/
    └── js/
```

## 🎨 Customization

### Changing Colors
Edit the CSS variables in `templates/base.html`:
```css
:root {
    --primary-color: #2563eb;
    --secondary-color: #1e40af;
    /* ... other colors */
}
```

### Adding New Pages
1. Create a new template in `templates/`
2. Add a route in `app.py`
3. Update navigation in `templates/base.html`

## 📞 Support

If you encounter issues:
1. Check the FAQ section on the Contact page
2. Review the console output for error messages
3. Ensure all dependencies are installed
4. Verify your API key is correct

---

**Happy Coding! 🎉** 