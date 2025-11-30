#!/usr/bin/env python3
"""
MedDetect - AI-Powered Medicine Detection System
Startup script for the Flask application
"""

import os
import sys
from dotenv import load_dotenv

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("🔍 Checking requirements...")
    
    # Check if CSV file exists
    if not os.path.exists('Medicine_Details.csv'):
        print("❌ Error: Medicine_Details.csv not found!")
        print("   Please ensure the medicine database file is in the root directory.")
        return False
    
    # Check if templates directory exists
    if not os.path.exists('templates'):
        print("❌ Error: templates directory not found!")
        return False
    
    # Check if static directory exists
    if not os.path.exists('static'):
        print("❌ Error: static directory not found!")
        return False
    
    print("✅ All required files found")
    return True

def check_environment():
    """Check environment variables"""
    print("🔍 Checking environment...")
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  Warning: GOOGLE_API_KEY not set in environment")
        print("   The AI features will not work without a valid API key")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        print("   Create a .env file with: GOOGLE_API_KEY=your_key_here")
    else:
        print("✅ Google API key found")
    
    secret_key = os.getenv('FLASK_SECRET_KEY')
    if not secret_key:
        print("⚠️  Warning: FLASK_SECRET_KEY not set, using default")
        print("   For production, set a secure secret key in .env file")
    else:
        print("✅ Flask secret key found")
    
    return True

def main():
    """Main startup function"""
    print("🚀 MedDetect - AI-Powered Medicine Detection System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    print("\n" + "=" * 50)
    print("🎯 Starting Flask application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 