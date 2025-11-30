import os
import sys
import time
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MedDetectSystem:
    """
    Comprehensive Medicine Detection System that integrates:
    - Deep Learning Model (LSTM)
    - Machine Learning Model (RF/SVM)
    - OCR Processing
    - Web Interface
    - Model Evaluation
    """
    
    def __init__(self):
        self.models = {}
        self.ocr_processor = None
        self.database = None
        self.evaluator = None
        self.initialized = False
        
    def initialize_system(self) -> bool:
        """Initialize all components of the system"""
        print("🚀 Initializing MedDetect System...")
        
        try:
            # Initialize OCR
            print("📷 Initializing OCR processor...")
            from ocr import OCRProcessor
            self.ocr_processor = OCRProcessor()
            print("✅ OCR processor initialized")
            
            # Initialize database
            print("🗄️ Initializing database...")
            from database import MedicineDatabase
            self.database = MedicineDatabase()
            print("✅ Database initialized")
            
            # Load deep learning model
            print("🧠 Loading deep learning model...")
            try:
                from medicine_model import MedicinePredictor
                dl_predictor = MedicinePredictor()
                if dl_predictor.load_model():
                    self.models['deep_learning'] = dl_predictor
                    print("✅ Deep learning model loaded")
                else:
                    print("⚠️ Deep learning model not available")
            except Exception as e:
                print(f"⚠️ Deep learning model error: {e}")
            
            # Load machine learning model
            print("🤖 Loading machine learning model...")
            try:
                from medicine_ml_model import MedicineMLPredictor
                ml_predictor = MedicineMLPredictor()
                if ml_predictor.load_models():
                    self.models['machine_learning'] = ml_predictor
                    print("✅ Machine learning model loaded")
                else:
                    print("⚠️ Machine learning model not available")
            except Exception as e:
                print(f"⚠️ Machine learning model error: {e}")
            
            # Initialize evaluator
            print("📊 Initializing model evaluator...")
            from model_evaluation import ModelEvaluator
            self.evaluator = ModelEvaluator()
            print("✅ Model evaluator initialized")
            
            self.initialized = True
            print(f"🎉 System initialized successfully with {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def predict_medicine(self, medicine_name: str, model_type: str = 'ensemble') -> Dict:
        """
        Predict medicine information using specified model or ensemble
        
        Args:
            medicine_name: Name of the medicine to predict
            model_type: 'deep_learning', 'machine_learning', or 'ensemble'
        
        Returns:
            Dictionary with prediction results and confidence scores
        """
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        print(f"🔍 Predicting medicine: {medicine_name} using {model_type}")
        
        results = {}
        
        if model_type == 'ensemble' or model_type == 'deep_learning':
            if 'deep_learning' in self.models:
                try:
                    dl_result = self.models['deep_learning'].predict_medicine(medicine_name)
                    results['deep_learning'] = dl_result
                    print(f"✅ Deep learning prediction completed")
                except Exception as e:
                    print(f"❌ Deep learning prediction failed: {e}")
                    results['deep_learning'] = None
        
        if model_type == 'ensemble' or model_type == 'machine_learning':
            if 'machine_learning' in self.models:
                try:
                    ml_result = self.models['machine_learning'].predict_medicine(medicine_name)
                    results['machine_learning'] = ml_result
                    print(f"✅ Machine learning prediction completed")
                except Exception as e:
                    print(f"❌ Machine learning prediction failed: {e}")
                    results['machine_learning'] = None
        
        # Combine results for ensemble
        if model_type == 'ensemble':
            return self._combine_predictions(results)
        else:
            return results.get(model_type, {'error': f'Model {model_type} not available'})
    
    def _combine_predictions(self, results: Dict) -> Dict:
        """Combine predictions from multiple models"""
        combined = {
            'medicine_name': '',
            'predicted_uses': '',
            'predicted_side_effects': '',
            'uses_confidence': 0.0,
            'side_effects_confidence': 0.0,
            'model_contributions': {}
        }
        
        valid_predictions = []
        
        for model_name, result in results.items():
            if result and 'error' not in result:
                valid_predictions.append((model_name, result))
                combined['model_contributions'][model_name] = {
                    'uses_confidence': result.get('uses_confidence', 0),
                    'side_effects_confidence': result.get('side_effects_confidence', 0)
                }
        
        if not valid_predictions:
            return {'error': 'No valid predictions from any model'}
        
        # Use the prediction with highest confidence
        best_prediction = max(valid_predictions, 
                            key=lambda x: (x[1].get('uses_confidence', 0) + x[1].get('side_effects_confidence', 0)) / 2)
        
        combined.update(best_prediction[1])
        combined['ensemble_confidence'] = (combined['uses_confidence'] + combined['side_effects_confidence']) / 2
        
        return combined
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process image using OCR and predict medicine information
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary with OCR results and predictions
        """
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        print(f"📷 Processing image: {image_path}")
        
        try:
            # Extract medicine name using OCR
            medicine_name = self.ocr_processor.extract_medicine_name(image_path)
            
            if not medicine_name:
                return {'error': 'Could not extract medicine name from image'}
            
            print(f"🔍 Extracted medicine name: {medicine_name}")
            
            # Predict medicine information
            prediction = self.predict_medicine(medicine_name, 'ensemble')
            
            return {
                'ocr_result': medicine_name,
                'prediction': prediction,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return {'error': f'Image processing failed: {e}'}
    
    def evaluate_models(self, test_medicines: List[str] = None) -> Dict:
        """
        Evaluate all loaded models
        
        Args:
            test_medicines: List of medicine names to test
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        print("📊 Starting model evaluation...")
        
        if test_medicines is None:
            test_medicines = [
                'Paracetamol', 'Aspirin', 'Ibuprofen', 'Omeprazole', 
                'Amoxicillin', 'Metformin', 'Atorvastatin', 'Lisinopril',
                'Amlodipine', 'Simvastatin', 'Losartan', 'Metoprolol'
            ]
        
        # Use the evaluator to run comprehensive evaluation
        self.evaluator.load_models()
        self.evaluator.evaluate_models(test_medicines)
        comparison_df = self.evaluator.generate_comprehensive_report()
        
        return {
            'success': True,
            'comparison_table': comparison_df.to_dict('records'),
            'message': 'Evaluation completed successfully'
        }
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'initialized': self.initialized,
            'models_loaded': len(self.models),
            'available_models': list(self.models.keys()),
            'ocr_available': self.ocr_processor is not None,
            'database_available': self.database is not None,
            'evaluator_available': self.evaluator is not None
        }
    
    def search_database(self, query: str) -> List[Dict]:
        """
        Search the medicine database
        
        Args:
            query: Search query
        
        Returns:
            List of matching medicines
        """
        if not self.initialized or not self.database:
            return []
        
        return self.database.search_medicines(query)
    
    def get_medicine_info(self, medicine_name: str) -> Dict:
        """
        Get medicine information from database
        
        Args:
            medicine_name: Name of the medicine
        
        Returns:
            Medicine information dictionary
        """
        if not self.initialized or not self.database:
            return {}
        
        return self.database.get_medicine_info(medicine_name)

def main():
    """Main function to demonstrate system integration"""
    print("🏥 MedDetect System Integration Demo")
    print("=" * 50)
    
    # Initialize system
    system = MedDetectSystem()
    
    if not system.initialize_system():
        print("❌ Failed to initialize system")
        return
    
    # Display system status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Demo: Search database
    print(f"\n🔍 Database Search Demo:")
    results = system.search_database("paracetamol")
    if results:
        print(f"   Found {len(results)} medicines")
        for medicine in results[:3]:
            print(f"   - {medicine['name']}")
    
    # Demo: Predict medicine
    print(f"\n🤖 Prediction Demo:")
    prediction = system.predict_medicine("Paracetamol", "ensemble")
    if 'error' not in prediction:
        print(f"   Medicine: {prediction.get('medicine_name', 'Unknown')}")
        print(f"   Uses: {prediction.get('predicted_uses', 'Not found')[:100]}...")
        print(f"   Confidence: {prediction.get('ensemble_confidence', 0):.3f}")
    else:
        print(f"   Prediction failed: {prediction['error']}")
    
    # Demo: Model evaluation
    print(f"\n📊 Model Evaluation Demo:")
    evaluation = system.evaluate_models(['Paracetamol', 'Aspirin', 'Ibuprofen'])
    if 'success' in evaluation:
        print("   Evaluation completed successfully")
        print("   Check generated files for detailed results")
    else:
        print(f"   Evaluation failed: {evaluation.get('error', 'Unknown error')}")
    
    print(f"\n✅ Integration demo completed!")
    print(f"💡 Use the web interface for full functionality")

if __name__ == "__main__":
    main() 