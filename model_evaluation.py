import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def load_models(self):
        """Load all trained models"""
        print("📂 Loading trained models...")
        
        # Try to load deep learning model
        try:
            from medicine_model import MedicinePredictor
            dl_predictor = MedicinePredictor()
            if dl_predictor.load_model():
                self.models['Deep Learning (LSTM)'] = dl_predictor
                print("✅ Deep Learning model loaded")
            else:
                print("⚠️ Deep Learning model not available")
        except Exception as e:
            print(f"⚠️ Deep Learning model error: {e}")
        
        # Try to load ML model
        try:
            from medicine_ml_model import MedicineMLPredictor
            ml_predictor = MedicineMLPredictor()
            if ml_predictor.load_models():
                self.models['Machine Learning (RF/SVM)'] = ml_predictor
                print("✅ Machine Learning model loaded")
            else:
                print("⚠️ Machine Learning model not available")
        except Exception as e:
            print(f"⚠️ Machine Learning model error: {e}")
        
        print(f"📊 Loaded {len(self.models)} models")
    
    def evaluate_models(self, test_medicines=None):
        """Evaluate all loaded models"""
        if not self.models:
            print("❌ No models loaded")
            return
        
        if test_medicines is None:
            test_medicines = [
                'Paracetamol', 'Aspirin', 'Ibuprofen', 'Omeprazole', 
                'Amoxicillin', 'Metformin', 'Atorvastatin', 'Lisinopril',
                'Amlodipine', 'Simvastatin', 'Losartan', 'Metoprolol'
            ]
        
        print(f"\n🧪 Evaluating {len(self.models)} models with {len(test_medicines)} test medicines")
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Testing {model_name}...")
            
            results = {
                'predictions': [],
                'response_times': [],
                'confidences': []
            }
            
            for medicine in test_medicines:
                start_time = time.time()
                
                try:
                    prediction = model.predict_medicine(medicine)
                    end_time = time.time()
                    
                    if prediction:
                        results['predictions'].append({
                            'medicine': medicine,
                            'uses': prediction.get('predicted_uses', 'Not found'),
                            'side_effects': prediction.get('predicted_side_effects', 'Not found'),
                            'uses_confidence': prediction.get('uses_confidence', 0),
                            'side_effects_confidence': prediction.get('side_effects_confidence', 0)
                        })
                        results['response_times'].append(end_time - start_time)
                        results['confidences'].append(
                            (prediction.get('uses_confidence', 0) + prediction.get('side_effects_confidence', 0)) / 2
                        )
                    else:
                        results['predictions'].append({
                            'medicine': medicine,
                            'uses': 'Failed',
                            'side_effects': 'Failed',
                            'uses_confidence': 0,
                            'side_effects_confidence': 0
                        })
                        results['response_times'].append(end_time - start_time)
                        results['confidences'].append(0)
                        
                except Exception as e:
                    print(f"   ❌ Error predicting {medicine}: {e}")
                    results['predictions'].append({
                        'medicine': medicine,
                        'uses': 'Error',
                        'side_effects': 'Error',
                        'uses_confidence': 0,
                        'side_effects_confidence': 0
                    })
                    results['response_times'].append(0)
                    results['confidences'].append(0)
            
            self.results[model_name] = results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        print("\n📊 Generating comprehensive evaluation report...")
        
        # Create comparison table
        comparison_data = []
        
        for model_name, results in self.results.items():
            avg_response_time = np.mean(results['response_times'])
            avg_confidence = np.mean(results['confidences'])
            success_rate = len([p for p in results['predictions'] if p['uses'] != 'Failed' and p['uses'] != 'Error']) / len(results['predictions'])
            
            comparison_data.append({
                'Model': model_name,
                'Success Rate': f"{success_rate:.2%}",
                'Avg Response Time (s)': f"{avg_response_time:.3f}",
                'Avg Confidence': f"{avg_confidence:.3f}",
                'Total Predictions': len(results['predictions'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📋 Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Create detailed results
        self.create_detailed_results()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save report
        self.save_evaluation_report(comparison_df)
        
        return comparison_df
    
    def create_detailed_results(self):
        """Create detailed results for each model"""
        print("\n📝 Creating detailed results...")
        
        for model_name, results in self.results.items():
            print(f"\n🔍 {model_name} Detailed Results:")
            
            successful_predictions = [p for p in results['predictions'] if p['uses'] != 'Failed' and p['uses'] != 'Error']
            
            if successful_predictions:
                avg_uses_confidence = np.mean([p['uses_confidence'] for p in successful_predictions])
                avg_side_effects_confidence = np.mean([p['side_effects_confidence'] for p in successful_predictions])
                
                print(f"   ✅ Successful Predictions: {len(successful_predictions)}/{len(results['predictions'])}")
                print(f"   🎯 Average Uses Confidence: {avg_uses_confidence:.3f}")
                print(f"   🎯 Average Side Effects Confidence: {avg_side_effects_confidence:.3f}")
                print(f"   ⚡ Average Response Time: {np.mean(results['response_times']):.3f}s")
                
                # Show sample predictions
                print(f"\n   📋 Sample Predictions:")
                for i, pred in enumerate(successful_predictions[:3]):
                    print(f"      {i+1}. {pred['medicine']}")
                    print(f"         Uses: {pred['uses'][:80]}...")
                    print(f"         Side Effects: {pred['side_effects'][:80]}...")
                    print(f"         Confidence: {pred['uses_confidence']:.3f}, {pred['side_effects_confidence']:.3f}")
            else:
                print("   ❌ No successful predictions")
    
    def create_visualizations(self):
        """Create visualizations for model comparison"""
        print("\n📈 Creating visualizations...")
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        success_rates = []
        response_times = []
        confidences = []
        
        for model_name in model_names:
            results = self.results[model_name]
            success_rate = len([p for p in results['predictions'] if p['uses'] != 'Failed' and p['uses'] != 'Error']) / len(results['predictions'])
            avg_response_time = np.mean(results['response_times'])
            avg_confidence = np.mean(results['confidences'])
            
            success_rates.append(success_rate)
            response_times.append(avg_response_time)
            confidences.append(avg_confidence)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success Rate Comparison
        axes[0, 0].bar(model_names, success_rates, color=['#2E86AB', '#A23B72'])
        axes[0, 0].set_title('Model Success Rate Comparison')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
        
        # Response Time Comparison
        axes[0, 1].bar(model_names, response_times, color=['#F18F01', '#C73E1D'])
        axes[0, 1].set_title('Average Response Time Comparison')
        axes[0, 1].set_ylabel('Response Time (seconds)')
        for i, v in enumerate(response_times):
            axes[0, 1].text(i, v + 0.001, f'{v:.3f}s', ha='center', va='bottom')
        
        # Confidence Comparison
        axes[1, 0].bar(model_names, confidences, color=['#4CAF50', '#FF9800'])
        axes[1, 0].set_title('Average Confidence Comparison')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(confidences):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Model Performance Radar Chart
        categories = ['Success Rate', 'Speed (1/Response Time)', 'Confidence']
        values = []
        for i, model_name in enumerate(model_names):
            # Normalize response time (inverse for speed)
            speed_score = 1 / (response_times[i] + 0.001)  # Add small value to avoid division by zero
            values.append([success_rates[i], speed_score, confidences[i]])
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = np.array(values)
        
        ax = axes[1, 1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), categories)
        
        for i, model_name in enumerate(model_names):
            ax.plot(angles, values[i], 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values[i], alpha=0.25)
        
        ax.set_title('Model Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_report(self, comparison_df):
        """Save comprehensive evaluation report"""
        print("\n💾 Saving evaluation report...")
        
        with open('comprehensive_evaluation_report.txt', 'w') as f:
            f.write("Comprehensive Medicine Prediction Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Model Comparison Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("Detailed Analysis:\n")
            f.write("-" * 20 + "\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"{'='*len(model_name)+'='*8}\n")
                
                successful_predictions = [p for p in results['predictions'] if p['uses'] != 'Failed' and p['uses'] != 'Error']
                failed_predictions = [p for p in results['predictions'] if p['uses'] == 'Failed' or p['uses'] == 'Error']
                
                f.write(f"Total Predictions: {len(results['predictions'])}\n")
                f.write(f"Successful Predictions: {len(successful_predictions)}\n")
                f.write(f"Failed Predictions: {len(failed_predictions)}\n")
                f.write(f"Success Rate: {len(successful_predictions)/len(results['predictions']):.2%}\n")
                f.write(f"Average Response Time: {np.mean(results['response_times']):.3f} seconds\n")
                f.write(f"Average Confidence: {np.mean(results['confidences']):.3f}\n\n")
                
                if successful_predictions:
                    f.write("Sample Successful Predictions:\n")
                    for i, pred in enumerate(successful_predictions[:5]):
                        f.write(f"{i+1}. {pred['medicine']}\n")
                        f.write(f"   Uses: {pred['uses'][:100]}...\n")
                        f.write(f"   Side Effects: {pred['side_effects'][:100]}...\n")
                        f.write(f"   Confidence: {pred['uses_confidence']:.3f}, {pred['side_effects_confidence']:.3f}\n\n")
                
                if failed_predictions:
                    f.write("Failed Predictions:\n")
                    for pred in failed_predictions:
                        f.write(f"- {pred['medicine']}: {pred['uses']}\n")
                
                f.write("\n" + "="*50 + "\n\n")
            
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")
            f.write("1. Use the model with the highest success rate for production\n")
            f.write("2. Consider response time requirements for real-time applications\n")
            f.write("3. Monitor confidence scores to identify low-confidence predictions\n")
            f.write("4. Regularly retrain models with new data for better performance\n")
            f.write("5. Implement ensemble methods to combine multiple models\n")
        
        print("✅ Evaluation report saved as 'comprehensive_evaluation_report.txt'")
        print("✅ Visualization saved as 'model_comparison_analysis.png'")

def main():
    """Main function to run comprehensive model evaluation"""
    print("🔍 Comprehensive Model Evaluation System")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models()
    
    if not evaluator.models:
        print("❌ No models available for evaluation")
        print("💡 Please train models first using:")
        print("   - python medicine_model.py (for deep learning)")
        print("   - python medicine_ml_model.py (for machine learning)")
        return
    
    # Define test medicines
    test_medicines = [
        'Paracetamol', 'Aspirin', 'Ibuprofen', 'Omeprazole', 
        'Amoxicillin', 'Metformin', 'Atorvastatin', 'Lisinopril',
        'Amlodipine', 'Simvastatin', 'Losartan', 'Metoprolol',
        'Cetirizine', 'Ranitidine', 'Diclofenac', 'Tramadol'
    ]
    
    # Evaluate models
    evaluator.evaluate_models(test_medicines)
    
    # Generate comprehensive report
    comparison_df = evaluator.generate_comprehensive_report()
    
    print("\n✅ Comprehensive evaluation completed!")
    print("📊 Check the generated files for detailed analysis")

if __name__ == "__main__":
    main() 