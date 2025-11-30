import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MedicineMLPredictor:
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        self.uses_encoder = LabelEncoder()
        self.side_effects_encoder = LabelEncoder()
        self.uses_model = None
        self.side_effects_model = None
        self.best_uses_model = None
        self.best_side_effects_model = None
        
    def load_data(self, csv_file='Medicine_Details.csv'):
        """Load and preprocess the medicine dataset"""
        print("📊 Loading medicine dataset...")
        self.df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(self.df)} medicines")
        
        # Clean the data
        self.df = self.df.dropna(subset=['Medicine Name', 'Uses', 'Side_effects'])
        self.df = self.df[self.df['Uses'].str.len() > 10]
        self.df = self.df[self.df['Side_effects'].str.len() > 10]
        
        print(f"📈 After cleaning: {len(self.df)} medicines")
        return self.df
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self):
        """Prepare data for training"""
        print("🔧 Preparing data for training...")
        
        # Preprocess medicine names
        self.df['Medicine_Name_Clean'] = self.df['Medicine Name'].apply(self.preprocess_text)
        self.df['Uses_Clean'] = self.df['Uses'].apply(self.preprocess_text)
        self.df['Side_Effects_Clean'] = self.df['Side_effects'].apply(self.preprocess_text)
        
        # Create TF-IDF features for medicine names
        medicine_names = self.df['Medicine_Name_Clean'].tolist()
        self.X = self.tfidf_vectorizer.fit_transform(medicine_names)
        
        # Encode uses and side effects
        self.df['Uses_Encoded'] = self.uses_encoder.fit_transform(self.df['Uses_Clean'])
        self.df['Side_Effects_Encoded'] = self.side_effects_encoder.fit_transform(self.df['Side_Effects_Clean'])
        
        # Prepare labels
        self.y_uses = self.df['Uses_Encoded'].values
        self.y_side_effects = self.df['Side_Effects_Encoded'].values
        
        print(f"📝 TF-IDF features: {self.X.shape[1]}")
        print(f"🏷️ Number of unique uses: {len(self.uses_encoder.classes_)}")
        print(f"🏷️ Number of unique side effects: {len(self.side_effects_encoder.classes_)}")
        
        return self.X, self.y_uses, self.y_side_effects
    
    def train_models(self):
        """Train multiple models and select the best ones"""
        print("🚀 Training machine learning models...")
        
        # Split the data
        X_train, X_test, y_uses_train, y_uses_test, y_side_effects_train, y_side_effects_test = train_test_split(
            self.X, self.y_uses, self.y_side_effects, test_size=0.2, random_state=42
        )
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_uses_score = 0
        best_side_effects_score = 0
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n🔍 Training {name}...")
            
            # Train for uses
            model_uses = model.__class__(**model.get_params())
            model_uses.fit(X_train, y_uses_train)
            uses_score = model_uses.score(X_test, y_uses_test)
            
            # Train for side effects
            model_side_effects = model.__class__(**model.get_params())
            model_side_effects.fit(X_train, y_side_effects_train)
            side_effects_score = model_side_effects.score(X_test, y_side_effects_test)
            
            print(f"   Uses Accuracy: {uses_score:.4f}")
            print(f"   Side Effects Accuracy: {side_effects_score:.4f}")
            
            # Update best models
            if uses_score > best_uses_score:
                best_uses_score = uses_score
                self.best_uses_model = model_uses
                self.uses_model_name = name
            
            if side_effects_score > best_side_effects_score:
                best_side_effects_score = side_effects_score
                self.best_side_effects_model = model_side_effects
                self.side_effects_model_name = name
        
        print(f"\n🏆 Best Uses Model: {self.uses_model_name} (Accuracy: {best_uses_score:.4f})")
        print(f"🏆 Best Side Effects Model: {self.side_effects_model_name} (Accuracy: {best_side_effects_score:.4f})")
        
        # Evaluate best models
        self.evaluate_models(X_test, y_uses_test, y_side_effects_test)
        
        # Save models
        self.save_models()
        
        return best_uses_score, best_side_effects_score
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best models"""
        print("\n🔧 Performing hyperparameter tuning...")
        
        X_train, X_test, y_uses_train, y_uses_test, y_side_effects_train, y_side_effects_test = train_test_split(
            self.X, self.y_uses, self.y_side_effects, test_size=0.2, random_state=42
        )
        
        # Tune Random Forest for uses
        if self.uses_model_name == 'Random Forest':
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            rf_grid_search.fit(X_train, y_uses_train)
            self.best_uses_model = rf_grid_search.best_estimator_
            print(f"Best RF parameters for uses: {rf_grid_search.best_params_}")
        
        # Tune SVM for side effects
        if self.side_effects_model_name == 'SVM':
            svm_param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
            
            svm_grid_search = GridSearchCV(
                SVC(random_state=42),
                svm_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            svm_grid_search.fit(X_train, y_side_effects_train)
            self.best_side_effects_model = svm_grid_search.best_estimator_
            print(f"Best SVM parameters for side effects: {svm_grid_search.best_params_}")
    
    def evaluate_models(self, X_test, y_uses_test, y_side_effects_test):
        """Evaluate the best models"""
        print("\n📊 Model Evaluation:")
        
        # Uses predictions
        uses_pred = self.best_uses_model.predict(X_test)
        uses_accuracy = accuracy_score(y_uses_test, uses_pred)
        print(f"🎯 Uses Prediction Accuracy: {uses_accuracy:.4f}")
        
        # Side effects predictions
        side_effects_pred = self.best_side_effects_model.predict(X_test)
        side_effects_accuracy = accuracy_score(y_side_effects_test, side_effects_pred)
        print(f"🎯 Side Effects Prediction Accuracy: {side_effects_accuracy:.4f}")
        
        # Cross-validation scores
        uses_cv_scores = cross_val_score(self.best_uses_model, self.X, self.y_uses, cv=5)
        side_effects_cv_scores = cross_val_score(self.best_side_effects_model, self.X, self.y_side_effects, cv=5)
        
        print(f"📊 Uses CV Score: {uses_cv_scores.mean():.4f} (+/- {uses_cv_scores.std() * 2:.4f})")
        print(f"📊 Side Effects CV Score: {side_effects_cv_scores.mean():.4f} (+/- {side_effects_cv_scores.std() * 2:.4f})")
        
        # Detailed classification reports
        print("\n📋 Uses Classification Report:")
        print(classification_report(y_uses_test, uses_pred, target_names=self.uses_encoder.classes_[:10]))
        
        print("\n📋 Side Effects Classification Report:")
        print(classification_report(y_side_effects_test, side_effects_pred, target_names=self.side_effects_encoder.classes_[:10]))
        
        # Save evaluation results
        self.save_evaluation_results(uses_accuracy, side_effects_accuracy, uses_pred, side_effects_pred, y_uses_test, y_side_effects_test)
        
        return uses_accuracy, side_effects_accuracy
    
    def save_evaluation_results(self, uses_accuracy, side_effects_accuracy, uses_pred, side_effects_pred, y_uses_true, y_side_effects_true):
        """Save evaluation results and create visualizations"""
        print("💾 Saving evaluation results...")
        
        # Create confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Uses confusion matrix
        cm_uses = confusion_matrix(y_uses_true, uses_pred)
        sns.heatmap(cm_uses, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Uses Confusion Matrix ({self.uses_model_name})\nAccuracy: {uses_accuracy:.4f}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Side effects confusion matrix
        cm_side_effects = confusion_matrix(y_side_effects_true, side_effects_pred)
        sns.heatmap(cm_side_effects, annot=True, fmt='d', cmap='Reds', ax=axes[1])
        axes[1].set_title(f'Side Effects Confusion Matrix ({self.side_effects_model_name})\nAccuracy: {side_effects_accuracy:.4f}')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('ml_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results to file
        with open('ml_evaluation_results.txt', 'w') as f:
            f.write("Medicine ML Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Uses Model: {self.uses_model_name}\n")
            f.write(f"Uses Prediction Accuracy: {uses_accuracy:.4f}\n")
            f.write(f"Best Side Effects Model: {self.side_effects_model_name}\n")
            f.write(f"Side Effects Prediction Accuracy: {side_effects_accuracy:.4f}\n")
            f.write(f"Average Accuracy: {(uses_accuracy + side_effects_accuracy) / 2:.4f}\n\n")
            f.write("Model Features:\n")
            f.write(f"- TF-IDF Vectorizer (max_features=2000, ngram_range=(1,2))\n")
            f.write(f"- Feature dimensions: {self.X.shape[1]}\n")
            f.write(f"- Training samples: {self.X.shape[0]}\n")
            f.write(f"- Unique uses categories: {len(self.uses_encoder.classes_)}\n")
            f.write(f"- Unique side effects categories: {len(self.side_effects_encoder.classes_)}\n")
        
        print("✅ Evaluation results saved!")
    
    def save_models(self):
        """Save the trained models and components"""
        print("💾 Saving models and components...")
        
        # Save models
        with open('best_uses_model.pickle', 'wb') as handle:
            pickle.dump(self.best_uses_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('best_side_effects_model.pickle', 'wb') as handle:
            pickle.dump(self.best_side_effects_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save TF-IDF vectorizer
        with open('ml_tfidf_vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.tfidf_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save encoders
        with open('ml_uses_encoder.pickle', 'wb') as handle:
            pickle.dump(self.uses_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('ml_side_effects_encoder.pickle', 'wb') as handle:
            pickle.dump(self.side_effects_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save model names
        with open('ml_model_names.pickle', 'wb') as handle:
            pickle.dump({
                'uses_model': self.uses_model_name,
                'side_effects_model': self.side_effects_model_name
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("✅ Models and components saved successfully!")
    
    def load_models(self):
        """Load the trained models and components"""
        print("📂 Loading saved models and components...")
        
        try:
            # Load models
            with open('best_uses_model.pickle', 'rb') as handle:
                self.best_uses_model = pickle.load(handle)
            
            with open('best_side_effects_model.pickle', 'rb') as handle:
                self.best_side_effects_model = pickle.load(handle)
            
            # Load TF-IDF vectorizer
            with open('ml_tfidf_vectorizer.pickle', 'rb') as handle:
                self.tfidf_vectorizer = pickle.load(handle)
            
            # Load encoders
            with open('ml_uses_encoder.pickle', 'rb') as handle:
                self.uses_encoder = pickle.load(handle)
            
            with open('ml_side_effects_encoder.pickle', 'rb') as handle:
                self.side_effects_encoder = pickle.load(handle)
            
            # Load model names
            with open('ml_model_names.pickle', 'rb') as handle:
                model_names = pickle.load(handle)
                self.uses_model_name = model_names['uses_model']
                self.side_effects_model_name = model_names['side_effects_model']
            
            print("✅ Models and components loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_medicine(self, medicine_name):
        """Predict uses and side effects for a given medicine name"""
        if self.best_uses_model is None or self.best_side_effects_model is None:
            print("❌ Models not loaded. Please train or load the models first.")
            return None
        
        # Preprocess the medicine name
        clean_name = self.preprocess_text(medicine_name)
        
        # Transform using TF-IDF
        features = self.tfidf_vectorizer.transform([clean_name])
        
        # Make predictions
        uses_pred = self.best_uses_model.predict(features)[0]
        side_effects_pred = self.best_side_effects_model.predict(features)[0]
        
        # Get confidence scores (probability estimates if available)
        try:
            uses_confidence = np.max(self.best_uses_model.predict_proba(features)[0])
            side_effects_confidence = np.max(self.best_side_effects_model.predict_proba(features)[0])
        except:
            uses_confidence = 1.0
            side_effects_confidence = 1.0
        
        # Get original text from dataset
        uses_text = self.df[self.df['Uses_Encoded'] == uses_pred]['Uses'].iloc[0] if len(self.df[self.df['Uses_Encoded'] == uses_pred]) > 0 else "Not found"
        side_effects_text = self.df[self.df['Side_Effects_Encoded'] == side_effects_pred]['Side_effects'].iloc[0] if len(self.df[self.df['Side_Effects_Encoded'] == side_effects_pred]) > 0 else "Not found"
        
        return {
            'medicine_name': medicine_name,
            'predicted_uses': uses_text,
            'uses_confidence': uses_confidence,
            'predicted_side_effects': side_effects_text,
            'side_effects_confidence': side_effects_confidence,
            'uses_model': self.uses_model_name,
            'side_effects_model': self.side_effects_model_name
        }

def main():
    """Main function to train and evaluate the ML model"""
    print("🚀 Medicine ML Model Training")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = MedicineMLPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Prepare data
    X, y_uses, y_side_effects = predictor.prepare_data()
    
    # Train models
    uses_accuracy, side_effects_accuracy = predictor.train_models()
    
    # Optional: Hyperparameter tuning
    predictor.hyperparameter_tuning()
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_medicines = ['Paracetamol', 'Aspirin', 'Ibuprofen', 'Omeprazole', 'Amoxicillin']
    
    for medicine in test_medicines:
        prediction = predictor.predict_medicine(medicine)
        if prediction:
            print(f"\n💊 {prediction['medicine_name']}:")
            print(f"   Uses: {prediction['predicted_uses'][:100]}...")
            print(f"   Uses Confidence: {prediction['uses_confidence']:.4f}")
            print(f"   Side Effects: {prediction['predicted_side_effects'][:100]}...")
            print(f"   Side Effects Confidence: {prediction['side_effects_confidence']:.4f}")
            print(f"   Uses Model: {prediction['uses_model']}")
            print(f"   Side Effects Model: {prediction['side_effects_model']}")
    
    print("\n✅ ML model training and evaluation completed!")

if __name__ == "__main__":
    main() 