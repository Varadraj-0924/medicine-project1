import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Attention, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class MedicinePredictor:
    def __init__(self):
        self.df = None
        self.tokenizer = Tokenizer()
        self.max_sequence_length = 50
        self.vocab_size = 1000
        self.model = None
        self.uses_encoder = LabelEncoder()
        self.side_effects_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_data(self, csv_file='Medicine_Details.csv'):
        """Load and preprocess the medicine dataset"""
        print("📊 Loading medicine dataset...")
        self.df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(self.df)} medicines")
        
        # Clean the data
        self.df = self.df.dropna(subset=['Medicine Name', 'Uses', 'Side_effects'])
        self.df = self.df[self.df['Uses'].str.len() > 10]  # Remove very short descriptions
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
        self.tfidf_features = self.tfidf_vectorizer.fit_transform(medicine_names)
        
        # Encode uses and side effects
        self.df['Uses_Encoded'] = self.uses_encoder.fit_transform(self.df['Uses_Clean'])
        self.df['Side_Effects_Encoded'] = self.side_effects_encoder.fit_transform(self.df['Side_Effects_Clean'])
        
        # Tokenize medicine names
        self.tokenizer.fit_on_texts(self.df['Medicine_Name_Clean'])
        sequences = self.tokenizer.texts_to_sequences(self.df['Medicine_Name_Clean'])
        self.X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Prepare labels
        self.y_uses = self.df['Uses_Encoded'].values
        self.y_side_effects = self.df['Side_Effects_Encoded'].values
        
        print(f"📝 Vocabulary size: {len(self.tokenizer.word_index) + 1}")
        print(f"🏷️ Number of unique uses: {len(self.uses_encoder.classes_)}")
        print(f"🏷️ Number of unique side effects: {len(self.side_effects_encoder.classes_)}")
        
        return self.X, self.y_uses, self.y_side_effects
    
    def create_model(self):
        """Create the deep learning model"""
        print("🤖 Creating deep learning model...")
        
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding_layer = Embedding(
            input_dim=len(self.tokenizer.word_index) + 1,
            output_dim=128,
            input_length=self.max_sequence_length
        )(input_layer)
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(embedding_layer)
        lstm2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(lstm1)
        
        # Dense layers
        dense1 = Dense(256, activation='relu')(lstm2)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        # Output layers
        uses_output = Dense(len(self.uses_encoder.classes_), activation='softmax', name='uses')(dropout2)
        side_effects_output = Dense(len(self.side_effects_encoder.classes_), activation='softmax', name='side_effects')(dropout2)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=[uses_output, side_effects_output])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully!")
        return self.model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Split the data
        X_train, X_test, y_uses_train, y_uses_test, y_side_effects_train, y_side_effects_test = train_test_split(
            self.X, self.y_uses, self.y_side_effects, test_size=0.2, random_state=42
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        history = self.model.fit(
            X_train,
            [y_uses_train, y_side_effects_train],
            validation_data=(X_test, [y_uses_test, y_side_effects_test]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate the model
        self.evaluate_model(X_test, y_uses_test, y_side_effects_test)
        
        # Save the model and components
        self.save_model()
        
        return history
    
    def evaluate_model(self, X_test, y_uses_test, y_side_effects_test):
        """Evaluate the model performance"""
        print("\n📊 Model Evaluation:")
        
        # Predictions
        uses_pred, side_effects_pred = self.model.predict(X_test)
        uses_pred_classes = np.argmax(uses_pred, axis=1)
        side_effects_pred_classes = np.argmax(side_effects_pred, axis=1)
        
        # Uses accuracy
        uses_accuracy = accuracy_score(y_uses_test, uses_pred_classes)
        print(f"🎯 Uses Prediction Accuracy: {uses_accuracy:.4f}")
        
        # Side effects accuracy
        side_effects_accuracy = accuracy_score(y_side_effects_test, side_effects_pred_classes)
        print(f"🎯 Side Effects Prediction Accuracy: {side_effects_accuracy:.4f}")
        
        # Detailed classification reports
        print("\n📋 Uses Classification Report:")
        print(classification_report(y_uses_test, uses_pred_classes, target_names=self.uses_encoder.classes_[:10]))
        
        print("\n📋 Side Effects Classification Report:")
        print(classification_report(y_side_effects_test, side_effects_pred_classes, target_names=self.side_effects_encoder.classes_[:10]))
        
        # Save evaluation results
        self.save_evaluation_results(uses_accuracy, side_effects_accuracy, uses_pred_classes, side_effects_pred_classes, y_uses_test, y_side_effects_test)
        
        return uses_accuracy, side_effects_accuracy
    
    def save_evaluation_results(self, uses_accuracy, side_effects_accuracy, uses_pred, side_effects_pred, y_uses_true, y_side_effects_true):
        """Save evaluation results and create visualizations"""
        print("💾 Saving evaluation results...")
        
        # Create confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Uses confusion matrix
        cm_uses = confusion_matrix(y_uses_true, uses_pred)
        sns.heatmap(cm_uses, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Uses Confusion Matrix (Accuracy: {uses_accuracy:.4f})')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Side effects confusion matrix
        cm_side_effects = confusion_matrix(y_side_effects_true, side_effects_pred)
        sns.heatmap(cm_side_effects, annot=True, fmt='d', cmap='Reds', ax=axes[1])
        axes[1].set_title(f'Side Effects Confusion Matrix (Accuracy: {side_effects_accuracy:.4f})')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results to file
        with open('evaluation_results.txt', 'w') as f:
            f.write("Medicine Prediction Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Uses Prediction Accuracy: {uses_accuracy:.4f}\n")
            f.write(f"Side Effects Prediction Accuracy: {side_effects_accuracy:.4f}\n")
            f.write(f"Average Accuracy: {(uses_accuracy + side_effects_accuracy) / 2:.4f}\n\n")
            f.write("Model Architecture:\n")
            f.write("- Embedding Layer (128 dimensions)\n")
            f.write("- Bidirectional LSTM (128 units)\n")
            f.write("- Bidirectional LSTM (64 units)\n")
            f.write("- Dense Layers (256, 128 units)\n")
            f.write("- Dropout (0.3)\n")
            f.write("- Output Layers (Softmax)\n")
        
        print("✅ Evaluation results saved!")
    
    def save_model(self):
        """Save the trained model and components"""
        print("💾 Saving model and components...")
        
        # Save the model
        self.model.save('medicine_prediction_model.h5')
        
        # Save tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save encoders
        with open('uses_encoder.pickle', 'wb') as handle:
            pickle.dump(self.uses_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('side_effects_encoder.pickle', 'wb') as handle:
            pickle.dump(self.side_effects_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save TF-IDF vectorizer
        with open('tfidf_vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.tfidf_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("✅ Model and components saved successfully!")
    
    def load_model(self):
        """Load the trained model and components"""
        print("📂 Loading saved model and components...")
        
        try:
            # Load the model
            self.model = tf.keras.models.load_model('medicine_prediction_model.h5')
            
            # Load tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            
            # Load encoders
            with open('uses_encoder.pickle', 'rb') as handle:
                self.uses_encoder = pickle.load(handle)
            
            with open('side_effects_encoder.pickle', 'rb') as handle:
                self.side_effects_encoder = pickle.load(handle)
            
            # Load TF-IDF vectorizer
            with open('tfidf_vectorizer.pickle', 'rb') as handle:
                self.tfidf_vectorizer = pickle.load(handle)
            
            print("✅ Model and components loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_medicine(self, medicine_name):
        """Predict uses and side effects for a given medicine name"""
        if self.model is None:
            print("❌ Model not loaded. Please train or load the model first.")
            return None
        
        # Preprocess the medicine name
        clean_name = self.preprocess_text(medicine_name)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([clean_name])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
        
        # Make prediction
        uses_pred, side_effects_pred = self.model.predict(padded_sequence)
        
        # Get predicted classes
        uses_class = np.argmax(uses_pred[0])
        side_effects_class = np.argmax(side_effects_pred[0])
        
        # Get confidence scores
        uses_confidence = np.max(uses_pred[0])
        side_effects_confidence = np.max(side_effects_pred[0])
        
        # Get original text from dataset
        uses_text = self.df[self.df['Uses_Encoded'] == uses_class]['Uses'].iloc[0] if len(self.df[self.df['Uses_Encoded'] == uses_class]) > 0 else "Not found"
        side_effects_text = self.df[self.df['Side_Effects_Encoded'] == side_effects_class]['Side_effects'].iloc[0] if len(self.df[self.df['Side_Effects_Encoded'] == side_effects_class]) > 0 else "Not found"
        
        return {
            'medicine_name': medicine_name,
            'predicted_uses': uses_text,
            'uses_confidence': uses_confidence,
            'predicted_side_effects': side_effects_text,
            'side_effects_confidence': side_effects_confidence
        }

def main():
    """Main function to train and evaluate the model"""
    print("🚀 Medicine Prediction Model Training")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = MedicinePredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Prepare data
    X, y_uses, y_side_effects = predictor.prepare_data()
    
    # Create and train model
    model = predictor.create_model()
    history = predictor.train_model(epochs=30, batch_size=32)
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_medicines = ['Paracetamol', 'Aspirin', 'Ibuprofen', 'Omeprazole']
    
    for medicine in test_medicines:
        prediction = predictor.predict_medicine(medicine)
        if prediction:
            print(f"\n💊 {prediction['medicine_name']}:")
            print(f"   Uses: {prediction['predicted_uses'][:100]}...")
            print(f"   Uses Confidence: {prediction['uses_confidence']:.4f}")
            print(f"   Side Effects: {prediction['predicted_side_effects'][:100]}...")
            print(f"   Side Effects Confidence: {prediction['side_effects_confidence']:.4f}")
    
    print("\n✅ Model training and evaluation completed!")

if __name__ == "__main__":
    main() 