import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer

class DrugInteractionPredictor:
    def __init__(self):
        self.model = None
        self.le_drug1 = LabelEncoder()
        self.le_drug2 = LabelEncoder()
        self.feature_names = None

    def preprocess_data(self, data):
        # Handle missing values
        imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        # Encode drug names
        data_imputed['drug1'] = self.le_drug1.fit_transform(data_imputed['drug1'])
        data_imputed['drug2'] = self.le_drug2.fit_transform(data_imputed['drug2'])
        
        return data_imputed

    def extract_features(self, data):
        # Create feature extraction pipeline
        feature_extractor = ColumnTransformer([
            ('drug1', CountVectorizer(analyzer='char', ngram_range=(2,3)), 'drug1'),
            ('drug2', CountVectorizer(analyzer='char', ngram_range=(2,3)), 'drug2')
        ])
        
        # Fit and transform the data
        X_features = feature_extractor.fit_transform(data)
        self.feature_names = feature_extractor.get_feature_names()
        
        return X_features

    def train(self, X, y):
        X_processed = self.preprocess_data(X)
        X_features = self.extract_features(X_processed)
        
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
        
        # Create and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_features, y, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    def predict_interaction(self, drug1, drug2):
        if not drug1 or not drug2:
            raise ValueError("Both drug names must be non-empty strings.")
        
        input_data = pd.DataFrame([[drug1, drug2]], columns=['drug1', 'drug2'])
        processed_input = self.preprocess_data(input_data)
        input_features = self.extract_features(processed_input)
        
        prediction = self.model.predict(input_features)
        probability = self.model.predict_proba(input_features)[0]
        
        result = "Interaction" if prediction[0] == 1 else "No Interaction"
        confidence = probability[1] if prediction[0] == 1 else probability[0]
        
        return result, confidence

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        importances = self.model.feature_importances_
        feature_importance = sorted(zip(importances, self.feature_names), reverse=True)
        return feature_importance[:10]  # Return top 10 features

# Example usage
if __name__ == "__main__":
    # Load data (this is a placeholder - you'd need real data)
    data = pd.DataFrame({
        'drug1': ['DrugA', 'DrugB', 'DrugC', 'DrugA', 'DrugB', 'DrugX'],
        'drug2': ['DrugB', 'DrugC', 'DrugD', 'DrugC', 'DrugD', 'DrugY'],
        'interaction': [1, 0, 1, 0, 1, 0]
    })
    
    predictor = DrugInteractionPredictor()
    
    X = data[['drug1', 'drug2']]
    y = data['interaction']
    
    predictor.train(X, y)
    
    # Example prediction
    drug1, drug2 = 'DrugA', 'DrugD'
    result, confidence = predictor.predict_interaction(drug1, drug2)
    print(f"\nPredicted interaction between {drug1} and {drug2}: {result} (Confidence: {confidence:.2f})")
    
    # Print top features
    print("\nTop 10 important features:")
    for importance, feature in predictor.get_feature_importances():
        print(f"{feature}: {importance:.4f}")