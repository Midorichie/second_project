import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class DrugNameFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2,4))

    def fit(self, X, y=None):
        self.tfidf.fit(X.values.ravel())
        return self

    def transform(self, X):
        return self.tfidf.transform(X.values.ravel())

class DrugInteractionPredictor:
    def __init__(self):
        self.model = None
        self.le_drug1 = LabelEncoder()
        self.le_drug2 = LabelEncoder()
        self.feature_names = None
        self.feature_extractor = None

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
        self.feature_extractor = ColumnTransformer([
            ('drug1', DrugNameFeaturizer(), 'drug1'),
            ('drug2', DrugNameFeaturizer(), 'drug2')
        ])
        
        # Fit and transform the data
        X_features = self.feature_extractor.fit_transform(data)
        self.feature_names = self.feature_extractor.get_feature_names_out()
        
        return X_features

    def train(self, X, y):
        X_processed = self.preprocess_data(X)
        X_features = self.extract_features(X_processed)
        
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
        
        # Create model pipeline
        model_pipeline = Pipeline([
            ('classifier', XGBClassifier(random_state=42))
        ])

        # Define hyperparameters for grid search
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.3]
        }

        # Perform grid search
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get best model
        self.model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print(f"ROC AUC Score: {roc_auc:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_features, y, cv=5, scoring='roc_auc')
        print(f"\nCross-validation ROC AUC scores: {cv_scores}")
        print(f"Mean CV ROC AUC score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    def predict_interaction(self, drug1, drug2):
        if not drug1 or not drug2:
            raise ValueError("Both drug names must be non-empty strings.")
        
        input_data = pd.DataFrame([[drug1, drug2]], columns=['drug1', 'drug2'])
        processed_input = self.preprocess_data(input_data)
        input_features = self.feature_extractor.transform(processed_input)
        
        prediction = self.model.predict(input_features)
        probability = self.model.predict_proba(input_features)[0]
        
        result = "Interaction" if prediction[0] == 1 else "No Interaction"
        confidence = probability[1] if prediction[0] == 1 else probability[0]
        
        return result, confidence

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        feature_importance = self.model.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return importance_df

    def save_model(self, filename):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename):
        return joblib.load(filename)

# Example usage
if __name__ == "__main__":
    # Load data (this is a placeholder - you'd need real data)
    data = pd.DataFrame({
        'drug1': ['DrugA', 'DrugB', 'DrugC', 'DrugA', 'DrugB', 'DrugX'] * 100,
        'drug2': ['DrugB', 'DrugC', 'DrugD', 'DrugC', 'DrugD', 'DrugY'] * 100,
        'interaction': [1, 0, 1, 0, 1, 0] * 100
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
    predictor.get_feature_importances()
    
    # Save the model
    predictor.save_model('drug_interaction_model.joblib')
    
    # Load the model
    loaded_predictor = DrugInteractionPredictor.load_model('drug_interaction_model.joblib')
    
    # Test the loaded model
    result, confidence = loaded_predictor.predict_interaction(drug1, drug2)
    print(f"\nPredicted interaction using loaded model between {drug1} and {drug2}: {result} (Confidence: {confidence:.2f})")