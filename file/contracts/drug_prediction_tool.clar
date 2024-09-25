import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class DrugInteractionPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.le = LabelEncoder()
        
    def preprocess_data(self, data):
        # Encode categorical variables
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = self.le.fit_transform(data[column])
        return data
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def predict_interaction(self, drug1, drug2):
        # This is a placeholder. In a real scenario, we'd need to process these drugs
        # into the same format as our training data.
        input_data = pd.DataFrame([[drug1, drug2]], columns=['drug1', 'drug2'])
        processed_input = self.preprocess_data(input_data)
        prediction = self.model.predict(processed_input)
        return "Interaction" if prediction[0] == 1 else "No Interaction"

# Example usage
if __name__ == "__main__":
    # Load data (this is a placeholder - you'd need real data)
    data = pd.DataFrame({
        'drug1': ['DrugA', 'DrugB', 'DrugC', 'DrugA', 'DrugB'],
        'drug2': ['DrugB', 'DrugC', 'DrugD', 'DrugC', 'DrugD'],
        'interaction': [1, 0, 1, 0, 1]
    })
    
    predictor = DrugInteractionPredictor()
    
    X = data[['drug1', 'drug2']]
    y = data['interaction']
    
    X_processed = predictor.preprocess_data(X)
    predictor.train(X_processed, y)
    
    # Example prediction
    result = predictor.predict_interaction('DrugA', 'DrugD')
    print(f"\nPredicted interaction between DrugA and DrugD: {result}")