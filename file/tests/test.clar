import unittest
import pandas as pd
import numpy as np
from drug_interaction_predictor import DrugInteractionPredictor
import matplotlib.pyplot as plt

class TestDrugInteractionPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = DrugInteractionPredictor()
        self.sample_data = pd.DataFrame({
            'drug1': ['DrugA', 'DrugB', 'DrugC', 'DrugA', 'DrugB', 'DrugX'] * 10,
            'drug2': ['DrugB', 'DrugC', 'DrugD', 'DrugC', 'DrugD', 'DrugY'] * 10,
            'interaction': [1, 0, 1, 0, 1, 0] * 10
        })

    def test_preprocess_data(self):
        processed_data = self.predictor.preprocess_data(self.sample_data[['drug1', 'drug2']])
        self.assertTrue(processed_data.dtypes.all() != 'object')
        self.assertEqual(len(processed_data), len(self.sample_data))
        self.assertFalse(processed_data.isnull().any().any())

    def test_extract_features(self):
        processed_data = self.predictor.preprocess_data(self.sample_data[['drug1', 'drug2']])
        features = self.predictor.extract_features(processed_data)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], len(self.sample_data))
        self.assertIsInstance(features, type(self.predictor.feature_extractor.transform(processed_data)))

    def test_train(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.feature_names)
        self.assertIsInstance(self.predictor.model.named_steps['classifier'], type(self.predictor.model.named_steps['classifier']))

    def test_predict_interaction(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)

        result, confidence = self.predictor.predict_interaction('DrugA', 'DrugB')
        self.assertIn(result, ['Interaction', 'No Interaction'])
        self.assertTrue(0 <= confidence <= 1)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_interaction('', 'DrugB')
        with self.assertRaises(ValueError):
            self.predictor.predict_interaction('DrugA', '')

    def test_get_feature_importances(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        
        importances = self.predictor.get_feature_importances()
        self.assertIsNotNone(importances)
        self.assertTrue(len(importances) > 0)
        self.assertIsInstance(importances, pd.DataFrame)
        self.assertTrue('feature' in importances.columns)
        self.assertTrue('importance' in importances.columns)

    def test_missing_value_handling(self):
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'drug1'] = np.nan
        processed_data = self.predictor.preprocess_data(data_with_missing[['drug1', 'drug2']])
        self.assertFalse(processed_data.isnull().any().any())

    def test_prediction_on_unseen_drugs(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        
        result, confidence = self.predictor.predict_interaction('NewDrug1', 'NewDrug2')
        self.assertIn(result, ['Interaction', 'No Interaction'])
        self.assertTrue(0 <= confidence <= 1)

    def test_model_persistence(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        
        # Save the model
        self.predictor.save_model('test_model.joblib')
        
        # Load the model
        loaded_predictor = DrugInteractionPredictor.load_model('test_model.joblib')
        
        # Test the loaded model
        result, confidence = loaded_predictor.predict_interaction('DrugA', 'DrugB')
        self.assertIn(result, ['Interaction', 'No Interaction'])
        self.assertTrue(0 <= confidence <= 1)

    def test_feature_importance_plot(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        
        # Call get_feature_importances to generate the plot
        self.predictor.get_feature_importances()
        
        # Check if a plot was created
        self.assertTrue(plt.fignum_exists(1))
        plt.close()  # Close the plot to free up memory

if __name__ == '__main__':
    unittest.main()