import unittest
import pandas as pd
import numpy as np
from drug_interaction_predictor import DrugInteractionPredictor

class TestDrugInteractionPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = DrugInteractionPredictor()
        self.sample_data = pd.DataFrame({
            'drug1': ['DrugA', 'DrugB', 'DrugC', 'DrugA', 'DrugB', 'DrugX'],
            'drug2': ['DrugB', 'DrugC', 'DrugD', 'DrugC', 'DrugD', 'DrugY'],
            'interaction': [1, 0, 1, 0, 1, 0]
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

    def test_train(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.feature_names)

    def test_predict_interaction(self):
        # First, train the model
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)

        # Now test prediction
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
        self.assertTrue(all(isinstance(importance, tuple) for importance in importances))

    def test_missing_value_handling(self):
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'drug1'] = np.nan
        processed_data = self.predictor.preprocess_data(data_with_missing[['drug1', 'drug2']])
        self.assertFalse(processed_data.isnull().any().any())

    def test_cross_validation(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        # We can't directly test the cross-validation results, but we can ensure it runs without errors

    def test_prediction_on_unseen_drugs(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        self.predictor.train(X, y)
        
        result, confidence = self.predictor.predict_interaction('NewDrug1', 'NewDrug2')
        self.assertIn(result, ['Interaction', 'No Interaction'])
        self.assertTrue(0 <= confidence <= 1)

if __name__ == '__main__':
    unittest.main()