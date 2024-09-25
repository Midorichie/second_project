import unittest
import pandas as pd
import numpy as np
from drug_interaction_predictor import DrugInteractionPredictor

class TestDrugInteractionPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = DrugInteractionPredictor()
        self.sample_data = pd.DataFrame({
            'drug1': ['DrugA', 'DrugB', 'DrugC', 'DrugA', 'DrugB'],
            'drug2': ['DrugB', 'DrugC', 'DrugD', 'DrugC', 'DrugD'],
            'interaction': [1, 0, 1, 0, 1]
        })

    def test_preprocess_data(self):
        processed_data = self.predictor.preprocess_data(self.sample_data[['drug1', 'drug2']])
        self.assertTrue(processed_data.dtypes.all() != 'object')
        self.assertEqual(len(processed_data), len(self.sample_data))

    def test_train(self):
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        X_processed = self.predictor.preprocess_data(X)
        self.predictor.train(X_processed, y)
        self.assertIsNotNone(self.predictor.model)

    def test_predict_interaction(self):
        # First, train the model
        X = self.sample_data[['drug1', 'drug2']]
        y = self.sample_data['interaction']
        X_processed = self.predictor.preprocess_data(X)
        self.predictor.train(X_processed, y)

        # Now test prediction
        result = self.predictor.predict_interaction('DrugA', 'DrugB')
        self.assertIn(result, ['Interaction', 'No Interaction'])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_interaction('', 'DrugB')
        with self.assertRaises(ValueError):
            self.predictor.predict_interaction('DrugA', '')

if __name__ == '__main__':
    unittest.main()