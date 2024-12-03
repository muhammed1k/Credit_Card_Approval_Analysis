from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from preprocessing import *

class RiskPredictor():
    def __init__(self):
        self.model = RandomForestClassifier(max_depth=33, max_features=None, min_samples_leaf= 8, min_samples_split= 6, n_estimators=1373,n_jobs=-1)

    def train_model(self,features,labels):
        print('Training Model....')
        self.model.fit(features,labels)

    def eval_model(self,features,labels):
        if self.model is None:
            print("No model loaded or trained. Please train or load a model first.")
        else:
            print('Evaluating Model....')
            probabilities = self.model.predict_proba(features)[:,1]
            predictions = np.where(probabilities > 0.5 , 1,0)
            score = accuracy_score(labels,predictions)
            print(f'Accuracy Score on Test Set = {score}')

    def run_inference(self,features):
        if self.model is None:
            print("No model loaded or trained. Please train or load a model first.")
        else:
            print('Predicting Application Risk....')
            probabilities = self.model.predict_proba(features)[:,1][0]
            risk = np.where(probabilities > 0.5 , 'High Risk','Low Risk').tolist()
            print(f'Applicant is a {risk} with Probability = {probabilities}')

        return probabilities,risk

    def save_model(self, file_path):
        if self.model is None:
            print("No model found to save. Train the model first.")
        else:
            joblib.dump(self.model, file_path/'model.pkl')
            print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        try:
            self.model = joblib.load(file_path/'model.pkl')
            print(f"Model loaded from {file_path}")
        except FileNotFoundError:
            print(f"Model file not found at {file_path}. Train the model first.")



