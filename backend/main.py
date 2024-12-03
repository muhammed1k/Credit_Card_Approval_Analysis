from flask import Flask,jsonify,request
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import argparse
from models import *
from config import *
from preprocessing import *
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR logs

app = Flask(__name__)
CORS(app)

def clean_data_types(df):
    floatcols= ['AMT_INCOME_TOTAL','CNT_FAM_MEMBERS']
    intcols = ['ID','CNT_CHILDREN','DAYS_BIRTH','DAYS_EMPLOYED','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL']
    for col in floatcols:
        df[col] = df[col].astype(np.float32)
    for col in intcols:
        df[col] = df[col].astype(np.int32)

    return df
        

def main(train_model=False,output_path=None,input_path=None,file_names=None):
    
    input_path = Path(input_path)
    application_details, application_history = load_data(input_path, file_names)
    risk_predictor = RiskPredictor()
    print("Data successfully loaded:")

    if train_model:
        X_train_scaled,X_test_scaled,y_train,y_test = Prepare_DATA(application_details, application_history )
        print("Training model...")
        risk_predictor.train_model(X_train_scaled,y_train)
        print("Evaluating model...")
        risk_predictor.eval_model(X_test_scaled,y_test)

        if output_path:
            print(f"Saving model to {output_path}...")
            risk_predictor.save_model(Path(output_path))
    

    @app.route('/predict_risk',methods=['POST'])
    def predict_risk():
        applicant_form = request.json

        if not applicant_form:
            return jsonify({'error': 'Invalid input, please provide applicant details.'}),

        applicant_data = pd.DataFrame([applicant_form])
        applicant_data = clean_data_types(applicant_data)

        applicant_history = application_history[application_history['ID'] == applicant_data['ID'].values[0]]
        X = Prepare_DATA(applicant_data, applicant_history,inference=True)
        print(X.shape)
        print('Loading Model....')
        risk_predictor.load_model(Path(output_path))
        print('Predicting Risk')

        probability,risk = risk_predictor.run_inference(X)
        
        if risk == 'Low Risk':
            status = 'Accepted'
            probability = 1 - probability
        else:
            status = 'Rejected'
            
        return jsonify({
            'status': f'Your Credit Card Application Was {status}',
            'risk_level': f'Applicant is {risk} with Probability of {probability:.2f}'
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Risk Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--output_location', type=str, help='Output location to save model',default=r'saved_model')
    parser.add_argument('--train_data', type=str, help='Path to the training data',default=r'data')
    parser.add_argument('--file_names', nargs='+', help='List of file names (space-separated)',
                                        default=['application_record.csv','credit_record.csv'])
    args = parser.parse_args()

    main(
        train_model=args.train,
        output_path=args.output_location,
        input_path=args.train_data,
        file_names=args.file_names
    )
    app.run(host='localhost',port=9897,debug=True)

else:
    main()