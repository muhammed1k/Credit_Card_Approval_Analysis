import pandas as pd

#loading data Files
def load_data(path, file_names):
    application_details = pd.read_csv(path / file_names[0])
    application_history = pd.read_csv(path / file_names[1])
    return application_details, application_history



config = {'input_path':r'data','output_path':r'saved_model','file_names':['application_record.csv','credit_record.csv']}