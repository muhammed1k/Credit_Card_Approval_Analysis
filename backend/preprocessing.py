from config import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
import joblib

def Prepare_DATA(application_details,application_history,inference=False):
    copy = application_details.copy()
    #encoding gender column
    application_details['CODE_GENDER'] = application_details['CODE_GENDER'].map({'M':1,'F':0})

    #encoding Own car column
    application_details['FLAG_OWN_CAR'] = application_details['FLAG_OWN_CAR'].map({'Y':1,'N':0})
    application_details['FLAG_OWN_REALTY'] = application_details['FLAG_OWN_REALTY'].map({'Y':1,'N':0})

    #oridanl encoding education col
    education_order = {'Lower secondary':0, 'Secondary / secondary special':1, 
                    'Incomplete higher':2, 'Higher education':3, 'Academic degree':4}
    application_details['NAME_EDUCATION_TYPE'] = application_details['NAME_EDUCATION_TYPE'].map(education_order)
     
    # create age in years based on days birth
    application_details['Age'] = -application_details['DAYS_BIRTH'] // 365

    application_details[application_details['DAYS_EMPLOYED'] ==365243]

    application_details.loc[application_details['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = -application_details.loc[application_details['DAYS_EMPLOYED'] == 365243, 'DAYS_BIRTH'] 

    application_details['Employed_since'] = np.where(application_details['DAYS_EMPLOYED'] < 0,-application_details['DAYS_EMPLOYED']//365 , 0)
    application_details['UnEmployed_since'] = np.where(application_details['DAYS_EMPLOYED'] > 0,application_details['DAYS_EMPLOYED']//365 , 0)

    application_details['CNT_FAM_MEMBERS'] = application_details['CNT_FAM_MEMBERS'].astype(int)

    application_details['IS_CURR_EMPLOYED'] = application_details['DAYS_EMPLOYED'].map(lambda x : 0 if x>0 else 1)
    application_details = application_details.drop(['DAYS_EMPLOYED','DAYS_BIRTH'],axis=1)

    ##################

    mode_map = (copy.groupby(['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE'])['OCCUPATION_TYPE'].agg(
        lambda x:x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index().rename(columns={'OCCUPATION_TYPE':'mode'}))

    df = copy.merge(mode_map,on=['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE'],how='left')
    df['OCCUPATION_TYPE']= df['OCCUPATION_TYPE'].fillna(df['mode'])

    df = df.fillna('Unemployed')
    application_details['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE']
    
    cols_to_encode = ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']

    ohe_encoder = joblib.load('encoders/ohe_encoder.pkl')
    encoded_values = ohe_encoder.transform(application_details[cols_to_encode])
    # Convert the result to a DataFrame
    encoded_df = pd.DataFrame(encoded_values, columns=ohe_encoder.get_feature_names_out(cols_to_encode))
    application_details = pd.concat([application_details, encoded_df], axis=1)

    # Drop the original categorical columns if you no longer need them
    application_details = application_details.drop(cols_to_encode, axis=1)

    ##app history
    application_history['MONTHS_BALANCE'] = -application_history['MONTHS_BALANCE']

    status_map = {'C':0,'X':1,'0':2,'1':3,'2':4,'3':5,'4':6,'5':7}
    application_history['STATUS'] = application_history['STATUS'].map(status_map)
    stats_freq = application_history.groupby(['ID'])['STATUS'].agg(freq_status = (lambda x : x.mode()[0])).reset_index()

    application_history = application_history.merge(stats_freq,how='inner',on='ID')

    application_history['Risk'] = np.where(application_history['freq_status'] >= 2 , 1,0)

    status_weights = {
        0: 1,  # Completed Payment
        1: 0,  # No loans
        2: -1, # 0-29 days overdue
        3: -2, # 30-59 days overdue
        4: -3, # 60-89 days overdue
        5: -4, # 90+ days overdue
        6: -5, # 90-120 days overdue
        7: -6    # writeoffs - bad depts
    }

    application_history['status_weight'] = application_history['STATUS'].map(status_weights)

    application_history['decay_factor'] = np.exp(-application_history['MONTHS_BALANCE'] / 60)
    application_history['credit_score'] = application_history['status_weight'] * application_history['decay_factor']

    credit_scores = application_history.groupby('ID').agg(
        total_scores = ('credit_score','sum'), 
    ).reset_index()

    application_history = application_history.merge(credit_scores,how='left',on='ID').drop_duplicates(subset=['ID'],keep='first')

    application_history= application_history.drop(['MONTHS_BALANCE','STATUS','freq_status','status_weight','decay_factor','credit_score'],axis=1).rename(columns={'total_scores':'credit_score'})


    print('dsadsadsadsadsadsa')
    print(application_details.head())
    print('dsadsadsadasd')
    print(application_history.head())
    minmax_scaler = joblib.load('encoders/minmax_encoder.pkl')
    credit_scores_scaled = minmax_scaler.transform(application_history['credit_score'].values.reshape(-1,1)).astype(int)

    application_history['credit_score'] = credit_scores_scaled
    data = application_details.merge(application_history,how='inner',on='ID')
   
    X = data.drop(['ID','Risk'],axis=1)
    y = data['Risk'].values

    if inference:
        return X

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.8,random_state=42)

    cols_to_scale = ['CNT_CHILDREN','AMT_INCOME_TOTAL','Age', 'Employed_since', 'UnEmployed_since','credit_score']

    std_scaler = joblib.load('encoders/std_scaler.pkl')
    X_train[cols_to_scale] = std_scaler.transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = std_scaler.transform(X_test[cols_to_scale])

    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

    return X_train_scaled,X_test_scaled,y_train,y_test

