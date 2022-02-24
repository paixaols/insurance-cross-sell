# -*- coding: utf-8 -*-
import pickle

class HealthInsurance(object):
    def __init__(self):
        self.home = ''
        self.age_scaler = pickle.load(open(self.home+'features/age_scaler.pkl', 'rb'))
        self.annual_premium_scaler = pickle.load(open(self.home+'features/annual_premium_scaler.pkl', 'rb'))
        self.vintage_scaler = pickle.load(open(self.home+'features/vintage_scaler.pkl', 'rb'))
        self.gender_encoder = pickle.load(open(self.home+'features/gender_encoder.pkl', 'rb'))
        self.region_encoder = pickle.load(open(self.home+'features/region_encoder.pkl', 'rb'))
        self.sales_encoder = pickle.load(open(self.home+'features/sales_encoder.pkl', 'rb'))
    
    def clean_data(self, df):
        return df
    
    def feature_engineering(self, df):
        df['vehicle_age'] = df['vehicle_age'].apply(
            lambda x: 1 if x == '< 1 Year' else 2 if x == '1-2 Year' else 3)
        df['vehicle_damage'] = df['vehicle_damage'].apply(
            lambda x: 0 if x == 'No' else 1)
        return df
    
    def data_preparation(self, df):
        # Rescaling
        df['annual_premium'] = self.annual_premium_scaler.transform(df[['annual_premium']].values)
        df['vintage'] = self.vintage_scaler.transform(df[['vintage']].values)
        df['age'] = self.age_scaler.transform(df[['age']].values)
        
        # Encoding
        df.loc[:, 'region_code'] = df['region_code'].map(self.region_encoder)
        df.loc[:, 'policy_sales_channel'] = df['policy_sales_channel'].map(self.sales_encoder)
        df.loc[:, 'gender'] = df['gender'].map(self.gender_encoder)
        
        selected_features = ['vintage', 'annual_premium', 'age', 
                             'region_code', 'policy_sales_channel']
        return df[selected_features]
    
    def prediction(self, model, original_data, test_data):
        # Prediction
        pred = model.predict_proba(test_data)
        
        # Join original data and prediction
        original_data['score'] = [ b for a, b in pred ]
        
        return original_data.to_json(orient = 'records', date_format = 'iso')
