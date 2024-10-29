import streamlit as st 
import pandas as pd
import numpy as np
import joblib

models = {
    'Linear Regression': joblib.load('linear_regression_model.joblib'),
    'Ridge Regression': joblib.load('ridge_regression_model.joblib'),
    'Neural Network': joblib.load('mlp_regressor_model.joblib'),
    'Stacking Model': joblib.load('stacking_regressor_model.joblib')
}

data = pd.read_csv('housing.csv')

data = data.dropna()
data = pd.get_dummies(data, columns=['ocean_proximity'])
feature_columns = data.drop('median_house_value', axis=1).columns.tolist()

numeric_features = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
categorical_features = ['ocean_proximity']

st.title('🏠 Dự đoán Giá nhà California')

st.write('Nhập thông tin bên dưới để dự đoán giá nhà:')

input_data = {}

for feature in numeric_features:
    input_value = st.text_input(f'Nhập {feature}', '')
    input_data[feature] = input_value

input_data['ocean_proximity'] = st.selectbox(
    'Chọn ocean_proximity',
    options=data.columns[data.columns.str.startswith('ocean_proximity_')].str.replace('ocean_proximity_', ''),
    index=0
)

model_name = st.selectbox(
    'Chọn mô hình để dự đoán',
    options=list(models.keys()),
    index=3 
)

if st.button('Dự đoán'):
    input_df = pd.DataFrame([input_data])

    for col in numeric_features:
        if input_df[col][0] == '':
            input_df[col] = data[col].median()
        else:
            try:
                input_df[col] = float(input_df[col][0])
            except ValueError:
                st.error(f'Giá trị nhập vào cho {col} không hợp lệ. Vui lòng nhập số.')
                st.stop()

    ocean_proximity_dummies = pd.get_dummies(input_df['ocean_proximity'], prefix='ocean_proximity')
    input_df = pd.concat([input_df[numeric_features], ocean_proximity_dummies], axis=1)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  

    input_df = input_df[feature_columns] 

    selected_model = models[model_name]

    prediction = selected_model.predict(input_df)[0]

    st.success(f'Mô hình sử dụng: **{model_name}**')
    st.success(f'Giá nhà dự đoán: **${prediction:,.2f}**')
