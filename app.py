import streamlit as st 
import pandas as pd
import numpy as np
import joblib

# 1. Tải các mô hình đã lưu
models = {
    'Linear Regression': joblib.load('linear_regression_model.joblib'),
    'Ridge Regression': joblib.load('ridge_regression_model.joblib'),
    'Neural Network': joblib.load('mlp_regressor_model.joblib'),
    'Stacking Model': joblib.load('stacking_regressor_model.joblib')
}

# 2. Đọc dữ liệu gốc để lấy giá trị trung vị cho các cột số và lấy danh sách các cột
data = pd.read_csv('housing.csv')

# Lưu danh sách các cột sau khi đã thực hiện get_dummies trong quá trình huấn luyện
data = data.dropna()
data = pd.get_dummies(data, columns=['ocean_proximity'])
feature_columns = data.drop('median_house_value', axis=1).columns.tolist()

# 3. Lấy danh sách các đặc trưng
numeric_features = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
categorical_features = ['ocean_proximity']

# 4. Tạo giao diện người dùng
st.title('🏠 Dự đoán Giá nhà California')

st.write('Nhập thông tin bên dưới để dự đoán giá nhà:')

# 5. Tạo các input cho người dùng
input_data = {}

# Nhập các đặc trưng số
for feature in numeric_features:
    input_value = st.text_input(f'Nhập {feature}', '')
    input_data[feature] = input_value

# Nhập đặc trưng phân loại
input_data['ocean_proximity'] = st.selectbox(
    'Chọn ocean_proximity',
    options=data.columns[data.columns.str.startswith('ocean_proximity_')].str.replace('ocean_proximity_', ''),
    index=0
)

# Lựa chọn mô hình
model_name = st.selectbox(
    'Chọn mô hình để dự đoán',
    options=list(models.keys()),
    index=3  # Mặc định chọn Stacking Model
)

# 6. Khi người dùng nhấn nút 'Dự đoán'
if st.button('Dự đoán'):
    # Chuyển đổi input_data thành DataFrame
    input_df = pd.DataFrame([input_data])

    # Xử lý các ô không nhập dữ liệu
    # Chuyển các giá trị số từ chuỗi sang số thực
    for col in numeric_features:
        if input_df[col][0] == '':
            # Sử dụng giá trị trung vị nếu người dùng không nhập
            input_df[col] = data[col].median()
        else:
            try:
                input_df[col] = float(input_df[col][0])
            except ValueError:
                st.error(f'Giá trị nhập vào cho {col} không hợp lệ. Vui lòng nhập số.')
                st.stop()

    # One-hot encoding cho 'ocean_proximity'
    ocean_proximity_dummies = pd.get_dummies(input_df['ocean_proximity'], prefix='ocean_proximity')
    input_df = pd.concat([input_df[numeric_features], ocean_proximity_dummies], axis=1)

    # Đảm bảo rằng các cột trong input_df khớp với các cột trong quá trình huấn luyện
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Thêm cột thiếu với giá trị 0

    input_df = input_df[feature_columns]  # Sắp xếp lại thứ tự các cột

    # Lấy mô hình được chọn
    selected_model = models[model_name]

    # Dự đoán
    prediction = selected_model.predict(input_df)[0]

    # Hiển thị kết quả
    st.success(f'Mô hình sử dụng: **{model_name}**')
    st.success(f'Giá nhà dự đoán: **${prediction:,.2f}**')
