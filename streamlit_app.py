import streamlit as st
import pandas as pd
import joblib

# Tải mô hình
model = joblib.load('model.pkl')

st.title("Stroke Prediction App")
st.markdown("## Nhập dữ liệu để dự đoán nguy cơ đột quỵ.")

# Nhập thông tin từ người dùng
age = st.slider("Tuổi", 0, 100, 50)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
avg_glucose_level = st.slider("Mức đường huyết trung bình", 50.0, 300.0, 100.0)

# Tạo DataFrame từ input
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'avg_glucose_level': [avg_glucose_level]
})

# Hiển thị dữ liệu nhập
st.write("**Dữ liệu nhập:**", input_data)

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    prediction = model.predict(input_data)
    st.success(f"Nguy cơ đột quỵ: {'Có' if prediction[0] == 1 else 'Không'}")
