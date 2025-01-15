import streamlit as st
import pandas as pd
import joblib

# Tải mô hình
model = joblib.load('app_model.pkl')

# Giá trị min và max của tuổi từ quá trình huấn luyện
AGE_MIN = 0  # Thay bằng giá trị thực tế của min tuổi trong dữ liệu huấn luyện
AGE_MAX = 100  # Thay bằng giá trị thực tế của max tuổi trong dữ liệu huấn luyện

st.title("Stroke Prediction App")
st.markdown("## Nhập tuổi để dự đoán nguy cơ đột quỵ.")

# Nhập thông tin từ người dùng
age = st.slider("Tuổi", AGE_MIN, AGE_MAX, 50)

# Chuẩn hóa min-max
normalized_age = (age - AGE_MIN) / (AGE_MAX - AGE_MIN)

# Tạo DataFrame từ input
input_data = pd.DataFrame({
    'age': [normalized_age]  # Chỉ có một cột 'age' đã chuẩn hóa
})

# Hiển thị dữ liệu đã chuẩn hóa
st.write("**Tuổi sau khi chuẩn hóa:**", input_data)

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    prediction = model.predict(input_data)
    st.success(f"Nguy cơ đột quỵ: {'Có' if prediction[0] == 1 else 'Không'}")
