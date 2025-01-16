import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Tải mô hình, danh sách cột và scaler
model, train_columns, scaler = joblib.load('app_model.pkl')

# Giao diện ứng dụng
st.title("Stroke Prediction App")
st.markdown("## Nhập thông tin bệnh nhân để dự đoán nguy cơ đột quỵ.")

# Nhập thông tin từ người dùng
gender = st.selectbox("Giới tính", ["Male", "Female"])
age = st.slider("Tuổi", 0, 100, 50)
hypertension = st.selectbox("Tăng huyết áp", [0, 1])
heart_disease = st.selectbox("Bệnh tim", [0, 1])
ever_married = st.selectbox("Đã từng kết hôn", ["Yes", "No"])
work_type = st.selectbox("Loại công việc", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Loại nơi cư trú", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Mức đường huyết trung bình (mmol/L)", 0.0, 300.0, 100.0)
bmi = st.number_input("Chỉ số BMI", 0.0, 50.0, 25.0)
smoking_status = st.selectbox("Tình trạng hút thuốc", ["formerly smoked", "never smoked", "smokes"])

# Bước mã hóa dữ liệu đầu vào
# Label Encoding cho các cột phân loại
label_encoder = LabelEncoder()
gender_encoded = label_encoder.fit_transform(["Male", "Female"]).tolist().index(gender)
ever_married_encoded = label_encoder.fit_transform(["Yes", "No"]).tolist().index(ever_married)
residence_type_encoded = label_encoder.fit_transform(["Urban", "Rural"]).tolist().index(residence_type)

# One-hot Encoding cho các cột `work_type` và `smoking_status`
work_type_columns = ["Children", "Never_worked", "Private", "Self-employed"]
work_type_encoded = [1 if work_type == wt else 0 for wt in work_type_columns]

smoking_status_columns = ["never smoked", "smokes"]
smoking_status_encoded = [1 if smoking_status == ss else 0 for ss in smoking_status_columns]

# Tạo DataFrame đầu vào
input_data = pd.DataFrame({
    'gender': [gender_encoded],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married_encoded],
    'Residence_type': [residence_type_encoded],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'work_type_Children': [work_type_encoded[0]],
    'work_type_Never_worked': [work_type_encoded[1]],
    'work_type_Private': [work_type_encoded[2]],
    'work_type_Self-employed': [work_type_encoded[3]],
    'smoking_status_never smoked': [smoking_status_encoded[0]],
    'smoking_status_smokes': [smoking_status_encoded[1]]
})

# Chuẩn hóa dữ liệu số với scaler
scaled_data = scaler.transform(input_data[train_columns])

# Hiển thị dữ liệu đầu vào
st.write("**Dữ liệu đầu vào đã xử lý:**")
st.dataframe(input_data)

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    prediction = model.predict(scaled_data)
    st.success(f"Nguy cơ đột quỵ: {'Có' if prediction[0] == 1 else 'Không'}")

