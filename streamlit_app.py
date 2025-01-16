import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Tải mô hình, danh sách cột và scaler
model, train_columns, scaler = joblib.load('app_model.pkl')

# Cài đặt giao diện
st.title("Stroke Prediction App")
st.markdown("## Nhập thông tin để dự đoán nguy cơ đột quỵ.")

# Nhập thông tin từ người dùng
gender = st.selectbox("Giới tính", ["Male", "Female"])
age = st.slider("Tuổi", 0, 100, 50)
hypertension = st.selectbox("Tăng huyết áp", [0, 1])
heart_disease = st.selectbox("Bệnh tim", [0, 1])
ever_married = st.selectbox("Đã từng kết hôn", ["Yes", "No"])
work_type = st.selectbox("Loại công việc", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
Residence_type = st.selectbox("Loại nơi ở", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Mức glucose trung bình (mmol/L)", 0.0, 300.0, 100.0)
bmi = st.number_input("Chỉ số BMI", 0.0, 50.0, 25.0)
smoking_status = st.selectbox("Tình trạng hút thuốc", ["formerly smoked", "never smoked", "smokes"])

# Mã hóa các giá trị đầu vào
label_encoder = LabelEncoder()
gender_encoded = label_encoder.fit_transform(["Male", "Female"]).tolist().index(gender)
ever_married_encoded = label_encoder.fit_transform(["Yes", "No"]).tolist().index(ever_married)
Residence_type_encoded = label_encoder.fit_transform(["Urban", "Rural"]).tolist().index(Residence_type)

# Chuẩn hóa cột số
input_data = pd.DataFrame({
    'gender': [gender_encoded],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married_encoded],
    'Residence_type': [Residence_type_encoded],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'work_type_Children': [1 if work_type == "Children" else 0],
    'work_type_Never_worked': [1 if work_type == "Never_worked" else 0],
    'work_type_Private': [1 if work_type == "Private" else 0],
    'work_type_Self-employed': [1 if work_type == "Self-employed" else 0],
    'smoking_status_never smoked': [1 if smoking_status == "never smoked" else 0],
    'smoking_status_smokes': [1 if smoking_status == "smokes" else 0]
})

# Áp dụng scaler
scaled_data = scaler.transform(input_data[train_columns])

# Hiển thị dữ liệu đã chuẩn bị
st.write("**Dữ liệu đầu vào:**", input_data)

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    prediction = model.predict(scaled_data)
    st.success(f"Nguy cơ đột quỵ: {'Có' nếu prediction[0] == 1 khác 'Không'}")
