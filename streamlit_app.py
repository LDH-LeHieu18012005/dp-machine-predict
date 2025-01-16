import streamlit as st
import pandas as pd
import joblib

# Tải mô hình và các thông tin đã lưu
model, train_columns, min_max_scaler = joblib.load('app_model.pkl')

# Giá trị Min-Max từ quá trình huấn luyện (dùng để chuẩn hóa)
AGE_MIN, AGE_MAX = 0, 100  # Thay bằng giá trị thực tế
GLUCOSE_MIN, GLUCOSE_MAX = 50, 300  # Thay bằng giá trị thực tế
BMI_MIN, BMI_MAX = 10, 50  # Thay bằng giá trị thực tế

st.title("Stroke Prediction App")
st.markdown("## Nhập thông tin để dự đoán nguy cơ đột quỵ.")

# Nhập thông tin từ người dùng
gender = st.selectbox("Giới tính", ["Male", "Female"])
age = st.slider("Tuổi", AGE_MIN, AGE_MAX, 50)
hypertension = st.selectbox("Tăng huyết áp", [0, 1])
heart_disease = st.selectbox("Bệnh tim", [0, 1])
ever_married = st.selectbox("Đã kết hôn", ["Yes", "No"])
work_type = st.selectbox("Loại công việc", ["Private", "Self-employed", "Govt_job", "Never_worked"])
residence_type = st.selectbox("Loại nơi ở", ["Urban", "Rural"])
avg_glucose_level = st.slider("Mức đường huyết trung bình", GLUCOSE_MIN, GLUCOSE_MAX, 100)
bmi = st.slider("Chỉ số BMI", BMI_MIN, BMI_MAX, 25)
smoking_status = st.selectbox("Tình trạng hút thuốc", ["never smoked", "formerly smoked", "smokes"])

# Chuẩn hóa và mã hóa dữ liệu đầu vào
input_data = pd.DataFrame({
    'gender': [1 if gender == "Male" else 0],
    'age': [(age - AGE_MIN) / (AGE_MAX - AGE_MIN)],  # Chuẩn hóa Min-Max
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [1 if ever_married == "Yes" else 0],
    'Residence_type': [1 if residence_type == "Urban" else 0],
    'avg_glucose_level': [(avg_glucose_level - GLUCOSE_MIN) / (GLUCOSE_MAX - GLUCOSE_MIN)],  # Chuẩn hóa
    'bmi': [(bmi - BMI_MIN) / (BMI_MAX - BMI_MIN)],  # Chuẩn hóa
    'work_type_Never_worked': [1 if work_type == "Never_worked" else 0],
    'work_type_Private': [1 if work_type == "Private" else 0],
    'work_type_Self-employed': [1 if work_type == "Self-employed" else 0],
    'smoking_status_never smoked': [1 if smoking_status == "never smoked" else 0],
    'smoking_status_smokes': [1 if smoking_status == "smokes" else 0],
})

# Đảm bảo đầu vào có đủ các cột giống dữ liệu huấn luyện
input_data = input_data.reindex(columns=train_columns, fill_value=0)

st.write("**Dữ liệu đầu vào đã mã hóa:**", input_data)

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Nguy cơ đột quỵ: {'Có' if prediction[0] == 1 else 'Không'}")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
