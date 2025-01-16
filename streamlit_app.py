import joblib
import pandas as pd
import streamlit as st

# Tải mô hình
model, train_columns, scaler = joblib.load('app_model.pkl')

st.title("Stroke Prediction App")
st.markdown("## Nhập thông tin để dự đoán nguy cơ đột quỵ.")

# Thu thập dữ liệu từ người dùng
gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
age = st.slider("Tuổi", 0, 100, 50)
hypertension = st.selectbox("Cao huyết áp", ["Không", "Có"])
heart_disease = st.selectbox("Bệnh tim mạch", ["Không", "Có"])
ever_married = st.selectbox("Đã từng kết hôn", ["Không", "Có"])
work_type = st.selectbox("Loại công việc", ["Never_worked", "Private", "Self-employed"])
residence_type = st.selectbox("Loại nơi cư trú", ["Thành thị", "Nông thôn"])
avg_glucose_level = st.slider("Mức đường huyết trung bình", 50.0, 200.0, 100.0)
bmi = st.slider("Chỉ số BMI", 10.0, 50.0, 22.0)
smoking_status = st.selectbox("Tình trạng hút thuốc", ["never smoked", "smokes"])

# Mã hóa và chuẩn hóa dữ liệu
data = {
    "gender": 1 if gender == "Nữ" else 0,
    "age": age / 100,  # Chuẩn hóa tuổi
    "hypertension": 1 if hypertension == "Có" else 0,
    "heart_disease": 1 if heart_disease == "Có" else 0,
    "ever_married": 1 if ever_married == "Có" else 0,
    "Residence_type": 1 if residence_type == "Thành thị" else 0,
    "avg_glucose_level": avg_glucose_level / 200,  # Chuẩn hóa đường huyết
    "bmi": bmi / 50,  # Chuẩn hóa BMI
    "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
    "work_type_Private": 1 if work_type == "Private" else 0,
    "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
    "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
    "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
}

# Chuyển đổi thành DataFrame với đúng cột theo mô hình
input_df = pd.DataFrame([data])

# Đảm bảo rằng các cột trong input_df khớp với tên cột mô hình đã học
input_df = input_df[train_columns]  # Sắp xếp lại cột theo đúng thứ tự

# Hiển thị dữ liệu đầu vào
st.write("**Dữ liệu đầu vào:**")
st.write(input_df)

# Dự đoán
if st.button("Dự đoán"):
    prediction = model.predict(input_df)
    result = "Có" if prediction[0] == 1 else "Không"
    st.success(f"Nguy cơ đột quỵ: {result}")
