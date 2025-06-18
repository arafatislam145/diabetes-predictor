import joblib

# মডেল লোড করো
model = joblib.load('model/model.pkl')

# ইউজার ইনপুট নিও (সবগুলো ইনপুট float টাইপে কনভার্ট করো)
print("🔢 Please enter the following values:")

pregnancies = float(input("Number of Pregnancies: "))
glucose = float(input("Glucose Level: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# ইনপুটকে লিস্টে কনভার্ট করো
features = [[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]]

# প্রেডিকশন
prediction = model.predict(features)

# রেজাল্ট দেখাও
if prediction[0] == 1:
    print("🔴 The person is likely to have diabetes.")
else:
    print("🟢 The person is not likely to have diabetes.")
