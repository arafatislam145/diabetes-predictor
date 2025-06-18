# 🩺 Diabetes Predictor using Random Forest

এই প্রজেক্টে Python দিয়ে তৈরি করা হয়েছে একটি Diabetes Prediction Tool। এটি Random Forest ব্যবহার করে বলে দিতে পারে একজন মানুষের ডায়াবেটিস হওয়ার সম্ভাবনা আছে কিনা।

## 🛠 ব্যবহার করা টুলস

- Python 3
- pandas
- scikit-learn
- joblib

## 🧠 কিভাবে কাজ করে

১. `diabetes.csv` ডেটাসেট থেকে ট্রেনিং নেয়  
২. `RandomForestClassifier` দিয়ে মডেল তৈরি করে  
৩. মডেল `.pkl` ফাইলে সেভ করে  
৪. `predict.py` দিয়ে যেকোনো ইনপুট দিলে প্রেডিকশন করে

## 🖥️ রান করার নিয়ম

```bash
# ভার্চুয়াল এনভায়রনমেন্ট তৈরি করো
python3 -m venv venv
source venv/bin/activate

# ডিপেনডেন্সি ইনস্টল করো
pip install -r requirements.txt

# মডেল ট্রেন করো
python train_model.py

# প্রেডিকশন চালাও
python predict.py
