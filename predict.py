import joblib
import os
import sys

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_float_input(prompt):
    while True:
        try:
            val = input(prompt)
            if val.lower() == 'exit':
                print("👋 Exiting...")
                sys.exit()
            return float(val)
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def main():
    clear_screen()
    print("🧪 Diabetes Prediction Tool (type 'exit' anytime to quit)\n")

    try:
        model = joblib.load('model/model.pkl')
    except FileNotFoundError:
        print("❌ Model file not found! Please train the model first using `train_model.py`.")
        return

    while True:
        print("🔢 Enter patient data:")

        pregnancies     = get_float_input("Number of Pregnancies: ")
        glucose         = get_float_input("Glucose Level: ")
        blood_pressure  = get_float_input("Blood Pressure: ")
        skin_thickness  = get_float_input("Skin Thickness: ")
        insulin         = get_float_input("Insulin Level: ")
        bmi             = get_float_input("BMI: ")
        dpf             = get_float_input("Diabetes Pedigree Function: ")
        age             = get_float_input("Age: ")

        features = [[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        print("\n🧾 Result:")
        if prediction == 1:
            print(f"🔴 Likely to have diabetes ({probability:.2f}% probability)\n")
        else:
            print(f"🟢 Not likely to have diabetes ({probability:.2f}% probability)\n")

        again = input("🔁 Predict again? (Y/n): ").strip().lower()
        if again == 'n':
            print("👋 Goodbye!")
            break
        print("\n")

if __name__ == "__main__":
    main()
