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
                print("👋 Exiting the program. Thank you!")
                sys.exit()
            return float(val)
        except ValueError:
            print("❌ Please enter a valid number.")

def get_binary_input(prompt):
    while True:
        val = input(prompt + " (Y/N): ").strip().lower()
        if val == 'exit':
            print("👋 Exiting the program.")
            sys.exit()
        if val in ['y', 'n']:
            return 1 if val == 'y' else 0
        print("❌ Please enter Y or N.")

def get_gender_input():
    while True:
        val = input("Sex (M/F): ").strip().lower()
        if val == 'exit':
            sys.exit()
        if val in ['m', 'f']:
            return 1 if val == 'm' else 0
        print("❌ Please enter M for Male or F for Female.")

def main():
    clear_screen()
    print("💡 Diabetes Prediction Tool (type 'exit' to quit)\n")

    try:
        model = joblib.load('model/model.pkl')
    except FileNotFoundError:
        print("❌ Model not found. Please train it first using 'train_model.py'.")
        return

    while True:
        print("🔢 Please enter the following health information:")

        pregnancies     = get_float_input("How many times pregnant: ")
        glucose         = get_float_input("Glucose level (e.g., 120): ")
        blood_pressure  = get_float_input("Blood pressure (e.g., 70): ")
        skin_thickness  = get_float_input("Skin thickness (e.g., 20): ")
        insulin         = get_float_input("Insulin level (e.g., 85): ")
        bmi             = get_float_input("BMI (e.g., 24.5): ")
        dpf             = get_float_input("Diabetes Pedigree Function (e.g., 0.4): ")
        age             = get_float_input("Age in years: ")

        sex             = get_gender_input()
        smoker          = get_binary_input("Does the person smoke?")
        alcohol         = get_binary_input("Does the person drink alcohol?")
        family_history  = get_binary_input("Any family history of diabetes?")

        # Final feature list (must match the training dataset structure)
        features = [[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, sex, smoker, alcohol, family_history
        ]]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        print("\n🧾 Result:")
        if prediction == 1:
            print(f"⚠️ This person is likely to have diabetes ({probability:.2f}% chance)\n")
        else:
            print(f"✅ This person is not likely to have diabetes ({probability:.2f}% chance)\n")

        again = input("Do you want to try again? (Y/n): ").strip().lower()
        if again == 'n':
            print("👋 Thank you for using the predictor!")
            break
        print("\n")

if __name__ == "__main__":
    main()
