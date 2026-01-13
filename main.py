import joblib
import pandas as pd
import tensorflow as tf


def main():
    model = tf.keras.models.load_model("./models/diabitics.keras")
    print("✅ Model loaded")

    scaler = joblib.load("./models/diabetes_scaler.joblib")
    print("✅ Scaler loaded")

    feature_order = joblib.load("./models/diabetes_features.joblib")
    print("✅ Feature order loaded")

    my_df = pd.DataFrame(
        [
            {
                "preg": 6.0,  # Number of times pregnant
                "plas": 148.0,  # Plasma glucose concentration (mg/dL) after 2-hour oral glucose tolerance test
                "pres": 72.0,  # Diastolic blood pressure (mm Hg)
                "skin": 35.0,  # Triceps skin fold thickness (mm)
                "insu": 0.0,  # 2-hour serum insulin (µU/mL) — 0 often means missing
                "mass": 33.6,  # Body Mass Index (BMI): weight (kg) / height² (m²)
                "pedi": 0.627,  # Diabetes pedigree function (genetic risk score)
                "age": 50.0,  # Age in years
            }
        ]
    )

    my_df = my_df[feature_order]

    print("\n📥 Input data:")
    print(my_df)

    my_scaled = scaler.transform(my_df)

    print("\n⚙️ Scaled input:")
    print(my_scaled[0])

    probability = model.predict(my_scaled, verbose=0)[0][0]
    prediction = "Diabetic" if probability >= 0.5 else "Non-Diabetic"

    print("\n🧠 Prediction Result")
    print(f"Probability (positive): {probability:.4f}")
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
