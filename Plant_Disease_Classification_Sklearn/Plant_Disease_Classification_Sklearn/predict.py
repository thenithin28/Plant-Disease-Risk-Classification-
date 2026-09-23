import joblib
import pandas as pd

model = joblib.load("plant_disease_classifier.pkl")

sample_leaf = pd.DataFrame([{
    "Leaf_Color_Score": 5.2,
    "Leaf_Spot_Count": 20,
    "Leaf_Moisture": 61,
    "Leaf_Texture_Score": 5.3,
    "Plant_Age_Days": 70,
    "Soil_Moisture": 58,
    "Temperature": 28
}])

prediction = model.predict(sample_leaf)[0]
probabilities = model.predict_proba(sample_leaf)[0]
confidence = probabilities.max()

print("Plant Disease Classification")
print("----------------------------")
print("Predicted Class:", prediction)
print(f"Confidence: {confidence:.2%}")
