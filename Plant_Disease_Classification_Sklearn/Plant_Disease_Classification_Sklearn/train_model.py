import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
os.makedirs("data", exist_ok=True)

n_per_class = 180
classes = ["Healthy", "Powdery Mildew", "Leaf Spot", "Rust"]
rows = []

def add_class(label, color, spots, moisture, texture, age, soil, temp):
    for _ in range(n_per_class):
        rows.append({
            "Leaf_Color_Score": np.clip(np.random.normal(color, 0.45), 1, 10),
            "Leaf_Spot_Count": max(0, int(np.random.normal(spots, 3))),
            "Leaf_Moisture": np.clip(np.random.normal(moisture, 8), 10, 100),
            "Leaf_Texture_Score": np.clip(np.random.normal(texture, 0.5), 1, 10),
            "Plant_Age_Days": max(5, int(np.random.normal(age, 20))),
            "Soil_Moisture": np.clip(np.random.normal(soil, 8), 10, 100),
            "Temperature": np.clip(np.random.normal(temp, 3), 10, 45),
            "Disease": label
        })

add_class("Healthy", 8.2, 2, 58, 8.2, 65, 55, 25)
add_class("Powdery Mildew", 5.8, 12, 72, 5.0, 70, 68, 23)
add_class("Leaf Spot", 4.8, 25, 48, 4.8, 75, 45, 27)
add_class("Rust", 5.4, 18, 62, 5.5, 80, 60, 29)

df = pd.DataFrame(rows)
df.to_csv("data/plant_disease_data.csv", index=False)

X = df.drop(columns=["Disease"])
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "plant_disease_classifier.pkl")

cm = confusion_matrix(y_test, predictions, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=25)
plt.title("Plant Disease Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("plant_disease_confusion_matrix.png")
plt.close()

print("\nModel saved as plant_disease_classifier.pkl")
print("Dataset saved as data/plant_disease_data.csv")
