import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# CREATE DATASET
# =========================================

data = {
    "GrLivArea": [
        1500, 1800, 2400, 3000, 1700,
        2100, 1200, 2500, 1900, 2200,
        1600, 2800, 3200, 1400, 2600
    ],

    "BedroomAbvGr": [
        3, 4, 4, 5, 3,
        4, 2, 5, 3, 4,
        3, 5, 6, 2, 4
    ],

    "FullBath": [
        2, 2, 3, 4, 2,
        3, 1, 3, 2, 3,
        2, 4, 4, 1, 3
    ],

    "SalePrice": [
        200000, 250000, 320000, 450000, 230000,
        310000, 150000, 400000, 270000, 330000,
        220000, 420000, 500000, 170000, 390000
    ]
}

# =========================================
# CREATE DATAFRAME
# =========================================

df = pd.DataFrame(data)

print("DATASET PREVIEW\n")
print(df.head())

# =========================================
# SELECT FEATURES
# =========================================

X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]

y = df["SalePrice"]

# =========================================
# SPLIT DATASET
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# TRAIN MODEL
# =========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY!")

# =========================================
# MODEL EVALUATION
# =========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\n===== MODEL RESULTS =====")

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)

# =========================================
# USER INPUT
# =========================================

print("\n===== HOUSE PRICE PREDICTION =====")

square_feet = float(input("Enter Square Feet: "))

bedrooms = int(input("Enter Number of Bedrooms: "))

bathrooms = int(input("Enter Number of Bathrooms: "))

# =========================================
# PREDICT USER INPUT
# =========================================

user_data = [[square_feet, bedrooms, bathrooms]]

predicted_price = model.predict(user_data)

print("\n===== PREDICTION RESULT =====")

print("Square Feet:", square_feet)

print("Bedrooms:", bedrooms)

print("Bathrooms:", bathrooms)

print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")

# =========================================
# VISUALIZATION
# =========================================

plt.figure(figsize=(8,6))

# Dataset points
plt.scatter(
    df["GrLivArea"],
    df["SalePrice"],
    label="Dataset Houses"
)

# User prediction point
plt.scatter(
    square_feet,
    predicted_price[0],
    s=200,
    marker="X",
    label="Your Predicted House"
)

plt.xlabel("Square Feet")

plt.ylabel("House Price")

plt.title("House Price Prediction Visualization")

plt.legend()

plt.grid(True)

plt.show()