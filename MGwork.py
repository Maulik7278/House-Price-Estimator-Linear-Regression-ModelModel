"""
STEP 1 : Training a model to predict house prices and saving the model pipeline and RMSE in a .pkl file

This script has 3 sections :
    1. Dummy data generation steps (to be replaced with your own data pre-processing code)
    2. Model development code (to be replaced with your own model development code)
    3. Saving the Pipeline and RMSE in a .pkl file (to be replaced with your own saving code)

    
Command to run the complete script : python train_house_price_model.py    
    
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import joblib


#----------------------------------------------------------------------
#                     01 - Dummy data generation steps
#        (Change these code lines with your data pre-processing code)
#---------------------------------------------------------------------

# ---- Load data (works in Google Colab and locally) ----
df = pd.read_csv("housing_data.csv")

# ---- Cleaning (done once, on the full dataset, before splitting) ----
none_cols = [
    "Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu",
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Mas Vnr Type"
]
for col in none_cols:
    df[col] = df[col].fillna("None")

zero_cols = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath",
    "Garage Cars", "Garage Area", "Garage Yr Blt"
]
for col in zero_cols:
    df[col] = df[col].fillna(0)

df["Lot Frontage"] = (
    df.groupby("Neighborhood")["Lot Frontage"]
    .transform(lambda x: x.fillna(x.median()))
)
df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

# MS SubClass: numeric code -> string (so it's treated as categorical)
df["MS SubClass"] = df["MS SubClass"].astype(str)

print(df.shape)
print("Remaining missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())


#----------------------------------------------------------------------
#                     02 - Model development code
#        (Change these code lines with your model development code)
#---------------------------------------------------------------------

# ---- Sampling ----
X = df.drop(columns=['SalePrice', 'PID', 'Order'])
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_test = X_train.copy(), X_test.copy()
print(f"X_train --> {X_train.shape}   X_test --> {X_test.shape}")

# ---- Outlier treatment (fit bounds on train, apply to both) ----
cap_cols = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area", "Total Bsmt SF",
    "1st Flr SF", "Gr Liv Area", "Garage Area", "Wood Deck SF",
    "Open Porch SF", "Bsmt Unf SF", "BsmtFin SF 1", "2nd Flr SF"
]
lower = X_train[cap_cols].quantile(0.01)
upper = X_train[cap_cols].quantile(0.99)
X_train[cap_cols] = X_train[cap_cols].clip(lower=lower, upper=upper, axis=1)
X_test[cap_cols]  = X_test[cap_cols].clip(lower=lower, upper=upper, axis=1)

# ---- Column groups ----
# Ordinal columns: categories listed worst -> best (domain-knowledge order)
ordinal_maps = {
    "Exter Qual":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "Exter Cond":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Qual":      ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Cond":      ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Exposure":  ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFin Type 1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFin Type 2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Heating QC":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "Kitchen Qual":   ["Po", "Fa", "TA", "Gd", "Ex"],
    "Fireplace Qu":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Finish":  ["None", "Unf", "RFn", "Fin"],
    "Garage Qual":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Cond":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Pool QC":        ["None", "Fa", "TA", "Gd", "Ex"],
    "Paved Drive":    ["N", "P", "Y"],
    "Lot Shape":      ["IR3", "IR2", "IR1", "Reg"],
    "Land Slope":     ["Gtl", "Mod", "Sev"],
    "Functional":     ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Fence":          ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}
ordinal_features = list(ordinal_maps.keys())
ordinal_categories = list(ordinal_maps.values())

# Nominal columns: levels below 1% frequency (learned on train) are grouped, then one-hot
nominal_features = [
    "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour",
    "Lot Config", "Neighborhood", "Condition 1", "Condition 2",
    "Bldg Type", "House Style", "Roof Style", "Roof Matl",
    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Foundation",
    "Heating", "Central Air", "Electrical", "Garage Type",
    "Misc Feature", "Sale Type", "Sale Condition", "Utilities",
]

# Numeric columns to standardize
numeric_features = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area", "Total Bsmt SF",
    "1st Flr SF", "Gr Liv Area", "Garage Area", "Wood Deck SF",
    "Open Porch SF", "Bsmt Unf SF", "BsmtFin SF 1", "2nd Flr SF",
    "BsmtFin SF 2", "Low Qual Fin SF", "Enclosed Porch", "3Ssn Porch",
    "Screen Porch", "Pool Area", "Misc Val",
    "Overall Qual", "Overall Cond",
    "Garage Cars", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath",
    "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
    "Fireplaces", "Year Built", "Year Remod/Add", "Garage Yr Blt",
    "Yr Sold", "Mo Sold",
]

# Any column not listed in a group above is dropped by the ColumnTransformer
covered = set(ordinal_features + nominal_features + numeric_features)
print("Columns not used by the model:", sorted(set(X.columns) - covered))

# ---- Pipeline ----
preprocessor = ColumnTransformer([
    ("ord", OrdinalEncoder(categories=ordinal_categories,
                           handle_unknown="use_encoded_value", unknown_value=-1),
     ordinal_features),
    ("cat", OneHotEncoder(drop="first", min_frequency=0.01,
                          handle_unknown="infrequent_if_exist", sparse_output=False),
     nominal_features),
    ("num", StandardScaler(), numeric_features),
], remainder="drop")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)

# ---- Performance evaluation ----
y_pred_train = pipeline.predict(X_train)
y_pred_test  = pipeline.predict(X_test)

print("\n--- Training performance ---")
print(f"MSE       --> {mean_squared_error(y_train, y_pred_train):.0f}")
print(f"RMSE      --> {root_mean_squared_error(y_train, y_pred_train):.0f}")
print(f"R-squared --> {r2_score(y_train, y_pred_train):.4f}")

print("\n--- Test performance ---")
print(f"MSE       --> {mean_squared_error(y_test, y_pred_test):.0f}")
print(f"RMSE      --> {root_mean_squared_error(y_test, y_pred_test):.0f}")
print(f"R-squared --> {r2_score(y_test, y_pred_test):.4f}")

rmse = root_mean_squared_error(y_test, y_pred_test)


#----------------------------------------------------------------------
#               03 - Saving the Pipeline and RMSE in a .pkl file
#                       (Change these code lines as required)
#---------------------------------------------------------------------

joblib.dump({"pipeline": pipeline, "rmse": rmse}, "house_price_pipeline.pkl")
print("\nSaved -> house_price_pipeline.pkl (pipeline + rmse)")
