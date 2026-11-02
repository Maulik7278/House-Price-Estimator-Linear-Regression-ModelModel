# ============================================================
# 00 - Library Setup
# ============================================================
import pandas as pd, numpy as np
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/My Drive/ml/"

# ============================================================
# 1 - Data
# ============================================================
House = pd.read_csv(path + "housing_data.csv")

# ============================================================
# 2 - Cleaning (done once, on the full dataset, before splitting)
# ============================================================
none_cols = [
    "Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu",
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Mas Vnr Type"
]
for col in none_cols:
    House[col] = House[col].fillna("None")

zero_cols = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath",
    "Garage Cars", "Garage Area", "Garage Yr Blt"
]
for col in zero_cols:
    House[col] = House[col].fillna(0)

House["Lot Frontage"] = (
    House.groupby("Neighborhood")["Lot Frontage"]
    .transform(lambda x: x.fillna(x.median()))
)
House["Lot Frontage"] = House["Lot Frontage"].fillna(House["Lot Frontage"].median())

House["Electrical"] = House["Electrical"].fillna(House["Electrical"].mode()[0])

print("Remaining missing values:", House.isnull().sum().sum())
print("Duplicate rows:", House.duplicated().sum())

# ============================================================
# 3 - Sampling
# ============================================================
X = House.drop(['SalePrice', 'PID', 'Order'], axis=1)
y = House['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_test = X_train.copy(), X_test.copy()

print(f"X_train --> {X_train.shape}   X_test --> {X_test.shape}")

# ============================================================
# 4 - Outlier treatment (fit bounds on train, apply to both)
# ============================================================
cap_cols = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area", "Total Bsmt SF",
    "1st Flr SF", "Gr Liv Area", "Garage Area", "Wood Deck SF",
    "Open Porch SF", "Bsmt Unf SF", "BsmtFin SF 1", "2nd Flr SF"
]

lower = X_train[cap_cols].quantile(0.01)
upper = X_train[cap_cols].quantile(0.99)

X_train[cap_cols] = X_train[cap_cols].clip(lower=lower, upper=upper, axis=1)
X_test[cap_cols]  = X_test[cap_cols].clip(lower=lower, upper=upper, axis=1)

# ============================================================
# 5 - MS SubClass: numeric code -> string (so it's treated as categorical)
# ============================================================
X_train["MS SubClass"] = X_train["MS SubClass"].astype(str)
X_test["MS SubClass"]  = X_test["MS SubClass"].astype(str)

# ============================================================
# 6 - Ordinal encoding (domain-knowledge maps, applied identically to both)
# ============================================================
exter_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_qual_map = {"NA": 0, "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_cond_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_exposure_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
bsmtfin_type_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
heating_qc_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
fireplace_qu_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
garage_finish_map = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
garage_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
pool_qc_map = {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
paved_drive_map = {"N": 1, "P": 2, "Y": 3}
lot_shape_map = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
land_slope_map = {"Gtl": 1, "Mod": 2, "Sev": 3}
functional_map = {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}
fence_map = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

ordinal_maps = {
    "Exter Qual": exter_map, "Exter Cond": exter_map,
    "Bsmt Qual": bsmt_qual_map, "Bsmt Cond": bsmt_cond_map,
    "Bsmt Exposure": bsmt_exposure_map,
    "BsmtFin Type 1": bsmtfin_type_map, "BsmtFin Type 2": bsmtfin_type_map,
    "Heating QC": heating_qc_map, "Kitchen Qual": heating_qc_map,
    "Fireplace Qu": fireplace_qu_map, "Garage Finish": garage_finish_map,
    "Garage Qual": garage_map, "Garage Cond": garage_map,
    "Pool QC": pool_qc_map, "Paved Drive": paved_drive_map,
    "Lot Shape": lot_shape_map, "Land Slope": land_slope_map,
    "Functional": functional_map, "Fence": fence_map,
}

for col, mapping in ordinal_maps.items():
    X_train[col] = X_train[col].map(mapping)
    X_test[col]  = X_test[col].map(mapping)

# ============================================================
# 7 - Nominal categoricals: collapse rare levels (fit on train), then one-hot
#     Learning the rare-category list from X_train and reusing it on X_test
#     removes the risk of the two encodings drifting apart.
# ============================================================
nominal_cols = [
    "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour",
    "Lot Config", "Neighborhood", "Condition 1", "Condition 2",
    "Bldg Type", "House Style", "Roof Style", "Roof Matl",
    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Foundation",
    "Heating", "Central Air", "Electrical", "Garage Type",
    "Misc Feature", "Sale Type", "Sale Condition", "Utilities",
]

RARE_THRESHOLD = 0.01   # categories below 1% frequency in X_train get collapsed
rare_categories = {}    # remembered so X_test uses the exact same collapse

for col in nominal_cols:
    freq = X_train[col].value_counts(normalize=True)
    rare = freq[freq < RARE_THRESHOLD].index.tolist()
    if rare:
        rare_categories[col] = rare
        X_train[col] = X_train[col].replace(rare, "Other")
        X_test[col]  = X_test[col].replace(rare, "Other")

X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True, dtype=int)
X_test  = pd.get_dummies(X_test,  columns=nominal_cols, drop_first=True, dtype=int)

# Align test columns to the training feature set (any dummy column that
# exists in one but not the other is handled here, once, instead of
# risking a manual mismatch column-by-column).
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ============================================================
# 8 - Standardization (fit scaler on train only)
# ============================================================
numeric_scale_cols = [
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

scaler = StandardScaler()
X_train[numeric_scale_cols] = scaler.fit_transform(X_train[numeric_scale_cols])
X_test[numeric_scale_cols]  = scaler.transform(X_test[numeric_scale_cols])

# ============================================================
# 9 - Model fitting
# ============================================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

coef_df = pd.DataFrame(
    lr_model.coef_, index=lr_model.feature_names_in_, columns=["Coefficients"]
).round(1)

# ============================================================
# 10 - Predictions
# ============================================================
y_pred_train = lr_model.predict(X_train)
y_pred_test  = lr_model.predict(X_test)

# ============================================================
# 11 - Performance evaluation
# ============================================================
print("\n--- Training performance ---")
print(f"MSE       --> {mean_squared_error(y_train, y_pred_train):.0f}")
print(f"RMSE      --> {root_mean_squared_error(y_train, y_pred_train):.0f}")
print(f"R-squared --> {r2_score(y_train, y_pred_train):.4f}")

print("\n--- Test performance ---")
print(f"MSE       --> {mean_squared_error(y_test, y_pred_test):.0f}")
print(f"RMSE      --> {root_mean_squared_error(y_test, y_pred_test):.0f}")
print(f"R-squared --> {r2_score(y_test, y_pred_test):.4f}")