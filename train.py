import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ================= LOAD DATA =================
df = pd.read_csv("machine_maintenance.csv")

print("Shape:", df.shape)
print(df.head())

# ================= BASIC CLEANING =================
df = df.drop("machine_id", axis=1)

# categorical & numerical columns
cat = ['plant_location']
num = ['temperature','vibration','pressure','humidity',
       'runtime_hours','load_percentage','maintenance_history']

# label encode plant_location (optional since we also one-hot later,
# but keeping because you used it)
le = LabelEncoder()
df['plant_location'] = le.fit_transform(df['plant_location'])


# ================= SHOW BOXPLOT (BEFORE) =================
for name, col in df.items():
    plt.figure(figsize=(6, 8))
    plt.title(f"Before Outlier Removal - {name}")
    plt.boxplot(col)
    plt.show()


# ================= OUTLIER DETECTION =================
num_df = df.select_dtypes(include=['int64', 'float64'])

for col in num_df.columns:
    Q1 = num_df[col].quantile(0.25)
    Q3 = num_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = num_df[(num_df[col] < lower) | (num_df[col] > upper)]

    print(f"{col} -> {len(outliers)} outliers")


# ================= REMOVE OUTLIERS =================
cols_for_outliers = ['temperature', 'pressure']

for col in cols_for_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = df.shape[0]

    df = df[(df[col] >= lower) & (df[col] <= upper)]

    after = df.shape[0]

    print(f"{col} removed:", before - after)

print("New shape after removal:", df.shape)


# ================= SHOW BOXPLOT (AFTER) =================
for name, col in df.items():
    plt.figure(figsize=(6, 8))
    plt.title(f"After Outlier Removal - {name}")
    plt.boxplot(col.dropna())
    plt.show()


# ================= SPLIT =================
X = df.drop('failure_within_7days', axis=1)
y = df['failure_within_7days']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================= PIPELINES =================
num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num),
    ('cat', cat_pipe, cat)
])


# ================= FINAL MODEL PIPELINE =================
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])


# ================= TRAIN =================
pipeline.fit(X_train, y_train)


# ================= PREDICT =================
y_pred = pipeline.predict(X_test)

print("Predictions:", y_pred[:10])


# ================= ACCURACY =================
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))


# ================= SAVE MODEL =================
with open("machine_failure_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully ✅")
