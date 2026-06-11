import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# ================= LOAD DATA =================

df = pd.read_csv(r"D:\Sumalatha\Kidney_data.csv")

# Drop id column
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# ================= CLEAN TARGET =================

df['classification'] = (
    df['classification']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['classification'] = df['classification'].replace({
    'ckd\t': 'ckd',
    'notckd': 'not ckd'
})

df['classification'] = df['classification'].map({
    'ckd': 0,
    'not ckd': 1
})

# ================= FIX DATA TYPES =================

for col in ['pcv', 'wc', 'rc']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean categorical values

if 'dm' in df.columns:
    df['dm'] = df['dm'].replace({
        '\tno': 'no',
        '\tyes': 'yes',
        ' yes': 'yes'
    })

if 'cad' in df.columns:
    df['cad'] = df['cad'].replace({
        '\tno': 'no'
    })

# ================= HANDLE MISSING VALUES =================

cat_cols = df.select_dtypes(include='object').columns

num_cols = [
    col for col in df.columns
    if col not in cat_cols and col != 'classification'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Remove rows with missing target

df.dropna(subset=['classification'], inplace=True)

# ================= FEATURES & TARGET =================

X = df.drop('classification', axis=1)
y = df['classification']

print("\nClass Distribution:")
print(y.value_counts())

# Imputation

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Scaling

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ================= MODELS =================

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "matrix": confusion_matrix(y_test, y_pred)
    }

# ================= PDF =================

pdf = PdfPages("results.pdf")

def save_graph():
    pdf.savefig(bbox_inches='tight')
    plt.show()
    plt.close()

# ================= HEATMAP =================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
save_graph()

# ================= KDE =================

plt.figure(figsize=(8,5))

sns.kdeplot(
    data=df,
    x='hemo',
    hue='classification',
    fill=True
)

plt.title("Hemoglobin Distribution")
plt.tight_layout()
save_graph()

# ================= CLASS DISTRIBUTION =================

plt.figure(figsize=(6,4))

sns.countplot(
    x='classification',
    data=df
)

plt.title("CKD Class Distribution")
plt.tight_layout()
save_graph()

# ================= HISTOGRAMS =================

for col in ['bgr', 'bu', 'sc']:

    plt.figure(figsize=(8,4))

    sns.histplot(
        data=df,
        x=col,
        hue='classification',
        kde=True,
        bins=30
    )

    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    save_graph()

# ================= ACCURACY COMPARISON =================

acc_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]['accuracy'] for m in results]
})

plt.figure(figsize=(7,5))

sns.barplot(
    data=acc_df,
    x='Model',
    y='Accuracy'
)

plt.title("Model Accuracy Comparison")
plt.tight_layout()
save_graph()

# ================= CONFUSION MATRICES =================

for model_name in results:

    plt.figure(figsize=(5,4))

    sns.heatmap(
        results[model_name]["matrix"],
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    save_graph()

# ================= RESULTS =================

for model_name in results:

    print("\n====================")
    print(model_name)
    print("====================")

    print(
        "Accuracy:",
        round(results[model_name]["accuracy"] * 100, 2),
        "%"
    )

    print(results[model_name]["report"])

pdf.close()

print("\nGraphs displayed successfully.")
print("Graphs saved in results.pdf")