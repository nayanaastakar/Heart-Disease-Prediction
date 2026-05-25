import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / 'heart.csv'
if not csv_path.exists():
    csv_path = BASE_DIR / 'heart1.csv'

if not csv_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Expected heart.csv or heart1.csv in {}.".format(BASE_DIR)
    )

print("Loading dataset from: {}".format(csv_path))

df = pd.read_csv(csv_path)

if csv_path.name == 'heart.csv':
    df = df[[
        'age', 'sex', 'trestbps', 'chol', 'fbs',
        'exang', 'oldpeak', 'ca', 'target'
    ]].copy()
elif csv_path.name == 'heart1.csv':
    df = df[[
        'Age', 'Sex', 'RestingBP', 'Cholesterol',
        'FastingBS', 'ExerciseAngina', 'Oldpeak',
        'HeartDisease'
    ]].copy()
    df.columns = [
        'age', 'sex', 'trestbps', 'chol',
        'fbs', 'exang', 'oldpeak', 'target'
    ]
    df['ca'] = 0
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})
    df['exang'] = df['exang'].map({'Y': 1, 'N': 0})
    df['target'] = df['target'].map({'Presence': 1, 'Absence': 0})

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

FEATURES = [
    'age', 'sex', 'trestbps', 'chol',
    'fbs', 'exang', 'oldpeak', 'ca'
]

df = df[FEATURES + ['target']].copy()

X = df[FEATURES]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print('Accuracy:', round(accuracy_score(y_test, y_pred), 3))
print('ROC-AUC:', round(roc_auc_score(y_test, y_prob), 3))

model_path = BASE_DIR / 'heart_rf_model.pkl'
joblib.dump(model, model_path)
print('Model saved as {}'.format(model_path.name))


def predict_heart_disease(age, sex, bp, chol, sugar, exang, oldpeak, ca=0):
    input_df = pd.DataFrame([
        {
            'age': age,
            'sex': sex,
            'trestbps': bp,
            'chol': chol,
            'fbs': sugar,
            'exang': exang,
            'oldpeak': oldpeak,
            'ca': ca,
        }
    ])
    prob = model.predict_proba(input_df)[0][1]
    return round(prob * 100, 2)


if __name__ == '__main__':
    result = predict_heart_disease(
        age=55,
        sex=1,
        bp=140,
        chol=230,
        sugar=1,
        exang=1,
        oldpeak=1.0,
        ca=0
    )
    print('Heart Disease Probability: {}%'.format(result))
