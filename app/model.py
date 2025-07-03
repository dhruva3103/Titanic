import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scripts.registry import register_model

model_path = os.path.join(os.path.dirname(__file__), "..", "trained_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

def predict_passenger(data: dict) -> int:
    if model is None:
        raise RuntimeError("Model is not trained yet. Please train the model first.")
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return int(prediction[0])

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def build_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def build_training_pipeline(preprocessor):
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return clf

def train_and_evaluate(df, target_column, output_model_path='trained_model.pkl'):
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    model_pipeline = build_training_pipeline(preprocessor)

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

    joblib.dump(model_pipeline, output_model_path)
    print(f"\nModel saved to {output_model_path}")

    register_model(
        model_path=output_model_path,
        metrics={'accuracy': accuracy_score(y_test, y_pred)}
    )
