import pandas as pd
import os

def ingest_data(input_path='train.csv', output_path='data/processed/clean_titanic.csv'):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['Survived'])

    expected_columns = {'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                        'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'}
    if not expected_columns.issubset(set(df.columns)):
        raise ValueError("Dataset columns don't match expected schema.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Data ingested and saved to {output_path}")

if __name__ == "__main__":
    ingest_data()
