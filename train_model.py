"""Train a student placement classifier and save it for the Streamlit app."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "placement.csv"
MODEL_PATH = PROJECT_DIR / "model.pkl"
FEATURE_COLUMNS = ["cgpa", "iq"]
TARGET_COLUMN = "placed"


def main():
    # Read the example student records from the data folder.
    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains empty values in the required columns.")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("The 'placed' target must contain only 0 and 1.")

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]

    # Keep the same share of placed and not-placed students in both sets.
    stratify_target = target if target.value_counts().min() >= 2 else None
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    # A fixed random seed makes this example repeatable when it is run again.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_train, target_train)

    predictions = model.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(target_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()