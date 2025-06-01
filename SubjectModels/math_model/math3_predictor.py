import pandas as pd
import joblib

# Load the model
model = joblib.load("math3_model.joblib")

def preprocess_input(input_df):
    """
    Preprocess the input DataFrame to match the training data format.
    This includes feature engineering and renaming columns.
    """
    # Feature Engineering: Semester Percentages
    sem1_cols = ['Math-1 Theory', 'Physics Theory', 'Java-1 Theory', 'Software Engineering Theory']
    input_df['Semester1_Percentage'] = input_df[sem1_cols].mean(axis=1)

    sem2_cols = ['Math-2 Theory', 'Data Structures using Java Theory', 'DBMS Theory',
                 'Fundamental of Electronics and Electrical Theory', 'Java-2 Theory']
    input_df['Semester2_Percentage'] = input_df[sem2_cols].mean(axis=1)

    # Round percentages
    input_df['Semester1_Percentage'] = input_df['Semester1_Percentage'].round(2)
    input_df['Semester2_Percentage'] = input_df['Semester2_Percentage'].round(2)

    # Rename divisions to sections
    input_df = input_df.rename(columns={'Div-1': 'Section_1', 'Div-2': 'Section_2', 'Div-3': 'Section_3'})

    # Extract department letters from section fields
    for section in ['Section_1', 'Section_2', 'Section_3']:
        input_df[section] = input_df[section].str[0]

    return input_df

def predict_from_dataframe(input_df):
    """
    Preprocess the input DataFrame and predict the target value.
    """
    # Preprocess the input data
    processed_df = preprocess_input(input_df)

    # Predict using the pre-trained model
    predictions = model.predict(processed_df)
    return pd.Series(predictions, name="Predicted Values")

if __name__ == "__main__":
    # Load the test data
    new_data = pd.read_csv("../test_dataset.csv")

    # Predict
    predictions = predict_from_dataframe(new_data)

    # Output predictions
    print(predictions.head())
