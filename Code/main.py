import os
import pickle
from dataValidation import validate_data
from exploratoryDataAnalysis import perform_eda
from dataPreprocessing import preprocess_data, apply_resampling
from modelBuilding import train_and_evaluate, evaluate_on_test_with_shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def main():
    """
    Main script to load validated data, perform EDA, and proceed with further steps.
    """
    # Paths to input and output files
    input_train_path = './Data/fraudTrain.csv'
    input_test_path = './Data/fraudTest.csv'
    pickle_path = './../Data/validated_data.pkl'
    output_path = './../output'
    eda_marker_file = os.path.join(output_path, 'EDA', 'eda_output.txt')

    # Check if the pickle file exists
    if os.path.exists(pickle_path):
        print("Pickle file found! Loading validated dataset...")
        # Load validated data directly
        with open(pickle_path, 'rb') as pkl_file:
            df = pickle.load(pkl_file)
        print("Validated dataset loaded successfully!")
    else:
        print("Pickle file not found. Running data validation step...")
        # Run data validation and save the validated data as a pickle
        validated_data_path = validate_data(input_train_path, input_test_path, output_path)
        with open(validated_data_path, 'rb') as pkl_file:
            df = pickle.load(pkl_file)
        print("Data validation completed and dataset loaded!")

    # Check if EDA has already been performed
    if os.path.exists(eda_marker_file):
        print("EDA already performed. Skipping EDA step.")
    else:
        print("Performing Exploratory Data Analysis (EDA)...")
        perform_eda(df, os.path.join(output_path, 'EDA'))
        # Create a marker file to indicate EDA completion
        with open(eda_marker_file, 'w') as marker:
            marker.write("EDA completed successfully.")
        print("EDA completed. Outputs are stored in the specified output folder.")

    # Paths to data files
    data_path = '../Data'
    train_preprocessed_path = os.path.join(data_path, 'train_preprocessed.pkl')
    test_preprocessed_path = os.path.join(data_path, 'test_preprocessed.pkl')
    smote_path = os.path.join(data_path, 'SMOTE_resampled.pkl')
    adasyn_path = os.path.join(data_path, 'ADASYN_resampled.pkl')

    # Check if preprocessed data exists
    if os.path.exists(train_preprocessed_path) and os.path.exists(test_preprocessed_path):
        print("Preprocessed train and test data found! Loading...")
        with open(train_preprocessed_path, 'rb') as train_file:
            train_preprocessed = pickle.load(train_file)
        with open(test_preprocessed_path, 'rb') as test_file:
            test_preprocessed = pickle.load(test_file)
        print("Preprocessed train and test datasets loaded successfully!")
    else:
        print("Preprocessed data not found. Running data preprocessing...")
        train_preprocessed, test_preprocessed = preprocess_data(data_path)
        print("Data preprocessing completed and saved!")

    # Check if SMOTE and ADASYN datasets exist
    if os.path.exists(smote_path) and os.path.exists(adasyn_path):
        print("SMOTE and ADASYN resampled datasets found! Loading...")
        with open(smote_path, 'rb') as smote_file:
            smote_resampled_df = pickle.load(smote_file)
        with open(adasyn_path, 'rb') as adasyn_file:
            adasyn_resampled_df = pickle.load(adasyn_file)
        print("SMOTE and ADASYN datasets loaded successfully!")
    else:
        print("Resampled datasets not found. Applying SMOTE and ADASYN...")
        smote_resampled_df, adasyn_resampled_df = apply_resampling(data_path, train_preprocessed)
        print("Resampled datasets completed and saved!")

    # Proceed with further steps
    print("All preprocessing steps completed. Ready for modeling!")

    smote_path = os.path.join(data_path, 'SMOTE_resampled.pkl')
    adasyn_path = os.path.join(data_path, 'ADASYN_resampled.pkl')
    model_output_path = '../output/model_building'
    test_output_path = "../output/test_evaluation"

    # Check if resampled datasets exist
    if not os.path.exists(smote_path):
        raise FileNotFoundError(f"SMOTE file not found at {smote_path}")
    if not os.path.exists(adasyn_path):
        raise FileNotFoundError(f"ADASYN file not found at {adasyn_path}")

    # Create dictionary for datasets
    data_paths = {
        "SMOTE": smote_path,
        "ADASYN": adasyn_path,
    }
    # Define base models for classification
    base_models_classification = {
        "logistic_regression": LogisticRegression(random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }
    # Train and evaluate models for both datasets
    print("Starting training and cross-validation...")
    train_results = train_and_evaluate(data_paths, model_output_path, target_column="is_fraud")
    print("Training and cross-validation completed.")

    # Evaluate models on the test dataset
    print("Starting test evaluation...")
    test_results = evaluate_on_test_with_shap(
        data_paths,
        test_output_path,
        target_column="is_fraud",
        base_models=base_models_classification
    )
    print("Test evaluation completed.")

    # Handle results if needed
    print("Training Results:")
    for dataset, result in train_results.items():
        print(f"{dataset}: {result.head()}\n")

    print("Test Results:")
    for dataset, result in test_results.items():
        print(f"{dataset}: {result.head()}\n")

if __name__ == "__main__":
    main()
