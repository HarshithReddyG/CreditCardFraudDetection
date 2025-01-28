import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import pickle

def validate_data(input_train_path, input_test_path, output_path):
    """
    Perform data validation checks on the input dataset and save the output for reuse.

    Args:
        input_train_path (str): Path to the training dataset.
        input_test_path (str): Path to the testing dataset.
        output_path (str): Path to store the output validation results and plots.

    Returns:
        str: Path to the validated dataset saved as a pickle file.
    """
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'dataValidation'), exist_ok=True)

    # Redirect output to a file
    output_file = os.path.join(output_path, 'dataValidation.txt')
    with open(output_file, 'w') as f:
        # Load datasets
        X_train = pd.read_csv(input_train_path)
        X_test = pd.read_csv(input_test_path)

        # Add source_file column to identify the source of the rows
        X_train['source_file'] = 'fraudTrain'
        X_test['source_file'] = 'fraudTest'

        # Combine datasets
        df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

        # Print dataset information
        f.write(f"Shape of Training Dataset: \n{X_train.shape}\n")
        f.write(f"Shape of Testing Dataset: \n{X_test.shape}\n")
        f.write("Sample of Training Dataset:\n")
        f.write(X_train.sample(2).to_string() + "\n")
        f.write("Dataset Information:\n")
        X_train.info(buf=f)
        f.write("\nDescription of Training Dataset:\n")
        f.write(X_train.describe().T.to_string() + "\n")

        # Missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        f.write("\nMissing Values Per Column:\n")
        f.write(missing_values.to_string() + "\n")
        f.write("\nPercentage of Missing Data Per Column:\n")
        f.write(missing_percentage.to_string() + "\n")

        # Duplicate rows
        duplicate_rows = df.duplicated().sum()
        f.write(f"\nNumber of duplicate rows: {duplicate_rows}\n")

        # Data types
        f.write("\nData Types:\n")
        f.write(df.dtypes.to_string() + "\n")

        # Unique values per column
        for col in df.columns:
            f.write(f"{col}: {df[col].nunique()} unique values\n")

        # Dataset statistics
        f.write("\nDataset Statistics:\n")
        f.write(df.describe().to_string() + "\n")

        # Correlation matrix
        df_numerical = df.select_dtypes(include=[np.number])
        correlation_output_path = os.path.join(output_path, 'dataValidation/correlation_matrix.png')
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation matrix for numerical columns")
        plt.savefig(correlation_output_path)
        plt.close()

        # Categorical column categories
        for col in df.select_dtypes(include='object'):
            f.write(f"Number of unique values in column '{col}': {df[col].nunique()}\n")

        # Class imbalance
        class_distribution = df['is_fraud'].value_counts(normalize=True)
        f.write("\nClass Distribution (%):\n")
        f.write((class_distribution * 100).to_string() + "\n")

        imbalance_output_path = os.path.join(output_path, 'dataValidation/class_distribution.png')
        class_distribution.plot(kind='bar', color=['skyblue', 'orange'])
        plt.title("Class Distribution of 'is_fraud'")
        plt.xlabel("Class")
        plt.ylabel("Percentage")
        plt.xticks(rotation=0)
        plt.savefig(imbalance_output_path)
        plt.close()

    # Save the validated dataset to a pickle file
    pickle_path = './../Data/validated_data.pkl'
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as pkl_file:
        pickle.dump(df, pkl_file)

    print("All Data Validation checks completed and results are stored. Pickled dataset saved.")
    return pickle_path

# Example usage
if __name__ == "__main__":
    validated_data = validate_data('./Data/fraudTrain.csv', './Data/fraudTest.csv', './../output')
    print(f"Validated dataset saved at: {validated_data}")
