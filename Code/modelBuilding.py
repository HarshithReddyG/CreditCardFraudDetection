import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import json
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    make_scorer
)

# Define models for classification
base_models_classification = {
    "logistic_regression": LogisticRegression(random_state=42),
    #"svm": SVC(random_state=42, probability=True),
    "decision_tree": DecisionTreeClassifier(random_state=42),
}

def load_dataset(file_path):
    """
    Load a dataset from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def cross_validate_models(train_data, target_column='is_fraud', base_models=None, metric="f1", cv=None):
    """
    Perform cross-validation for each model and return a DataFrame with the scores.
    """
    if base_models is None:
        base_models = base_models_classification

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    scoring_metrics = {
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score),
    }

    if metric not in scoring_metrics:
        raise ValueError(f"Invalid metric: {metric}. Available metrics: {', '.join(scoring_metrics.keys())}")

    scorer = scoring_metrics[metric]

    for model_name, model in base_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer)
        for score in scores:
            results.append({"Model": model_name, f"{metric.capitalize()} Score": score})

    return pd.DataFrame(results)

def plot_boxplot(results, metric, output_path):
    """
    Plot a box plot for the given metric and save it to the output directory.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y=f"{metric.capitalize()} Score", data=results)
    plt.title(f"{metric.capitalize()} Score Boxplot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.join(output_path, f"{metric}_score_boxplot.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Boxplot saved to {file_path}")

def evaluate_classification_models(X_train, y_train, X_test, y_test, base_models):
    """
    Train base classifiers (default settings) and evaluate performance.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        base_models: Dictionary of base classifiers with their model names and objects
    
    Returns:
        report: Dictionary containing accuracy, precision, recall, F1 score, 
                confusion matrix, and classification report for each model
    """
    # Initialize an empty dictionary to store results
    report = {}

    # Loop through each base model
    for model_name, model in base_models.items():
        
        # Train the model on training data
        model.fit(X_train, y_train)

        # Make predictions on training and testing data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_precision_pos_class = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        test_precision_pos_class = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)

        train_recall_pos_class = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        test_recall_pos_class = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)

        train_f1_pos_class = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        test_f1_pos_class = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

        # Generate classification reports and confusion matrices for test data
        test_classification_report = classification_report(y_test, y_test_pred, zero_division=0)
        test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

        # Store evaluation metrics
        report[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision_pos_class': train_precision_pos_class,
            'test_precision_pos_class': test_precision_pos_class,
            'train_recall_pos_class': train_recall_pos_class,
            'test_recall_pos_class': test_recall_pos_class,
            'train_f1_pos_class': train_f1_pos_class,
            'test_f1_pos_class': test_f1_pos_class,
            'test_classification_report': test_classification_report,
            'test_confusion_matrix': test_confusion_matrix.tolist()  # Convert to list for better readability
        }

    return report

def train_and_evaluate(data_paths, output_path, target_column="is_fraud"):
    """
    Train and evaluate models for each dataset.

    Args:
        data_paths (dict): Dictionary containing dataset names as keys and file paths as values.
        output_path (str): Directory to save model results and plots.
        target_column (str): Name of the target column.
    """
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'training_output.txt')
    with open(log_file, 'w') as log:
        def log_print(*args):
            print(*args)
            print(*args, file=log)

        all_results = {}

        for dataset_name, file_path in data_paths.items():
            dataset = load_dataset(file_path)
            log_print(f"Evaluating {dataset_name} dataset...")

            results = cross_validate_models(dataset, target_column, base_models_classification, metric="f1")
            all_results[dataset_name] = results

            # Save results as a CSV file
            results_file = os.path.join(output_path, f"{dataset_name}_cv_results.csv")
            results.to_csv(results_file, index=False)
            log_print(f"Results for {dataset_name} saved to {results_file}")

            # Generate and save boxplot
            plot_boxplot(results, metric="f1", output_path=output_path)

        log_print("Training and evaluation completed.")

    return all_results

def print_results_nicely(results_dict):
    """
    Print the results from all_base_model_results in a structured format.
    
    Args:
        results_dict: Dictionary with dataset names as keys and evaluation results as values.
    """
    for dataset_name, model_results in results_dict.items():
        print(f"\nResults for Dataset: {dataset_name}")
        print("-" * 50)

        # Ensure model_results is a dictionary
        if isinstance(model_results, dict):
            # Convert the dictionary to a DataFrame
            results_df = pd.DataFrame([model_results]).T.reset_index()
            results_df.rename(columns={'index': 'Metric', 0: 'Value'}, inplace=True)

            # Display the DataFrame
            print(results_df.to_string(index=False))  # Print without row indices
        else:
            print(f"Model results for {dataset_name} are not in the expected format.")
        print("-" * 50)

def generate_shap_plots(model, X_train, X_test, output_path):
    """
    Generate SHAP plots for the given model.
    
    Args:
        model: Trained model to explain.
        X_train: Training data (used to fit SHAP explainer).
        X_test: Test data to explain.
        output_path: Directory to save SHAP plots.
    """
    os.makedirs(output_path, exist_ok=True)

    # Create a SHAP explainer
    if hasattr(model, "predict_proba") and isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_train)
    elif hasattr(model, "tree_") or isinstance(model, DecisionTreeClassifier):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification: Use the SHAP values for the positive class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(output_path, "shap_summary_plot.png"))
    plt.close()

    # Dependence Plot for the top feature
    top_feature = X_test.columns[0]  # Example: Change to a specific feature or sort by importance
    plt.figure()
    try:
        shap.dependence_plot(top_feature, shap_values, X_test, show=False)
        plt.savefig(os.path.join(output_path, f"shap_dependence_plot_{top_feature}.png"))
    except Exception as e:
        print(f"Error generating dependence plot for {top_feature}: {e}")
    plt.close()

    print(f"SHAP plots saved in {output_path}")



def evaluate_on_test_with_shap(data_paths, output_path, target_column="is_fraud", base_models=None):
    """
    Evaluate models on the test dataset and generate SHAP plots.

    Args:
        data_paths (dict): Dictionary containing dataset names as keys and file paths as values.
        output_path (str): Directory to save model results and plots.
        target_column (str): Name of the target column.
        base_models (dict): Dictionary of model names and corresponding model objects.

    Returns:
        dict: A dictionary containing test evaluation results for each dataset.
    """
    if base_models is None:
        raise ValueError("Base models must be provided.")

    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'test_evaluation_output.txt')
    with open(log_file, 'w') as log:
        def log_print(*args):
            print(*args)
            print(*args, file=log)

        # Load the test preprocessed dataset
        test_preprocessed_path = "../Data/test_preprocessed.pkl"
        with open(test_preprocessed_path, 'rb') as test_file:
            test_preprocessed = pickle.load(test_file)

        X_test = test_preprocessed.drop(columns=[target_column], axis='columns')
        y_test = test_preprocessed[target_column]

        test_results = {}

        for dataset_name, file_path in data_paths.items():
            # Load the dataset from the pickle file
            dataset = load_dataset(file_path)
            log_print(f"Evaluating {dataset_name} dataset on test data...")

            # Get train data from the dataset
            X_train = dataset.drop(columns=[target_column], axis='columns')
            y_train = dataset[target_column]

            # Evaluate models and generate SHAP plots
            for model_name, model in base_models.items():
                log_print(f"Training and evaluating {model_name} on {dataset_name} dataset...")
                model.fit(X_train, y_train)

                # Predict and calculate evaluation metrics
                y_pred = model.predict(X_test)
                test_results[f"{dataset_name}_{model_name}"] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "F1 Score": f1_score(y_test, y_pred)
                }

                # Generate SHAP plots
                shap_output_path = os.path.join(output_path, dataset_name, model_name)
                generate_shap_plots(model, X_train, X_test, shap_output_path)

        # Log the results
        log_print("\nTest Evaluation Results:")
        print_results_nicely(test_results)

        log_print("Test evaluation with SHAP completed.")
        return test_results
