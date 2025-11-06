#!/usr/bin/env python
"""
scenario_decision.py

Module for loading, expanding, and training decision trees from scenario tables.

This module provides comprehensive utilities for machine learning pipeline operations
on scenario-based data, including data preprocessing, "don't care" value expansion,
decision tree training, and model evaluation. It is specifically designed to work
with tabular scenario data where the last column represents decision labels and
feature columns may contain NaN values representing "don't care" conditions.

Key Features:
    - Load and clean scenario data from CSV/Excel files
    - Expand "don't care" (NaN) values into all possible combinations
    - Train scikit-learn DecisionTreeClassifier models
    - Evaluate trained models with flexible input formats
    - Extract model metadata (feature names, class names)
    - Print human-readable decision tree representations

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley
"""

from itertools import product
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


def load_scenarios(file_path, sheet_name=0):
    """
    Loads and preprocesses scenario data from a CSV or Excel file.
    
    Performs comprehensive data cleaning including removing comment columns,
    dropping empty rows/columns, converting non-numeric feature values to NaN,
    and removing constant feature columns. Assumes the last column contains
    decision labels and all preceding columns are features.
    
    Data Processing Steps:
        1. Load data from CSV or Excel file
        2. Remove columns named "Comments" or "Comment" (case-insensitive)
        3. Drop completely empty rows
        4. Convert non-numeric values in feature columns to NaN
        5. Remove columns that are entirely missing
        6. Remove feature columns with constant values (no variability)

    Args:
        file_path (str): Path to the CSV or Excel file containing training data.
            Supported formats: .csv, .xls, .xlsx
        sheet_name (str or int, optional): Sheet name or index to use if loading 
            from Excel. Defaults to 0 (first sheet).

    Returns:
        pd.DataFrame: Cleaned scenario data with feature columns (all but last)
            containing only numeric values or NaN, and label column (last) preserved.
            
    Raises:
        ValueError: If file format is not supported (not .csv, .xls, or .xlsx).
    """

    # Load data from file
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    # Drop any column named "Comments" or "Comment" (case-insensitive)
    cols_to_drop = [col for col in data.columns if col.lower()
                    in ("comments", "comment")]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    # Drop any rows that are all blank
    data = data.dropna(how='all')

    # Convert non-numeric values to missing in all columns except the label (final) column
    feature_cols = data.columns[:-1]
    data[feature_cols] = data[feature_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce')
    )

    # Drop all columns where all values are missing
    data = data.dropna(axis=1, how='all')

    # Drop feature columns with the same value for all rows
    feature_cols = data.columns[:-1]
    constant_cols = [
        col for col in feature_cols if data[col].nunique(dropna=False) == 1]
    data = data.drop(columns=constant_cols)

    return data


def expand_dont_care_rows(scenarios):
    """
    Expands rows containing "don't care" (NaN) values in feature columns.
    
    For each row with NaN values in feature columns, generates all possible
    combinations by replacing each NaN with every unique value found for that
    feature in the dataset. This allows scenarios with partial specifications
    to represent multiple concrete cases, enabling more comprehensive training
    data generation.
    
    Process:
        1. Identify all unique values for each feature column (excluding NaN)
        2. For each row with NaN values, create cartesian product of:
           - Existing values for non-NaN columns
           - All possible values for NaN columns
        3. Generate separate rows for each combination
        4. Preserve label column (last column) unchanged
    
    Args:
        scenarios (pd.DataFrame): Input dataframe where feature columns may 
            contain NaN values representing "don't care" conditions. 
            Last column is assumed to be the decision label.

    Returns:
        pd.DataFrame: Expanded dataframe with no NaN values in feature columns.
            May have significantly more rows than input due to expansion.
    """
    feature_cols = scenarios.columns[:-1]
    label_col = scenarios.columns[-1]

    # Identify possible values for each feature from the dataframe (excluding NaN)
    feature_possible_values = {
        col: sorted(scenarios[col].dropna().unique()) for col in feature_cols
    }

    expanded_rows = []

    for _, row in scenarios.iterrows():
        nan_cols = [col for col in feature_cols if pd.isna(row[col])]
        if not nan_cols:
            expanded_rows.append(row)
            continue

        # For each nan_col, get possible values, else use the existing value
        value_lists = []
        for col in feature_cols:
            if col in nan_cols:
                value_lists.append(feature_possible_values[col])
            else:
                value_lists.append([row[col]])

        # Generate all combinations for the NaN columns
        for values in product(*value_lists):
            new_row = list(values) + [row[label_col]]
            expanded_rows.append(pd.Series(new_row, index=scenarios.columns))

    return pd.DataFrame(expanded_rows, columns=scenarios.columns)


def train_decision_tree(data):
    """
    Trains a scikit-learn DecisionTreeClassifier on the provided scenario data.
    
    Creates and fits a decision tree model using default parameters, automatically
    splitting the input dataframe into feature matrix (all columns except last)
    and target vector (last column). The trained model retains feature names
    for later introspection and evaluation.
    
    Model Configuration:
        - Uses DecisionTreeClassifier with default parameters
        - Automatically infers feature names from dataframe columns
        - Supports both numeric and categorical target labels
        - No parameter tuning or cross-validation performed

    Args:
        data (pd.DataFrame): Preprocessed scenario data where all columns except 
            the last are features, and the last column contains decision labels.
            Feature columns should contain only numeric values (no NaN).

    Returns:
        DecisionTreeClassifier: Trained scikit-learn decision tree model with
            fitted parameters and accessible feature_names_in_ and classes_ attributes.
    """
    # Split data into features and labels
    feature_table = data.iloc[:, :-1]  # All columns except the last
    label_vector = data.iloc[:, -1]   # The last column

    # Train the decision tree model
    model = DecisionTreeClassifier()
    model.fit(feature_table, label_vector)
    return model


def print_decision_tree(model):
    """
    Prints a human-readable textual representation of the decision tree structure.
    
    Uses scikit-learn's export_text function to generate an indented tree
    representation showing decision rules, feature thresholds, and leaf nodes.
    Includes feature names for better readability. Useful for model debugging,
    understanding decision logic, and manual validation.
    
    Args:
        model (DecisionTreeClassifier or None): The trained decision tree model.
            If None, prints an error message instead of tree structure.
            
    Returns:
        None: Prints directly to console, no return value.
    """
    if model is None:
        print("Decision tree model is not trained. Call train_decision_tree() first.")
    else:
        tree_representation = export_text(
            model, feature_names=model.feature_names_in_)
        print(tree_representation)


def get_feature_names(model):
    """
    Extracts the feature names used during model training.
    
    Returns the column names from the original training dataframe that were
    used as features in the decision tree model. These names are automatically
    stored by scikit-learn during the fit() process and are useful for model
    introspection, feature mapping, and constructing evaluation inputs.

    Args:
        model (DecisionTreeClassifier or None): The trained decision tree model.
            Must have been fitted with a pandas DataFrame to retain feature names.
            
    Returns:
        list: Feature names as strings in the order used by the model, or 
            empty list if model is None. Order matches the expected input
            order for model evaluation.
    """
    if model is None:
        return []
    return list(model.feature_names_in_)


def get_class_names(model):
    """
    Extracts the unique class labels learned during model training.
    
    Returns the distinct decision labels (target values) that the model
    was trained to predict. These are automatically identified by scikit-learn
    from the training data's label column and stored in sorted order.
    Useful for understanding the model's output space and decision options.

    Args:
        model (DecisionTreeClassifier or None): The trained decision tree model.
            Must have been fitted to determine available classes.
            
    Returns:
        list: Unique class labels in sorted order, or empty list if model is None.
            These are the possible outputs from model.predict().
    """
    if model is None:
        return []
    return list(model.classes_)


def load_expand_train_scenarios(file_path, sheet_name=0, convert_to_int=True):
    """
    Complete pipeline for loading, preprocessing, expanding, and training decision trees.
    
    Combines all major processing steps into a single convenient function:
    data loading with cleaning, "don't care" expansion, optional type conversion,
    and model training. This is the primary entry point for most use cases.
    
    Pipeline Steps:
        1. Load and clean scenario data using load_scenarios()
        2. Expand "don't care" (NaN) values using expand_dont_care_rows()
        3. Optionally convert feature columns to int16 for memory efficiency
        4. Train decision tree model using train_decision_tree()
    
    Args:
        file_path (str): Path to the CSV or Excel file containing training scenarios.
            Supported formats: .csv, .xls, .xlsx
        sheet_name (str or int, optional): Sheet name or index for Excel files.
            Defaults to 0 (first sheet).
        convert_to_int (bool, optional): Whether to convert feature columns to int16
            data type for memory efficiency. Defaults to True. Set to False if
            features contain non-integer values.

    Returns:
        tuple: Three-element tuple containing:
            - scenarios (pd.DataFrame): Original cleaned data with possible NaN values
            - scenarios_expanded (pd.DataFrame): Expanded data with no NaN in features
            - model (DecisionTreeClassifier): Trained decision tree model
    """
    scenarios = load_scenarios(file_path, sheet_name=sheet_name)
    scenarios_expanded = expand_dont_care_rows(scenarios)

    if convert_to_int:
        # Convert all feature columns to int16
        feature_cols = scenarios_expanded.columns[:-1]
        scenarios_expanded[feature_cols] = scenarios_expanded[feature_cols].astype('Int16')

    model = train_decision_tree(scenarios_expanded)
    return scenarios, scenarios_expanded, model


def evaluate_model(model, features):
    """
    Evaluates the trained decision tree model on a single feature set.
    
    Accepts feature values in flexible formats (dictionary or list) and returns
    the model's predicted class label. Handles feature name mapping and order
    matching automatically. For dictionary input, missing features default to 0.
    
    Input Processing:
        - Dictionary: Maps feature names to values, handles missing features
        - List/tuple/array: Assumes values are in model.feature_names_in_ order
        - Automatic DataFrame construction for sklearn compatibility
        - Single prediction extraction from sklearn's array output
    
    Args:
        model (DecisionTreeClassifier): The trained decision tree model.
            Must not be None and must have feature_names_in_ attribute.
        features (dict, list, tuple, or np.ndarray): Feature values for prediction.
            - dict: {feature_name: value} mapping (missing features default to 0)
            - list/tuple/array: Values in order matching model.feature_names_in_

    Returns:
        The predicted class label (same type as training labels). Single value
        extracted from sklearn's prediction array.
        
    Raises:
        ValueError: If model is None, lacks feature_names_in_, or features
            format is not supported.
    """
    if model is None:
        raise ValueError("Model is None.")
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        raise ValueError("Model does not have feature_names_in_.")

    if isinstance(features, dict):
        # Ensure order matches model's feature_names_in_
        feature_table = pd.DataFrame([[features.get(name, 0)
                                       for name in feature_names]], columns=feature_names)
    elif isinstance(features, (list, tuple, np.ndarray)):
        feature_table = pd.DataFrame([features], columns=feature_names)
    else:
        raise ValueError("Features must be a dict or list/tuple/array.")
    return model.predict(feature_table)[0]
