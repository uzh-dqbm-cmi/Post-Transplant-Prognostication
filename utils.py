import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import re
import warnings

warnings.filterwarnings("ignore")
from collections import defaultdict
from typing import List, Tuple


primary_outcomes = [
    "death_y1",
    "death_y2",
    "death_y3",
    "death_y4",
    "death_y5",
    "death_y6",
    "death_y7",
    "death_y8",
    "death_y9",
    "death_y10",
    "death_y11",
    "death_y12",
    "death_y13",
    "graft_loss_y1",
    "graft_loss_y2",
    "graft_loss_y3",
    "graft_loss_y4",
    "graft_loss_y5",
    "graft_loss_y6",
    "graft_loss_y7",
    "graft_loss_y8",
    "graft_loss_y9",
    "graft_loss_y10",
    "graft_loss_y11",
    "graft_loss_y12",
    "graft_loss_y13",
]


def data_extraction():
    meta = pd.read_csv(f"data/variables.csv", index_col=0)
    data = pd.read_csv(f"data/data.csv", index_col=0)

    baseline_vars = meta[meta["io"] == "in"].index
    baseline_vars_cate = meta[(meta["io"] == "in") & (meta["type"] == "cat")].index
    baseline_vars_cont = meta[(meta["io"] == "in") & (meta["type"] == "cont")].index
    baseline_data_cont = data.loc[:, baseline_vars_cont]
    baseline_data_cate = data.loc[:, baseline_vars_cate]
    binary_vars = [
        col
        for col in baseline_vars_cate
        if data[col].nunique() == 2 or set(data[col].dropna().unique()).issubset({0, 1})
    ]

    baseline_data_cont = baseline_data_cont.dropna(thresh=0.9, axis=1)
    baseline_data_cate = baseline_data_cate.dropna(thresh=0.9, axis=1)
    # Update baseline_vars_cate to exclude binary variables
    baseline_vars_cate = baseline_vars_cate.difference(binary_vars)
    # Process categorical variables (excluding binary ones) with the existing pipeline
    baseline_data_cate = data.loc[:, baseline_vars_cate]
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False)),
        ]
    )
    baseline_data_cate = pd.DataFrame(
        categorical_pipeline.fit_transform(baseline_data_cate),
        index=baseline_data_cate.index,
        columns=categorical_pipeline.named_steps["onehot"].get_feature_names_out(
            baseline_data_cate.columns
        ),
    )
    # Process binary variables separately, including imputation of missing values
    baseline_data_binary = data.loc[:, binary_vars]
    # Use SimpleImputer to fill missing binary values with the most frequent value (0 or 1)
    binary_imputer = SimpleImputer(strategy="most_frequent")
    baseline_data_binary = pd.DataFrame(
        binary_imputer.fit_transform(baseline_data_binary),
        index=baseline_data_binary.index,
        columns=baseline_data_binary.columns,
    )
    baseline_data_binary = pd.get_dummies(baseline_data_binary, drop_first=True)
    # Combine continuous, categorical, and binary variables
    baseline_data = pd.concat(
        [baseline_data_cont, baseline_data_cate, baseline_data_binary], axis=1
    )

    follow_up_vars = meta[(meta["io"] == "out")].index
    follow_up_predictors = follow_up_vars[
        ~(follow_up_vars.str.contains("death") | follow_up_vars.str.contains("graft"))
    ]
    follow_up_outcomes = follow_up_vars[
        (follow_up_vars.str.contains("death") | follow_up_vars.str.contains("graft"))
    ]

    follow_up_vars_cate = meta[(meta["io"] == "out") & (meta["type"] == "cat")].index
    follow_up_vars_cont = meta[(meta["io"] == "out") & (meta["type"] == "cont")].index

    follow_up_predictors_total = data[follow_up_predictors].dropna(thresh=0.9, axis=1)
    follow_up_predictors_total = follow_up_predictors_total.loc[
        :, [c for c in follow_up_predictors_total.columns if "t1" not in c]
    ]

    follow_up_predictors_cate = follow_up_predictors_total.loc[
        :, follow_up_predictors_total.columns.isin(follow_up_vars_cate)
    ]

    follow_up_predictors_cate = follow_up_predictors_cate.loc[
        :, follow_up_predictors_cate.nunique() > 1
    ]
    follow_up_predictors_cont = follow_up_predictors_total.loc[
        :, follow_up_predictors_total.columns.isin(follow_up_vars_cont)
    ]
    follow_up_predictors_cate_vars_binary = [
        col
        for col in follow_up_predictors_cate.columns
        if follow_up_predictors_cate[col].nunique() == 2
    ]
    follow_up_predictors_cate_bianry = follow_up_predictors_cate[
        follow_up_predictors_cate_vars_binary
    ]
    if len(follow_up_predictors_cate_vars_binary) < len(
        follow_up_predictors_cate.columns
    ):
        ## handling multicategroical variables separately
        pass
    else:
        follow_up_predictors_cate = follow_up_predictors_cate_bianry

    cate_imputer = SimpleImputer(strategy="most_frequent")
    follow_up_predictors_cate = pd.DataFrame(
        cate_imputer.fit_transform(follow_up_predictors_cate),
        index=follow_up_predictors_cate.index,
        columns=follow_up_predictors_cate.columns,
    )

    follow_up_predictors_cont = follow_up_predictors_total.loc[
        :, follow_up_predictors_total.columns.isin(follow_up_vars_cont)
    ]
    follow_up_predictors_total = pd.concat(
        [follow_up_predictors_cont, follow_up_predictors_cate], axis=1
    )

    for year in range(13, 0, -1):
        old_year_str = f"_y{year}"
        new_year_str = f"_year_{year}/"
        # Replace only the exact matches, ensuring y13 won't be replaced by year_1_3
        follow_up_predictors_total.columns = (
            follow_up_predictors_total.columns.str.replace(
                old_year_str, new_year_str, regex=False
            )
        )

        follow_up_predictors_cont.columns = (
            follow_up_predictors_cont.columns.str.replace(
                old_year_str, new_year_str, regex=False
            )
        )
        follow_up_predictors_cate.columns = (
            follow_up_predictors_cate.columns.str.replace(
                old_year_str, new_year_str, regex=False
            )
        )

    # Now df.columns will have the updated column names

    follow_up_predictors_vars_by_year = {}
    for year in range(13, 0, -1):
        year_str = f"y{year}"
        follow_up_predictors_vars_by_year[year_str] = [
            col
            for col in follow_up_predictors_total.columns
            if re.search(f"_year_{year}/", col)
        ]

    return (
        data,
        baseline_data,
        baseline_data_cont,
        follow_up_predictors_total,
        follow_up_predictors_vars_by_year,
        follow_up_predictors_cont,
    )


def clean_follow_up_names(name):
    """
    Clean follow-up measurement names by removing year indicators
    """
    # Remove patterns like 'CUM2_year_X/' or '_year_X/
    name = re.sub(r"_year_\d+/?", "", name)
    name = re.sub(r"_year\d+/?", "", name)
    return name.rstrip("/")


def transform_outcomes(df):
    """
    Transform outcome data based on two rules:
    1. If death=1, set all following death and graft loss indicators to NaN
    2. If graft_loss=1, set all following graft loss indicators to NaN

    Parameters:
    df (pd.DataFrame): DataFrame with death_y* and graft_loss_y* columns

    Returns:
    pd.DataFrame: Transformed DataFrame
    """
    result = df.copy()

    # Get lists of death and graft loss columns with proper numeric sorting
    def get_year_num(col):
        return int(col.split("y")[-1])

    death_cols = sorted(
        [col for col in df.columns if col.startswith("death_y")], key=get_year_num
    )
    graft_cols = sorted(
        [col for col in df.columns if col.startswith("graft_loss_y")], key=get_year_num
    )

    # Process each row
    for idx in result.index:
        # Check death events
        for i, death_col in enumerate(death_cols):
            if result.loc[idx, death_col] == 1.0:
                # Set all subsequent death indicators to NaN
                for next_col in death_cols[i + 1 :]:
                    result.loc[idx, next_col] = np.nan

                # Set all subsequent graft loss indicators to NaN
                death_year = get_year_num(death_col)
                subsequent_graft_cols = [
                    col for col in graft_cols if get_year_num(col) > death_year
                ]
                for graft_col in subsequent_graft_cols:
                    result.loc[idx, graft_col] = np.nan
                break

        # Check graft loss events
        for i, graft_col in enumerate(graft_cols):
            if result.loc[idx, graft_col] == 1.0:
                # Set all subsequent graft loss indicators to NaN
                for next_col in graft_cols[i + 1 :]:
                    result.loc[idx, next_col] = np.nan
                break

    return result


def get_input_data_for_outcome(
    outcome,
    baseline_data,
    baseline_data_cont,
    follow_up_predictors_total,
    follow_up_predictors_vars_by_year,
    follow_up_predictors_cont,
):
    """
    Get appropriate input data for each outcome based on 1-year prediction horizon
    with cleaned follow-up variable names
    """
    # Extract the year from the outcome
    match = re.search(r"_y(\d+)", outcome)
    if not match:
        raise ValueError(f"Could not extract year from outcome: {outcome}")

    outcome_year = int(match.group(1))

    if outcome_year == 1:
        # For year 1 outcomes, use only baseline data
        return baseline_data, baseline_data_cont.columns.tolist()
    else:
        # For other years, use baseline + previous year data
        previous_year = outcome_year - 1
        follow_input_vars = follow_up_predictors_vars_by_year[f"y{previous_year}"]

        # Create a mapping of original column names to cleaned names
        name_mapping = {col: clean_follow_up_names(col) for col in follow_input_vars}

        # Get follow-up data with original names
        follow_up_data = follow_up_predictors_total[follow_input_vars].copy()

        # Rename the columns to cleaned names
        follow_up_data.columns = [name_mapping[col] for col in follow_up_data.columns]

        # Combine baseline and follow-up data with cleaned names
        combined_data = pd.concat([baseline_data, follow_up_data], axis=1)

        # Clean the names in follow_up_predictors_cont
        cleaned_follow_up_cont = [
            clean_follow_up_names(col) for col in follow_up_predictors_cont.columns
        ]

        # Get continuous variables with cleaned names
        cont_vars = (
            baseline_data_cont.columns.tolist()
            + follow_up_data.columns[
                follow_up_data.columns.isin(cleaned_follow_up_cont)
            ].tolist()
        )

        return combined_data, cont_vars

def get_input_data_for_outcome_multi_year(outcome, baseline_data, baseline_data_cont, 
                               follow_up_predictors_total, follow_up_predictors_vars_by_year, follow_up_predictors_cont):

    match = re.search(r'_y(\d+)', outcome)
    if not match:
        raise ValueError(f"Could not extract year from outcome: {outcome}")
    outcome_year = int(match.group(1))
    assert outcome_year != 1
    
    previous_year = outcome_year - 1
    follow_input_vars = follow_up_predictors_vars_by_year[f"y{previous_year}"]
    
    name_mapping = {col: clean_follow_up_names(col) for col in follow_input_vars}
    follow_up_data = follow_up_predictors_total[follow_input_vars].copy()
    follow_up_data.columns = [name_mapping[col] for col in follow_up_data.columns]
    combined_data = pd.concat([baseline_data, follow_up_data], axis=1)
    cleaned_follow_up_cont = [clean_follow_up_names(col) for col in follow_up_predictors_cont.columns]
    cont_vars = (
        baseline_data_cont.columns.tolist()
        + follow_up_data.columns[
            follow_up_data.columns.isin(cleaned_follow_up_cont)
        ].tolist()
    )
    return combined_data, cont_vars
        
def preprocess_data(X: pd.DataFrame, cont_vars: List[str], scaler=None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Preprocess data by scaling continuous variables and handling missing values."""
    X_processed = X.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        X_processed.loc[:, cont_vars] = scaler.fit_transform(X_processed[cont_vars])
    else:
        X_processed.loc[:, cont_vars] = scaler.transform(X_processed[cont_vars])
    
    # Handle missing values
    X_processed = X_processed.fillna(0)
    
    return X_processed, scaler

def fit_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series, 
                    model) -> Tuple[float, float, float, float]:
    """Fit model and evaluate performance on validation and test sets."""
    # For small datasets, use all CPU cores
    if hasattr(model, 'n_jobs'):
        model.n_jobs = -1
    
    # Add early stopping for LGBM
    if isinstance(model, lgbm.LGBMClassifier):
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)
    
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'val_auprc': average_precision_score(y_val, y_val_pred),
        'val_auroc': roc_auc_score(y_val, y_val_pred),
        'test_auprc': average_precision_score(y_test, y_test_pred),
        'test_auroc': roc_auc_score(y_test, y_test_pred)
    }
    
    return metrics

def create_cv_folds(input_data, outcomes, n_splits=5, random_state=42):
    np.random.seed(random_state)
    
    unique_patients = outcomes.index.unique()
    
    patient_label_groups = outcomes.groupby(level=0)['label'].agg(lambda x: x.value_counts().index[0])
    patient_labels = patient_label_groups.to_dict()
    
    label_to_patients = defaultdict(list)
    for patient, label in patient_labels.items():
        label_to_patients[label].append(patient)

    folds = []
    
    all_patients = []
    for label in label_to_patients:
        patients = label_to_patients[label]
        np.random.shuffle(patients)
        all_patients.extend(patients)
    
    # Calculate sizes for each fold
    n_patients = len(all_patients)
    fold_size = n_patients // n_splits
    
    # Create folds
    for fold_idx in range(n_splits):
        # Calculate validation set indices
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_splits - 1 else n_patients
        
        # Split patients into validation and non-validation sets
        val_patients = all_patients[val_start:val_end]
        other_patients = all_patients[:val_start] + all_patients[val_end:]
        
        # Further split non-validation patients into train and test
        n_other = len(other_patients)
        n_train = int(n_other * 0.8)  # 80% of remaining data goes to train
        
        train_patients = other_patients[:n_train]
        test_patients = other_patients[n_train:]
        
        # Get all indices for each split
        train_idx = outcomes.loc[train_patients].index
        val_idx = outcomes.loc[val_patients].index
        test_idx = outcomes.loc[test_patients].index
        
        # Store split indices
        fold_split = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        
        folds.append(fold_split)
    
    return folds
