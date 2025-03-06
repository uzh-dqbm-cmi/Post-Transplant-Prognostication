from utils import *
import argparse
import lightgbm as lgbm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="LGBM"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    return parser.parse_args()


args = parse_args()
model_name = args.model

(
    data,
    baseline_data,
    baseline_data_cont,
    follow_up_predictors_total,
    follow_up_predictors_vars_by_year,
    follow_up_predictors_cont,
) = data_extraction()
outcome_data = data[primary_outcomes]
outcome_data = transform_outcomes(outcome_data)

all_input_data = pd.DataFrame()
all_outcomes = pd.DataFrame()
for outcome in primary_outcomes:
    year_match = re.search(r"_y(\d+)", outcome)
    outcome_year = int(year_match.group(1))
    if outcome_year != 1:
        input_data, cont_vars = get_input_data_for_outcome_multi_year(
            outcome,
            baseline_data,
            baseline_data_cont,
            follow_up_predictors_total,
            follow_up_predictors_vars_by_year,
            follow_up_predictors_cont,
        )
        cont_vars.append("year")
        input_data["year"] = int(outcome.split("_y")[1]) - 1
        if "year" not in cont_vars:
            cont_vars.append("year")
        y = outcome_data[[outcome]]
        y["type"] = "death" if "death" in y.columns[0] else "graft_loss"
        y["outcome"] = outcome
        y.rename(columns={outcome: "label"}, inplace=True)
        y["year"] = outcome_year = int(outcome.split("_y")[1]) - 1

        all_input_data = pd.concat([all_input_data, input_data])
        all_outcomes = pd.concat([all_outcomes, y])

best_params = pd.read_pickle("configs/model_config_followup.pkl")

all_predictions = pd.DataFrame()
for outcome_type in ["death", "graft_loss"]:
    sub_input_data = all_input_data.loc[all_outcomes["type"] == outcome_type]
    sub_outcomes = all_outcomes.loc[all_outcomes["type"] == outcome_type]
    sub_input_data = sub_input_data.loc[sub_outcomes["label"].notnull()]
    sub_outcomes = sub_outcomes.loc[sub_outcomes["label"].notnull()][
        ["label", "outcome", "year"]
    ]

    outcome_years = [
        int(re.search(r"_y(\d+)", o).group(1)) for o in sub_outcomes["outcome"]
    ]
    sub_input_data.rename(
        columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace=True
    )
    cv_folds = create_cv_folds(sub_input_data, sub_outcomes, n_splits=5)

    for fold_idx, fold in enumerate(cv_folds):
        train_idx = fold["train_idx"].unique()
        val_idx = fold["val_idx"].unique()
        test_idx = fold["test_idx"].unique()

        X_train, y_train = sub_input_data.loc[train_idx], sub_outcomes.loc[train_idx]
        X_val, y_val = sub_input_data.loc[val_idx], sub_outcomes.loc[val_idx]
        X_test, y_test = sub_input_data.loc[test_idx], sub_outcomes.loc[test_idx]

        X_train, scaler = preprocess_data(X_train, cont_vars)
        X_val, _ = preprocess_data(X_val, cont_vars, scaler)
        X_test, _ = preprocess_data(X_test, cont_vars, scaler)
        
        model = lgbm.LGBMClassifier(**best_params[outcome_key], verbosity=-1)
        model.fit(X_train, y_train["label"], eval_set=[(X_val, y_val["label"])])
        y_test_pred = model.predict_proba(X_test)[:, 1]
        
        y_test["prob"] = y_test_pred
        y_test["outcome_type"] = y_test["outcome"].str.split("_y", expand=True)[0]
        all_predictions = pd.concat([all_predictions, y_test])

# Save results
all_predictions.to_csv(
    f"results/next_year_prediction_results_{model_name}_followup.csv",
    index=False,
)
