from utils import *
import lightgbm as lgbm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ML model training script')
    parser.add_argument('--model', type=str, default='RF',
                      help='Model type: RF or LGBM (default: RF)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
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

best_params = pd.read_pickle("configs/model_config_baseline.pkl")

all_predictions = pd.DataFrame()
for outcome_key in ['death_y1', 'graft_loss_y1']:
    print(f"Processing outcome: {outcome}")
    
    input_data, cont_vars = get_input_data_for_outcome(
        outcome, 
        baseline_data, 
        baseline_data_cont,
        follow_up_predictors_total, 
        follow_up_predictors_vars_by_year,
        follow_up_predictors_cont
    )
    y = outcome_data[outcome]
    input_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)
    valid_mask = y.notnull()
    X_valid = input_data[valid_mask]
    y_valid = y[valid_mask]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_valid, y_valid), 1):
        X_train_val, X_test = X_valid.iloc[train_val_idx], X_valid.iloc[test_idx]
        y_train_val, y_test = y_valid.iloc[train_val_idx], y_valid.iloc[test_idx]
        
        # Use custom split function for train/validation split
        X_train, X_val, y_train, y_val = split_ensuring_positives(
            X_train_val, y_train_val, test_size=0.1/0.8, random_state=42+fold
        )
        
        # Preprocess data
        X_train, scaler = preprocess_data(X_train, cont_vars)
        X_val, _ = preprocess_data(X_val, cont_vars, scaler)
        X_test, _ = preprocess_data(X_test, cont_vars, scaler)
        
        model = lgbm.LGBMClassifier(**best_params[outcome_key], verbosity=-1)
        
        model.fit(X_train, y_train["label"], eval_set=[(X_val, y_val["label"])])
        y_test_pred = model.predict_proba(X_test)[:, 1]
        
        y_test["prob"] = y_test_pred
        y_test["outcome_type"] = y_test["outcome"].str.split("_y", expand=True)[0]
        all_predictions = pd.concat([all_predictions, y_test])
        

results_df = pd.DataFrame(all_results)
results_df.to_csv(f'results/next_year_prediction_results_{model_name}_baseline.csv', index=False)