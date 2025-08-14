import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.packages import *  

# ---------------- Metric Functions ----------------
def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def calculate_metrics(y_true, y_pred, p):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, len(y_true), p)
    return mae, mse, rmse, r2, adj_r2

# ---------------- Data Preprocessing ----------------
def preprocess_data(raw_train_path, raw_test_path, processed_data_path):
    print("[STEP 1] Loading raw data...")
    train_df = pd.read_csv(raw_train_path)
    test_df = pd.read_csv(raw_test_path)
    print(f"[INFO] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    print("[STEP 2] Combining train and test datasets...")
    data = pd.concat(
        [train_df.assign(source="train_data"), test_df.assign(source="test_data")],
        ignore_index=True
    )
    print(f"[INFO] Combined data shape: {data.shape}")

    print("[STEP 3] Cleaning 'Item_Fat_Content' column...")
    data['Item_Fat_Content'] = data['Item_Fat_Content'].str.strip().str.lower()
    mapping = {'lf': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular', 'regular': 'Regular'}
    data['Item_Fat_Content'] = data['Item_Fat_Content'].map(mapping).fillna(data['Item_Fat_Content'])
    print(f"[INFO] Unique values in 'Item_Fat_Content': {data['Item_Fat_Content'].unique()}")

    print("[STEP 4] Filling missing 'Outlet_Size' values...")
    outlet_size_mode = data.groupby('Outlet_Type')['Outlet_Size'] \
                           .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Type'].map(outlet_size_mode))
    print(f"[INFO] Missing 'Outlet_Size' after fill: {data['Outlet_Size'].isnull().sum()}")

    print("[STEP 5] Filling missing 'Item_Weight' values...")
    data['Item_Weight'] = data.groupby(['Item_Identifier','Item_Type','Outlet_Location_Type'])['Item_Weight'] \
                              .transform(lambda x: x.fillna(x.mean()))
    print(f"[INFO] Missing 'Item_Weight' after fill: {data['Item_Weight'].isnull().sum()}")

    print("[STEP 6] Mapping 'Item_Category'...")
    category_map = {
        'Dairy': 'Food', 'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks',
        'Meat': 'Food', 'Fruits and Vegetables': 'Food', 'Household': 'Non-Consumable',
        'Baking Goods': 'Food', 'Snack Foods': 'Food', 'Frozen Foods': 'Food',
        'Breakfast': 'Food', 'Health and Hygiene': 'Non-Consumable', 'Canned': 'Food',
        'Breads': 'Food', 'Starchy Foods': 'Food', 'Others': 'Miscellaneous', 'Seafood': 'Food'
    }
    data['Item_Category'] = data['Item_Type'].map(category_map).fillna('Miscellaneous')
    print(f"[INFO] Unique categories after mapping: {data['Item_Category'].unique()}")

    print("[STEP 7] Calculating 'Outlet_Years'...")
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

    print("[STEP 8] Creating combined features...")
    data['Outlet_Combined'] = data[['Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']] \
                              .astype(str).apply(lambda row: '_'.join([str(x).replace(' ', '_') for x in row]), axis=1)
    data['Item_Combined'] = data[['Item_Fat_Content','Item_Category']] \
                            .astype(str).apply(lambda row: '_'.join([str(x).replace(' ', '_') for x in row]), axis=1)
    print(f"[INFO] Sample 'Outlet_Combined': {data['Outlet_Combined'].head(3).tolist()}")
    print(f"[INFO] Sample 'Item_Combined': {data['Item_Combined'].head(3).tolist()}")

    print("[STEP 9] Applying One-Hot Encoding...")
    data = pd.get_dummies(data, columns=['Outlet_Combined','Item_Combined'])
    print(f"[INFO] Data shape after encoding: {data.shape}")

    print("[STEP 10] Splitting back into train and test datasets...")
    train = data[data['source'] == "train_data"].drop(columns=['source'])
    test = data[data['source'] == "test_data"].drop(columns=['source', 'Item_Outlet_Sales'])
    print(f"[INFO] Preprocessed train shape: {train.shape}, Preprocessed test shape: {test.shape}")

    print("[STEP 11] Saving preprocessed data...")
    processed_data_path.mkdir(parents=True, exist_ok=True)
    train_file = processed_data_path / "train_preprocessed.csv"
    test_file = processed_data_path / "test_preprocessed.csv"
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"[INFO] Preprocessed train saved to {train_file}")
    print(f"[INFO] Preprocessed test saved to {test_file}")

    print("[INFO] Preprocessing completed successfully!")
    return train_file, test_file

# ---------------- Train & Validate Models ----------------
def train_and_validate_models(preprocessed_train_file, models_path, val_results_path, top_n=3):
    print("[INFO] Loading train data...")
    train = pd.read_csv(preprocessed_train_file)
    
    X = train.drop(columns=[
        'Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier',
        'Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Category','Item_Outlet_Sales'
    ])
    y = train['Item_Outlet_Sales']
    
    print("[INFO] Splitting train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models_params = {
        "LinearRegression": (LinearRegression(), {"fit_intercept":[True,False]}),
        "Ridge": (Ridge(), {"alpha":[0.1,1.0,10]}),
        "Lasso": (Lasso(), {"alpha":[0.001,0.01,0.1]}),
        "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators":[100,200], "max_depth":[None,10]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), {"n_estimators":[100,200],"learning_rate":[0.05,0.1]}),
        "SVR": (SVR(), {"C":[0.1,1,10], "kernel":["linear","rbf"]}),
        "XGBRegressor": (XGBRegressor(random_state=42, eval_metric="rmse"), {"n_estimators":[100,200],"learning_rate":[0.05,0.1]})
    }

    results = []
    models_path.mkdir(parents=True, exist_ok=True)
    best_models = {}

    baseline_pred = np.full_like(y_val, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
    print(f"[INFO] Baseline RMSE: {baseline_rmse:.4f}")

    # Train each model
    for name,(model,params) in models_params.items():
        print(f"[INFO] Training {name}...")
        grid = GridSearchCV(model, params, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model

        model_file = models_path / f"{name}_best_model.joblib"
        joblib.dump(best_model, model_file)
        print(f"[INFO] Saved {name} model to {model_file}")

        y_val_pred = np.clip(best_model.predict(X_val), 0, None)
        mae,mse,rmse,r2,adj_r2 = calculate_metrics(y_val, y_val_pred, X_val.shape[1])
        rel_error_pct = 100*(rmse/y_val.mean())

        cv_rmse = -cross_val_score(best_model,X_train,y_train,cv=5,scoring="neg_root_mean_squared_error")
        results.append({
            "Model":name,"Best Params":grid.best_params_,
            "MAE":mae,"MSE":mse,"RMSE":rmse,"R2":r2,"Adjusted R2":adj_r2,
            "Baseline RMSE":baseline_rmse,"Relative Error (%)":rel_error_pct,
            "CV RMSE Mean":cv_rmse.mean(),"CV RMSE Std":cv_rmse.std(),
            "CV RMSE Min":cv_rmse.min(),"CV RMSE Max":cv_rmse.max(),
            "Model Path":str(model_file)
        })

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    print("[INFO] Top models by RMSE:")
    print(results_df.head(top_n)[['Model','RMSE']])

    # ---------------- Top N Combined Model ----------------
    top_models_paths = results_df.head(top_n)['Model Path'].tolist()
    print("[INFO] Calculating combined prediction from top models...")
    X_val_comb = X_val.copy()
    val_preds = pd.DataFrame()
    for model_file in top_models_paths:
        model_name = Path(model_file).stem.replace("_best_model","")
        model = joblib.load(model_file)
        val_preds[model_name] = np.clip(model.predict(X_val_comb), 0, None)

    y_val_combined = val_preds.mean(axis=1)
    mae,mse,rmse,r2,adj_r2 = calculate_metrics(y_val, y_val_combined, X_val.shape[1])
    rel_error_pct = 100*(rmse/y_val.mean())
    
    combined_row = pd.DataFrame([{
        "Model": f"Top_{top_n}_Combined",
        "Best Params": "-",
        "MAE": mae,"MSE": mse,"RMSE": rmse,"R2": r2,"Adjusted R2": adj_r2,
        "Baseline RMSE":baseline_rmse,"Relative Error (%)":rel_error_pct,
        "CV RMSE Mean": "-","CV RMSE Std": "-",
        "CV RMSE Min": "-","CV RMSE Max": "-",
        "Model Path": "-"
    }])
    
    results_df = pd.concat([results_df, combined_row], ignore_index=True)

    val_results_path.mkdir(parents=True, exist_ok=True)
    results_csv = val_results_path / "model_validation_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"[INFO] Saved validation results to {results_csv}")

    return top_models_paths, results_df


# ---------------- Test Prediction ----------------
def predict_test_data(preprocessed_test_file, top_model_paths, output_path):
    print("[INFO] Loading test data...")
    test = pd.read_csv(preprocessed_test_file)
    test_df = test.copy()
    drop_cols = [c for c in test_df.columns if "_Combined" in c]
    test_df.drop(columns=drop_cols, inplace=True)

    X_test = test.drop(columns=[
        'Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier',
        'Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Category'
    ])

    for model_file in top_model_paths:
        model_name = Path(model_file).stem.replace("_best_model","")
        model = joblib.load(model_file)
        y_pred = np.clip(model.predict(X_test), 0, None)
        test_df[model_name+"_pred"] = y_pred

    pred_cols = [c for c in test_df.columns if "_pred" in c]
    test_df['Item_Outlet_Sales'] = test_df[pred_cols].mean(axis=1)

    out_dir = output_path / "test_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "test.csv"
    test_df[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']].to_csv(
        output_file,index=False,float_format="%.4f"
    )
    print(f"[INFO] Saved test predictions to {output_file}")
