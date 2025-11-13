import pandas as pd MoM-CLAM-Code
import lightgbm as lgb
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


try:
    daily_df = pd.read_csv('all_patients_daily_summary_final.csv')
except FileNotFoundError:
    print("❌ Error: 'all_patients_daily_summary_final.csv' not found.")
    exit()

daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df.sort_values(by=['p_num', 'date'], inplace=True)
print(f"✅ Loaded data with {len(daily_df)} rows and columns: {list(daily_df.columns)}")

# --- 2. Feature Engineering ---
print("\n⚙️ Creating advanced features...")

# --- Step 2.1: Per-patient normalization ---
def normalize_group(g):
    cols = ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl',
            'daily_tir_percent', 'daily_carbs_total', 'daily_insulin_total']
    g[cols] = (g[cols] - g[cols].mean()) / (g[cols].std() + 1e-3)
    return g

daily_df = daily_df.groupby('p_num', group_keys=False).apply(normalize_group)

# --- Step 2.2: Lag Features (1–7 days) ---
features_to_lag = [
    'daily_avg_glucose_mgdl',
    'daily_std_glucose_mgdl',
    'daily_tir_percent',
    'daily_carbs_total',
    'daily_insulin_total'
]
for feature in features_to_lag:
    for i in range(1, 8):
        daily_df[f'{feature}_lag_{i}'] = daily_df.groupby('p_num')[feature].shift(i)

# --- Step 2.3: Rolling trends (3-day, 7-day means) ---
for col in ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl', 'daily_tir_percent']:
    daily_df[f'{col}_roll3'] = daily_df.groupby('p_num')[col].transform(lambda x: x.rolling(3).mean())
    daily_df[f'{col}_roll7'] = daily_df.groupby('p_num')[col].transform(lambda x: x.rolling(7).mean())

# --- Step 2.4: Derived ratios & trends ---
daily_df['insulin_to_carb_ratio'] = daily_df['daily_insulin_total'] / (daily_df['daily_carbs_total'] + 1e-3)
daily_df['glucose_variability_index'] = daily_df['daily_std_glucose_mgdl'] / (daily_df['daily_avg_glucose_mgdl'] + 1e-3)
daily_df['carb_per_glucose'] = daily_df['daily_carbs_total'] / (daily_df['daily_avg_glucose_mgdl'] + 1e-3)
daily_df['glucose_change'] = daily_df.groupby('p_num')['daily_avg_glucose_mgdl'].diff(1)

# --- Step 2.5: Temporal feature ---
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek

# Drop NaNs from rolling/lag creation
daily_df.dropna(inplace=True)
daily_df['p_num'] = daily_df['p_num'].astype('category')
print(f"✅ Feature set created. Shape: {daily_df.shape}")

# --- 3. Define Features and Target ---
target_column = 'daily_avg_glucose_mgdl'
feature_cols = [col for col in daily_df.columns if col not in ['date', target_column]]
X = daily_df[feature_cols]
y = daily_df[target_column]

# --- 4. Chronological Train-Test Split ---
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
print(f"\n🧩 Train: {len(X_train)} | Test: {len(X_test)} samples")

# --- 5. LightGBM Training with Faster Grid Search 
print("\n🚀 Starting LightGBM Grid Search...")

lgbm = lgb.LGBMRegressor(random_state=42, device='gpu')

# Reduced grid for speed
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [31, 50],
    'max_depth': [5, -1],
}

tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=1,   # ✅ Limit CPU usage to 1 core
    verbose=2
)

grid_search.fit(
    X_train,
    y_train,
    categorical_feature=['p_num']
)

print("\n🏆 Best parameters found:", grid_search.best_params_)
best_model = grid_search.best_estimator_
print("✨ Model training complete!")

# --- 6. Evaluation ---
print("\n📊 Evaluating model performance...")
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"  -> RMSE: {rmse:.2f} mg/dL")
print(f"  -> R²: {r2:.2f}")

# --- 7. Per-patient R² breakdown ---
print("\n📈 R² by patient:")
for pid in sorted(daily_df['p_num'].unique()):
    mask = X_test['p_num'] == pid
    if mask.sum() > 10:
        r2_pid = r2_score(y_test[mask], y_pred[mask])
        print(f"    Patient {pid}: R² = {r2_pid:.2f}")

# --- 8. Save Model ---
model_filename = 'next_day_glucose_model_v4.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n🎉 Model saved successfully as '{model_filename}'!")


import shap

print("\n🧠 Creating and saving SHAP explainer...")

# Create the explainer using your best-trained model and the training data
explainer = shap.TreeExplainer(best_model)

# Save the explainer object to a file
explainer_filename = 'shap_explainer_v4.pkl'
with open(explainer_filename, 'wb') as f:
    pickle.dump(explainer, f)

print(f"✅ SHAP explainer saved successfully as '{explainer_filename}'!")
