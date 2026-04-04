import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    daily_df = pd.read_csv('all_patients_daily_summary_final.csv')
except FileNotFoundError:
    print("Error: CSV not found")
    exit()

daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df.sort_values(by=['p_num', 'date'], inplace=True)



# lag features 
features_to_lag = [
    'daily_avg_glucose_mgdl',
    'daily_std_glucose_mgdl',
    'daily_tir_percent',
    'daily_carbs_total',
    'daily_insulin_total'
]

for feature in features_to_lag:
    for i in range(1, 4):   
        daily_df[f'{feature}_lag_{i}'] = daily_df.groupby('p_num')[feature].shift(i)


daily_df['insulin_to_carb_ratio'] = daily_df['daily_insulin_total'] / (daily_df['daily_carbs_total'] + 1e-3)
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek

daily_df.dropna(inplace=True)
daily_df['p_num'] = daily_df['p_num'].astype('category')

print(f"Data shape: {daily_df.shape}")

unique_patients = daily_df['p_num'].unique()
split_idx = int(0.8 * len(unique_patients))

train_patients = unique_patients[:split_idx]
test_patients = unique_patients[split_idx:]

train_df = daily_df[daily_df['p_num'].isin(train_patients)]
test_df = daily_df[daily_df['p_num'].isin(test_patients)]

target_column = 'daily_avg_glucose_mgdl'
feature_cols = [col for col in daily_df.columns if col not in ['date', target_column]]

X_train = train_df[feature_cols]
y_train = train_df[target_column]

X_test = test_df[feature_cols]
y_test = test_df[target_column]

print(f"Train patients: {len(train_patients)} | Test patients: {len(test_patients)}")

lgbm = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=200,   
    max_depth=10,
    learning_rate=0.05,
    num_leaves=20
)

lgbm.fit(X_train, y_train, categorical_feature=['p_num'])


y_pred = lgbm.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")


with open('better_glucose_model.pkl', 'wb') as f:
    pickle.dump(lgbm, f)

print("Model saved.")