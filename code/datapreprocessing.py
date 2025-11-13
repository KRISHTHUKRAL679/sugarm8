import pandas as pd
import glob
import os
import re


directory_path = '3z/device_data/processed_state'

all_files = glob.glob(os.path.join(directory_path, "P*.csv"))
print(f"Found {len(all_files)} patient files to process.\n")

all_patients_daily_data = []


for file_path in all_files:
    try:
        filename = os.path.basename(file_path)
        p_num_match = re.search(r'P(\d+)\.csv', filename)
        if not p_num_match:
            continue
        p_num = int(p_num_match.group(1))

        df = pd.read_csv(file_path, low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['date'] = df['timestamp'].dt.date
        df['p_num'] = p_num

        
        for col in ['bg', 'insulin', 'carbs']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0

       
        bg_df = df.dropna(subset=['bg'])
        insulin_df = df.dropna(subset=['insulin'])
        carbs_df = df.dropna(subset=['carbs'])

        
        
        if not bg_df.empty:
            bg_summary = (
                bg_df.groupby('date')['bg']
                .agg(['mean', 'std', lambda x: (x.between(3.9, 10.0).sum() / len(x)) * 100])
                .reset_index()
                .rename(columns={'mean': 'daily_avg_glucose_mmol',
                                 'std': 'daily_std_glucose_mmol',
                                 '<lambda_0>': 'daily_tir_percent'})
            )
        else:
            bg_summary = pd.DataFrame(columns=['date', 'daily_avg_glucose_mmol', 'daily_std_glucose_mmol', 'daily_tir_percent'])

        # Insulin: total per day
        insulin_summary = insulin_df.groupby('date')['insulin'].sum().reset_index().rename(columns={'insulin': 'daily_insulin_total'})

        # Carbs: total per day
        carbs_summary = carbs_df.groupby('date')['carbs'].sum().reset_index().rename(columns={'carbs': 'daily_carbs_total'})

        
        daily_summary = (
            pd.merge(bg_summary, insulin_summary, on='date', how='outer')
            .merge(carbs_summary, on='date', how='outer')
        )

        
        daily_summary['daily_carbs_total'] = daily_summary['daily_carbs_total'].fillna(0)
        daily_summary['daily_insulin_total'] = daily_summary['daily_insulin_total'].fillna(0)
        daily_summary['daily_tir_percent'] = daily_summary['daily_tir_percent'].fillna(0)
        daily_summary['p_num'] = p_num

        # --- Convert glucose mmol -> mg/dL ---
        daily_summary['daily_avg_glucose_mgdl'] = (daily_summary['daily_avg_glucose_mmol'] * 18).round(2)
        daily_summary['daily_std_glucose_mgdl'] = (daily_summary['daily_std_glucose_mmol'] * 18).round(2)

        
        final_daily_columns = [
            'p_num', 'date', 'daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl',
            'daily_tir_percent', 'daily_carbs_total', 'daily_insulin_total'
        ]
        daily_summary = daily_summary[final_daily_columns]

        all_patients_daily_data.append(daily_summary)
        print(f"✅ Processed patient {p_num}: {len(daily_summary)} daily records.")

    except Exception as e:
        print(f"⚠️ Could not process {file_path}. Error: {e}")


if all_patients_daily_data:
    final_daily_summary_df = pd.concat(all_patients_daily_data, ignore_index=True)
    daily_output_filename = 'all_patients_daily_summary_final.csv'
    final_daily_summary_df.to_csv(daily_output_filename, index=False)
    print("\n" + "="*60)
    print(f"✅ Daily summary file saved as '{daily_output_filename}'")

   
    model_features = final_daily_summary_df.groupby('p_num').agg(
        mean_glucose_mgdl=('daily_avg_glucose_mgdl', 'mean'),
        day_to_day_variability_mgdl=('daily_avg_glucose_mgdl', 'std'),
        avg_daily_tir=('daily_tir_percent', 'mean'),
        avg_daily_carbs=('daily_carbs_total', 'mean'),
        avg_daily_insulin=('daily_insulin_total', 'mean')
    ).reset_index()

    model_output_filename = 'hba1c_model_features_final.csv'
    model_features.to_csv(model_output_filename, index=False)

    print(f"✅ Model features file saved as '{model_output_filename}'")
    print("="*60)
else:
    print("\nNo valid data found. Output files were not created.")
