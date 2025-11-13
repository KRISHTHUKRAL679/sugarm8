import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import altair as alt
import shap
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('next_day_glucose_model_v4.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'next_day_glucose_model_v4.pkl' not found. Please make sure it's in the same folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Load SHAP Explainer ---
@st.cache_resource
def load_explainer():
    try:
        with open('shap_explainer_v4.pkl', 'rb') as f:
            explainer = pickle.load(f)
        return explainer
    except FileNotFoundError:
        st.warning("SHAP explainer file 'shap_explainer.pkl' not found. Explanations will not be available.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the SHAP explainer: {e}")
        return None

model = load_model()
explainer = load_explainer()

# --- Initialize Session State ---
if 'user_data' not in st.session_state:
    st.session_state.user_data = []
if 'predicted_value' not in st.session_state:
    st.session_state.predicted_value = None
if 'shap_explanation' not in st.session_state:
    st.session_state.shap_explanation = None
if 'gemini_explanation' not in st.session_state:
    st.session_state.gemini_explanation = None
# <-- NEW: Add session state for Doctor View
if 'doctor_view_active' not in st.session_state:
    st.session_state.doctor_view_active = False

# --- Feature Engineering Function ---
def create_features_for_user(df):
    df = df.copy()
    df['p_num'] = 1  # single patient

    # Lag features
    features_to_lag = ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl',
                       'daily_tir_percent', 'daily_carbs_total', 'daily_insulin_total']
    for feature in features_to_lag:
        for i in range(1, 8):
            df[f'{feature}_lag_{i}'] = df.groupby('p_num')[feature].shift(i)

    # Rolling features
    for col in ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl', 'daily_tir_percent']:
        df[f'{col}_roll3'] = df.groupby('p_num')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_roll7'] = df.groupby('p_num')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Derived features
    df['insulin_to_carb_ratio'] = df['daily_insulin_total'] / (df['daily_carbs_total'] + 1e-3)
    df['glucose_variability_index'] = df['daily_std_glucose_mgdl'] / (df['daily_avg_glucose_mgdl'] + 1e-3)
    df['carb_per_glucose'] = df['daily_carbs_total'] / (df['daily_avg_glucose_mgdl'] + 1e-3)
    df['glucose_change'] = df.groupby('p_num')['daily_avg_glucose_mgdl'].diff(1)

    # Temporal feature
    df['day_of_week'] = df['date'].dt.dayofweek

    # Fill NaNs
    df.fillna(0, inplace=True)
    df['p_num'] = df['p_num'].astype('category')
    return df

# --- Gemini Explanation Function ---
def get_gemini_explanation(prediction, shap_df):
    """Generates a user-friendly explanation using the Gemini API."""
    try:
        explanation_points = []
        for index, row in shap_df.head(5).iterrows():
            feature_name = row['feature'].replace("_", " ").replace("mgdl", "(mg/dL)").replace("lag 1", "yesterday's").replace("lag 2", "2 days ago's").title()
            impact_dir = "increased" if row['shap_value'] > 0 else "decreased"
            explanation_points.append(f"- '{feature_name}' {impact_dir} the prediction.")

        feature_summary = "\n".join(explanation_points)

        prompt = f"""
        You are a helpful AI assistant for a person managing their diabetes.
        Your task is to explain a glucose prediction in simple, encouraging, and actionable terms.
        Do NOT use technical jargon like 'SHAP values', 'features', or 'model'.

        The predicted average glucose for tomorrow is {prediction:.1f} mg/dL.
        Here are the main factors that influenced this prediction:
        {feature_summary}

        Based on these factors, provide a short, easy-to-understand summary (under 80 words).
        Explain what the user might learn from this. For example, if yesterday's glucose was the biggest factor,
        explain the importance of daily consistency. End with a single, simple, encouraging tip in a new line.
        """
        # --- NOTE: Remember to secure your API key using st.secrets in a real application! ---
        genai.configure(api_key="AIzaSyBK6XIq1vDZQkkXq65humdnn9bBKr7pVLY") # Replace with your actual key
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate an explanation. Error: {e}"

# --- NEW: Doctor View Function ---
def render_doctor_view():
    """Displays a simplified, clean data log for a doctor."""
    st.header("👨‍⚕️ Doctor's Summary View")
    st.markdown("A simplified log of glucose, carbohydrate, and insulin data.")

    if st.button("⬅️ Back to Main Dashboard"):
        st.session_state.doctor_view_active = False
        st.rerun()

    if not st.session_state.user_data:
        st.warning("No data has been logged yet.")
        return

    # Transform the data from wide to long format
    processed_logs = []
    for day_log in st.session_state.user_data:
        # Breakfast
        if day_log.get('b_bg', 0) > 0 or day_log.get('b_carbs', 0) > 0 or day_log.get('b_insulin', 0) > 0:
            processed_logs.append({
                'Date': day_log['date'],
                'Meal': 'Breakfast',
                'Glucose (mg/dL)': day_log['b_bg'],
                'Carbs (g)': day_log['b_carbs'],
                'Insulin (U)': day_log['b_insulin']
            })
        # Lunch
        if day_log.get('l_bg', 0) > 0 or day_log.get('l_carbs', 0) > 0 or day_log.get('l_insulin', 0) > 0:
            processed_logs.append({
                'Date': day_log['date'],
                'Meal': 'Lunch',
                'Glucose (mg/dL)': day_log['l_bg'],
                'Carbs (g)': day_log['l_carbs'],
                'Insulin (U)': day_log['l_insulin']
            })
        # Dinner
        if day_log.get('d_bg', 0) > 0 or day_log.get('d_carbs', 0) > 0 or day_log.get('d_insulin', 0) > 0:
            processed_logs.append({
                'Date': day_log['date'],
                'Meal': 'Dinner',
                'Glucose (mg/dL)': day_log['d_bg'],
                'Carbs (g)': day_log['d_carbs'],
                'Insulin (U)': day_log['d_insulin']
            })

    if not processed_logs:
        st.info("No valid meal entries found in the logs.")
        return

    # Create and display the DataFrame
    doctor_df = pd.DataFrame(processed_logs)
    doctor_df['Date'] = pd.to_datetime(doctor_df['Date'])
    
    # Define the order of meals
    meal_order = ['Breakfast', 'Lunch', 'Dinner']
    doctor_df['Meal'] = pd.Categorical(doctor_df['Meal'], categories=meal_order, ordered=True)
    
    # Sort by date (most recent first) and then by meal
    doctor_df.sort_values(by=['Date', 'Meal'], ascending=[False, True], inplace=True)

    st.dataframe(
        doctor_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn(
                "Date",
                format="ddd, MMM D, YYYY",
            )
        }
    )

# --- UI Styling ---
st.markdown("""
<style>
.stApp {background: url('https://source.unsplash.com/1600x900/?health,technology,dark') center/cover !important;}
.main .block-container {background: rgba(0, 0, 0, 0.8); padding: 2rem; border-radius: 20px;}
.title {font-size: 3rem; font-weight: bold; color: #ff4757; text-shadow: 2px 2px 8px rgba(255,71,87,0.7); text-align: center; margin-bottom: 1rem;}
.nav { text-align: center; margin-bottom: 2rem; }
.nav a { color: #ff4757; font-weight: bold; text-decoration: none; padding: 0 15px; font-size: 1.1rem; }
h1, h2, h3 { text-align: center; }
</style>
<div class="title">Sugar M8</div>
<div class="nav">
    <a href="#">Dashboard</a> <a href="#">History</a> <a href="#">Profile</a>
</div>
""", unsafe_allow_html=True)

# --- Sidebar: Date & Prediction ---
st.sidebar.header("🗓️ Today")
st.sidebar.markdown(f"## {datetime.datetime.now().strftime('%A, %B %d')}")
st.sidebar.divider()
st.sidebar.header("🔮 Forecast")

days_logged = len(st.session_state.user_data)
if model is not None and days_logged < 8:
    days_needed = 8 - days_logged
    st.sidebar.info(f"Log {days_needed} more day(s) of data to unlock the prediction feature.")

if model is not None and days_logged >= 8:
    if st.sidebar.button("Predict Tomorrow's Glucose"):
        # (The prediction logic remains the same)
        with st.spinner("Analyzing your data and making a prediction..."):
            daily_summary_list = []
            for day_log in st.session_state.user_data:
                meal_bgs = [day_log['b_bg'], day_log['l_bg'], day_log['d_bg']]
                meal_bgs = [bg for bg in meal_bgs if bg > 0]
                if not meal_bgs:
                    continue
                daily_summary_list.append({
                    'date': pd.to_datetime(day_log['date']),
                    'daily_avg_glucose_mgdl': np.mean(meal_bgs),
                    'daily_std_glucose_mgdl': np.std(meal_bgs),
                    'daily_tir_percent': sum(1 for bg in meal_bgs if 70 <= bg <= 180)/len(meal_bgs)*100,
                    'daily_carbs_total': day_log['b_carbs'] + day_log['l_carbs'] + day_log['d_carbs'],
                    'daily_insulin_total': day_log['b_insulin'] + day_log['l_insulin'] + day_log['d_insulin'],
                })
            user_daily_df = pd.DataFrame(daily_summary_list).sort_values(by='date')

            target_mean = user_daily_df['daily_avg_glucose_mgdl'].mean()
            target_std = user_daily_df['daily_avg_glucose_mgdl'].std() + 1e-3
            cols_to_norm = ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl',
                            'daily_tir_percent', 'daily_carbs_total', 'daily_insulin_total']
            user_daily_df[cols_to_norm] = (user_daily_df[cols_to_norm] - user_daily_df[cols_to_norm].mean()) / (user_daily_df[cols_to_norm].std() + 1e-3)
            user_feature_df = create_features_for_user(user_daily_df)
            input_data = user_feature_df.tail(1).copy()
            for col in model.feature_name_:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data['p_num'] = input_data['p_num'].astype('category')
            input_data = input_data[model.feature_name_]
            normalized_prediction = model.predict(input_data)[0]
            actual_prediction = (normalized_prediction * target_std) + target_mean
            st.session_state.predicted_value = actual_prediction

            if explainer:
                shap_values = explainer.shap_values(input_data)
                shap_df = pd.DataFrame({
                    'feature': input_data.columns,
                    'shap_value': shap_values[0]
                })
                shap_df['abs_shap'] = shap_df['shap_value'].abs()
                shap_df = shap_df.sort_values(by='abs_shap', ascending=False)
                st.session_state.shap_explanation = shap_df
                # You might need to add your Gemini API key logic here if you removed it from the function
                st.session_state.gemini_explanation = get_gemini_explanation(actual_prediction, shap_df)

if st.session_state.predicted_value is not None:
    st.sidebar.metric("Predicted Avg. Glucose for Tomorrow", f"{st.session_state.predicted_value:.1f} mg/dL")

# --- Sidebar Explanation Section ---
st.sidebar.divider()
st.sidebar.header("💡 AI-Powered Insights")
if st.session_state.predicted_value is None:
    st.sidebar.info("Predict tomorrow's glucose to see what factors are influencing your levels.")

if st.session_state.gemini_explanation:
    with st.sidebar.expander("What does this prediction mean?", expanded=True):
        st.markdown(st.session_state.gemini_explanation)
else:
    if st.session_state.predicted_value is not None:
        st.sidebar.info("Add a Gemini API key to get a natural language explanation of your results.")

if st.session_state.shap_explanation is not None:
    with st.sidebar.expander("Top Factors Influencing Prediction", expanded=True):
        feature_name_map = {
            'daily_tir_percent_lag_1': "Yesterday's Time in Range (%)",
            'daily_avg_glucose_mgdl_lag_1': "Yesterday's Avg. Glucose",
            'glucose_change': "Day-over-Day Glucose Change",
            'daily_std_glucose_mgdl_lag_1': "Yesterday's Glucose Variability",
            'daily_carbs_total_lag_1': "Yesterday's Total Carbs",
            'daily_insulin_total_lag_1': "Yesterday's Total Insulin",
            'insulin_to_carb_ratio_lag_1': "Yesterday's Insulin-to-Carb Ratio"
        }
        shap_df = st.session_state.shap_explanation.copy()
        features_to_exclude = ['p_num']
        shap_df = shap_df[~shap_df['feature'].str.contains('|'.join(features_to_exclude), case=False)]
        shap_df['feature_display'] = shap_df['feature'].map(feature_name_map).fillna(shap_df['feature'])
        shap_display_df = shap_df.head(7)
        shap_display_df['influence'] = np.where(shap_display_df['shap_value'] > 0, 'Increases Prediction', 'Decreases Prediction')
        chart = alt.Chart(shap_display_df).mark_bar().encode(
            x=alt.X('shap_value:Q', title='Impact on Prediction'),
            y=alt.Y('feature_display:N', sort='-x', title='Factor'),
            color=alt.Color('influence:N',
                          scale=alt.Scale(domain=['Increases Prediction', 'Decreases Prediction'],
                                          range=['#ff4757', '#3498db']),
                          legend=alt.Legend(title="Effect")),
            tooltip=['feature_display', 'shap_value']
        ).properties(title="Top Factors")
        st.altair_chart(chart, use_container_width=True)

# <-- NEW: Add Doctor View button to sidebar ---
st.sidebar.divider()
st.sidebar.header("👨‍⚕️ Professional View")
if st.sidebar.button("Switch to Doctor View"):
    st.session_state.doctor_view_active = True
    st.rerun()
# --- END NEW SECTION ---


# --- MODIFIED: Main Page Logic ---
# This block now checks the session state. If doctor_view_active is True,
# it calls the new function. Otherwise, it shows the regular user dashboard.

if st.session_state.doctor_view_active:
    render_doctor_view()
else:
    # --- This is the original Main Page content ---
    st.header("📈 Your Weekly Glucose Trend")

    if days_logged > 1:
        plot_df_data = []
        for day_log in st.session_state.user_data:
            meal_bgs = [bg for bg in [day_log['b_bg'], day_log['l_bg'], day_log['d_bg']] if bg > 0]
            if meal_bgs:
                plot_df_data.append({
                    'date': day_log['date'],
                    'Daily Average Glucose': np.mean(meal_bgs),
                    'Min Glucose': min(meal_bgs),
                    'Max Glucose': max(meal_bgs)
                })

        if plot_df_data:
            plot_df = pd.DataFrame(plot_df_data)
            plot_df['date'] = pd.to_datetime(plot_df['date'])
            plot_df.sort_values('date', inplace=True)
            plot_df = plot_df.tail(7)

            area = alt.Chart(plot_df).mark_area(opacity=0.3, color='#3498db').encode(
                x=alt.X('date:T', axis=alt.Axis(title='Date', format="%a, %b %d", titleColor='white', labelColor='white', gridColor='#444')),
                y=alt.Y('Min Glucose:Q', axis=alt.Axis(title='Glucose (mg/dL)', titleColor='white', labelColor='white', gridColor='#444')),
                y2='Max Glucose:Q',
                tooltip=['date:T', 'Min Glucose:Q', 'Max Glucose:Q']
            )

            line = alt.Chart(plot_df).mark_line(point=alt.OverlayMarkDef(color="#3498db"), color='#3498db').encode(
                x='date:T',
                y=alt.Y('Daily Average Glucose:Q', title='Glucose (mg/dL)'),
                tooltip=['date:T', 'Daily Average Glucose:Q']
            )

            chart = (area + line).properties(
                title=alt.Title("Your 7-Day Glucose Trend", fontSize=18, color='white'),
                background='transparent'
            ).configure_view(
                strokeWidth=0
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No valid glucose readings found in your recent logs to create a chart.")

    elif days_logged == 1:
        st.info("Log at least two days of data to see your trend graph.")
    else:
        st.info("Your glucose trend will appear here once you start logging data.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.header("✍️ Log Your Daily Data")
        with st.form("daily_log_form"):
            log_date = st.date_input("Date", datetime.date.today())
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Breakfast")
                b_bg = st.number_input("BG (mg/dL)", key="b_bg", min_value=0, step=1)
                b_carbs = st.number_input("Carbs (g)", key="b_carbs", min_value=0, step=1)
                b_insulin = st.number_input("Insulin (U)", key="b_insulin", min_value=0.0, step=0.1, format="%.1f")
            with c2:
                st.subheader("Lunch")
                l_bg = st.number_input("BG (mg/dL)", key="l_bg", min_value=0, step=1)
                l_carbs = st.number_input("Carbs (g)", key="l_carbs", min_value=0, step=1)
                l_insulin = st.number_input("Insulin (U)", key="l_insulin", min_value=0.0, step=0.1, format="%.1f")
            with c3:
                st.subheader("Dinner")
                d_bg = st.number_input("BG (mg/dL)", key="d_bg", min_value=0, step=1)
                d_carbs = st.number_input("Carbs (g)", key="d_carbs", min_value=0, step=1)
                d_insulin = st.number_input("Insulin (U)", key="d_insulin", min_value=0.0, step=0.1, format="%.1f")

            submitted = st.form_submit_button("Add/Update Day's Log")
            if submitted:
                entry_data = {
                    "date": log_date, "b_bg": b_bg, "b_carbs": b_carbs, "b_insulin": b_insulin,
                    "l_bg": l_bg, "l_carbs": l_carbs, "l_insulin": l_insulin,
                    "d_bg": d_bg, "d_carbs": d_carbs, "d_insulin": d_insulin
                }
                dates_only = [item['date'] for item in st.session_state.user_data]
                if log_date in dates_only:
                    st.session_state.user_data[dates_only.index(log_date)] = entry_data
                    st.success(f"Updated log for {log_date}.")
                else:
                    st.session_state.user_data.append(entry_data)
                    st.success(f"Added new log for {log_date}.")
                
                st.session_state.predicted_value = None
                st.session_state.shap_explanation = None
                st.session_state.gemini_explanation = None
                st.rerun()

    with col2:
        st.header("📜 Your Logged History")
        if st.session_state.user_data:
            log_df = pd.DataFrame(st.session_state.user_data).sort_values(by="date", ascending=False)
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("Your logged data will appear here.")