import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import altair as alt
import shap
import google.generativeai as genai
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os

# --- CONFIGURATION ---
FIREBASE_WEB_API_KEY = "AIzaSyB0SgzujXUcuqzr8b86WK__yjJm7D2Zy-g"

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Sugar M8")



# --- UI STYLING ---
st.markdown("""
<style>
/* Background Image */
.stApp {
    background: url('https://source.unsplash.com/1600x900/?health,technology,dark') center/cover !important;
}
/* Main Content Container */
.main .block-container {
    background: rgba(0, 0, 0, 0.85); 
    padding: 2rem; 
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
/* Typography */
.title-text {
    font-size: 3rem; 
    font-weight: bold; 
    color: #ff4757; 
    text-shadow: 2px 2px 8px rgba(255,71,87,0.7); 
    text-align: center; 
    margin-bottom: 1rem;
}
h1, h2, h3 { text-align: center; }
/* Button Styling */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    font-weight: bold;
    border: 1px solid #444;
}
/* INSIGHT CARD STYLING (New) */
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
div[data-testid="metric-container"] label {
    color: #aaa; /* Label color */
    font-size: 0.9rem;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #fff; /* Value color */
    font-size: 1.8rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Firebase Initialization ---
if not firebase_admin._apps:
    try:
        if os.path.exists('serviceAccountKey.json'):
            cred = credentials.Certificate('serviceAccountKey.json')
        elif "firebase_creds" in st.secrets:
            import json
            creds_dict = dict(st.secrets["firebase_creds"])
            cred = credentials.Certificate(creds_dict)
        else:
            st.error("⚠️ Key Missing: serviceAccountKey.json not found.")
            st.stop()
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"⚠️ Firebase Initialization Error: {e}")

try:
    db = firestore.client()
except:
    db = None

# --- Authentication ---
def auth_user(email, password, mode="signin"):
    if mode == "signup":
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
    else:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as err:
        return {"error": r.json().get('error', {}).get('message', 'Unknown error')}

# --- Load Models ---
@st.cache_resource
def load_model():
    try:
        with open('next_day_glucose_model_v4.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("CRITICAL ERROR: The file 'next_day_glucose_model_v4.pkl' was not found.")
        return None
    except Exception as e:
        st.error(f"CRITICAL MODEL ERROR: {e}")
        return None

@st.cache_resource
def load_explainer():
    try:
        with open('shap_explainer_v4.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ EXPLAINER ERROR: File 'shap_explainer_v4.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"⚠️ EXPLAINER ERROR: {e}")
        return None

model = load_model()
model = load_model()
explainer = None

if model is not None:
    try:
        # Create the explainer directly from the model
        # This bypasses the version mismatch error entirely
        explainer = shap.TreeExplainer(model) 
    except Exception as e:
        # Fallback for some specific model types (like pipelines)
        try:
             # If model is a pipeline, grab the last step (the actual regressor)
             explainer = shap.TreeExplainer(model.steps[-1][1])
        except:
             st.warning("Could not create SHAP explainer. Predictions will work, but explanations will be hidden.")

# --- Session State ---
if 'user_info' not in st.session_state:
    st.session_state.user_info = None 
if 'user_data' not in st.session_state:
    st.session_state.user_data = []
if 'predicted_value' not in st.session_state:
    st.session_state.predicted_value = None
if 'shap_explanation' not in st.session_state:
    st.session_state.shap_explanation = None
if 'gemini_explanation' not in st.session_state:
    st.session_state.gemini_explanation = None
if 'doctor_view_active' not in st.session_state:
    st.session_state.doctor_view_active = False

# --- Data Logic ---
def sync_data_from_firebase():
    user_id = st.session_state.user_info.get('localId')
    if db is not None and user_id:
        try:
            docs = db.collection('users').document(user_id).collection('glucose_logs').stream()
            data_list = []
            for doc in docs:
                data_list.append(doc.to_dict())
            for item in data_list:
                if isinstance(item['date'], str):
                    item['date'] = datetime.datetime.strptime(item['date'], "%Y-%m-%d").date()
            st.session_state.user_data = data_list
        except Exception as e:
            st.error(f"Error fetching data: {e}")

def create_features_for_user(df):
    df = df.copy()
    df['p_num'] = 1
    features_to_lag = ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl',
                       'daily_tir_percent', 'daily_carbs_total', 'daily_insulin_total']
    for feature in features_to_lag:
        for i in range(1, 8):
            df[f'{feature}_lag_{i}'] = df.groupby('p_num')[feature].shift(i)
    for col in ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl', 'daily_tir_percent']:
        df[f'{col}_roll3'] = df.groupby('p_num')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_roll7'] = df.groupby('p_num')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['insulin_to_carb_ratio'] = df['daily_insulin_total'] / (df['daily_carbs_total'] + 1e-3)
    df['glucose_variability_index'] = df['daily_std_glucose_mgdl'] / (df['daily_avg_glucose_mgdl'] + 1e-3)
    df['carb_per_glucose'] = df['daily_carbs_total'] / (df['daily_avg_glucose_mgdl'] + 1e-3)
    df['glucose_change'] = df.groupby('p_num')['daily_avg_glucose_mgdl'].diff(1)
    df['day_of_week'] = df['date'].dt.dayofweek
    df.fillna(0, inplace=True)
    df['p_num'] = df['p_num'].astype('category')
    return df

def get_gemini_explanation(prediction, shap_df):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        explanation_points = []
        for index, row in shap_df.head(5).iterrows():
            feature_name = row['feature'].replace("_", " ").title()
            impact_dir = "increased" if row['shap_value'] > 0 else "decreased"
            explanation_points.append(f"{feature_name} {impact_dir} the prediction.")

        feature_summary = "\n".join(explanation_points)

        prompt = f"""
        The predicted glucose is {prediction:.1f} mg/dL.
        Key factors:
        {feature_summary}

        Write a short encouraging explanation under 80 words with one actionable tip.
        """

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"Gemini Error: {str(e)}"



# ==========================================
#               LOGIN PAGE
# ==========================================
if st.session_state.user_info is None:
    st.markdown('<div class="title-text">Sugar M8</div>', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #ddd;'>Your AI-Powered Glucose Companion</h3>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    res = auth_user(email, password, "signin")
                    if "error" in res:
                        st.error(f"Login Failed: {res['error']}")
                    else:
                        st.session_state.user_info = res
                        sync_data_from_firebase()
                        st.rerun()
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                if st.form_submit_button("Create Account"):
                    res = auth_user(new_email, new_password, "signup")
                    if "error" in res:
                        st.error(f"Signup Failed: {res['error']}")
                    else:
                        st.session_state.user_info = res
                        if db: db.collection('users').document(res['localId']).set({'created_at': datetime.datetime.now()})
                        st.rerun()

# ==========================================
#               MAIN DASHBOARD
# ==========================================
else:
    # --- SIDEBAR ---
    with st.sidebar:
        # 1. User Profile
        st.markdown("### 👤 User Profile")
        st.write(st.session_state.user_info['email'])
        if st.button("Logout"):
            st.session_state.user_info = None
            st.session_state.user_data = []
            st.rerun()
        st.divider()
        
        # 2. Date
        st.header("🗓️ Today")
        st.markdown(f"### {datetime.datetime.now().strftime('%A, %B %d')}")
        st.divider()

        # 3. Forecast Section
        st.header("🔮 Forecast") 
        days_logged = len(st.session_state.user_data)
        if model is not None:
            if days_logged >= 8:
                if st.button("Predict Tomorrow"):
                    with st.spinner("Processing..."):
                        daily_summary_list = []
                        for day_log in st.session_state.user_data:
                            meal_bgs = [day_log['b_bg'], day_log['l_bg'], day_log['d_bg']]
                            meal_bgs = [bg for bg in meal_bgs if bg > 0]
                            if not meal_bgs: continue
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
                        cols = ['daily_avg_glucose_mgdl', 'daily_std_glucose_mgdl', 'daily_tir_percent', 'daily_carbs_total', 'daily_insulin_total']
                        user_daily_df[cols] = (user_daily_df[cols] - user_daily_df[cols].mean()) / (user_daily_df[cols].std() + 1e-3)
                        
                        feats = create_features_for_user(user_daily_df).tail(1).copy()
                        for col in model.feature_name_:
                            if col not in feats.columns: feats[col] = 0
                        feats = feats[model.feature_name_]
                        
                        pred = model.predict(feats)[0]
                        st.session_state.predicted_value = (pred * target_std) + target_mean
                        
                        if explainer:
                            shap_vals = explainer.shap_values(feats)
                            shap_df = pd.DataFrame({'feature': feats.columns, 'shap_value': shap_vals[0]}).sort_values(by='shap_value', key=abs, ascending=False)
                            st.session_state.shap_explanation = shap_df
                            st.session_state.gemini_explanation = get_gemini_explanation(st.session_state.predicted_value, shap_df)
                
                if st.session_state.predicted_value:
                    st.metric("Tomorrow's Avg", f"{st.session_state.predicted_value:.1f} mg/dL")
                    if st.session_state.gemini_explanation:
                        st.info(st.session_state.gemini_explanation)
            else:
                st.info(f"Log {8 - days_logged} more day(s) of data to unlock the prediction feature.")
        st.divider()
        
        if st.button("Switch to Doctor View"):
            st.session_state.doctor_view_active = True
            st.rerun()

    # --- Main Content Area ---
    if st.session_state.doctor_view_active:
        st.header("👨‍⚕️ Doctor's Summary View")
        if st.button("⬅️ Back to Main Dashboard"):
            st.session_state.doctor_view_active = False
            st.rerun()
        
        if st.session_state.user_data:
            processed_logs = []
            for day_log in st.session_state.user_data:
                for meal_type in ['b', 'l', 'd']:
                    if day_log.get(f'{meal_type}_bg', 0) > 0:
                        meal_name = {'b': 'Breakfast', 'l': 'Lunch', 'd': 'Dinner'}[meal_type]
                        processed_logs.append({
                            'Date': day_log['date'], 'Meal': meal_name,
                            'Glucose': day_log[f'{meal_type}_bg'],
                            'Carbs': day_log[f'{meal_type}_carbs'],
                            'Insulin': day_log[f'{meal_type}_insulin']
                        })
            if processed_logs:
                doc_df = pd.DataFrame(processed_logs).sort_values(by='Date', ascending=False)
                st.dataframe(doc_df, use_container_width=True)
            else:
                st.info("No complete logs found.")
    else:
        # Standard Dashboard
        st.markdown('<div class="title-text">Sugar M8</div>', unsafe_allow_html=True)
        st.header("📈 Your Weekly Glucose Trend")
        
        if len(st.session_state.user_data) > 1:
            df = pd.DataFrame(st.session_state.user_data)
            df['date'] = pd.to_datetime(df['date'])
            plot_data = []
            for _, row in df.iterrows():
                bgs = [x for x in [row['b_bg'], row['l_bg'], row['d_bg']] if x > 0]
                if bgs:
                    plot_data.append({'date': row['date'], 'avg': np.mean(bgs)})
            if plot_data:
                chart = alt.Chart(pd.DataFrame(plot_data)).mark_line(point=True, color='#ff4757').encode(
                    x=alt.X('date', title='Date'), y=alt.Y('avg', title='Glucose (mg/dL)')
                ).properties(background='transparent').interactive()
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Trend chart requires at least 2 days of data.")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.header("✍️ Log Your Daily Data")
            with st.form("daily_log"):
                date = st.date_input("Date", datetime.date.today())
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("Breakfast")
                    bb = st.number_input("BG", key="bb", step=1)
                    bc = st.number_input("Carbs", key="bc", step=1)
                    bi = st.number_input("Insulin", key="bi", step=0.1)
                with c2:
                    st.subheader("Lunch")
                    lb = st.number_input("BG", key="lb", step=1)
                    lc = st.number_input("Carbs", key="lc", step=1)
                    li = st.number_input("Insulin", key="li", step=0.1)
                with c3:
                    st.subheader("Dinner")
                    db_ = st.number_input("BG", key="db", step=1)
                    dc = st.number_input("Carbs", key="dc", step=1)
                    di = st.number_input("Insulin", key="di", step=0.1)
                
                if st.form_submit_button("Save Log"):
                    user_id = st.session_state.user_info['localId']
                    entry = {
                        "date": date.strftime("%Y-%m-%d"),
                        "b_bg": bb, "b_carbs": bc, "b_insulin": bi,
                        "l_bg": lb, "l_carbs": lc, "l_insulin": li,
                        "d_bg": db_, "d_carbs": dc, "d_insulin": di
                    }
                    try:
                        db.collection('users').document(user_id).collection('glucose_logs').document(entry['date']).set(entry)
                        st.success("Log Saved!")
                        sync_data_from_firebase()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        with col2:
            st.header("📜 Recent History")
            if st.session_state.user_data:
                df = pd.DataFrame(st.session_state.user_data)
                st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)
            else:
                st.info("No logs yet.")

        # --- NEW: TODAY'S INSIGHTS SECTION ---
        st.divider()
        st.subheader("⚡ Today's Insights")

        # Logic to calculate today's stats
        today_date = datetime.date.today()
        
        # Find today's log entry
        today_log = None
        for log in st.session_state.user_data:
            if log['date'] == today_date:
                today_log = log
                break
        
        # Default values
        avg_bg, total_carbs, total_insulin, tir = 0, 0, 0, 0

        if today_log:
            # Calculate Average Glucose
            bgs = [today_log.get('b_bg', 0), today_log.get('l_bg', 0), today_log.get('d_bg', 0)]
            valid_bgs = [x for x in bgs if x > 0]
            if valid_bgs:
                avg_bg = int(sum(valid_bgs) / len(valid_bgs))
                # Calculate TIR (Simple estimation based on 3 points)
                in_range = sum(1 for x in valid_bgs if 70 <= x <= 180)
                tir = int((in_range / len(valid_bgs)) * 100)
            
            # Calculate Totals
            total_carbs = today_log.get('b_carbs', 0) + today_log.get('l_carbs', 0) + today_log.get('d_carbs', 0)
            total_insulin = today_log.get('b_insulin', 0) + today_log.get('l_insulin', 0) + today_log.get('d_insulin', 0)

        # Display Cards (using standard Streamlit metrics styled with CSS above)
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.metric("Average Glucose", f"{avg_bg} mg/dL")
        with ic2:
            st.metric("Time In Range", f"{tir}%")
        with ic3:
            st.metric("Total Bolus", f"{total_insulin:.1f} units")
        with ic4:
            st.metric("Total Carbs", f"{total_carbs} g")
