import streamlit as st
import pandas as pd

import numpy as np
import datetime
import google.generativeai as genai



# ✅ Set your Gemini API key here
genai.configure(api_key="AIzaSyDOTHmTYrOxInkOUYy8CTkBvjEz6KZLYiY")
model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ Correct model name usage

st.set_page_config(page_title='Sugar M8 - Glucose Tracker', layout='centered')

# 🎨 Custom Styling and Navbar
st.markdown(
    """
    <style>
    body {
        background: url('https://source.unsplash.com/1600x900/?insulin') center/cover !important;
        color: white;
    }
    .main .block-container {
        background: rgba(0, 0, 0, 0.85);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(255, 0, 0, 0.2);
        text-align: center;
        max-width: 800px;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4757;
        text-shadow: 2px 2px 8px rgba(255, 71, 87, 0.7);
        text-align: center;
    }
    .nav a {
        color: #ff4757;
        font-weight: bold;
        text-decoration: none;
        padding: 0 15px;
    }
    .nav a:hover {
        color: #f1c40f;
    }
    .button {
        background: #ff4757;
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
    }
    .button:hover {
        background: #f1c40f;
    }
    .a1c-section {
        margin-top: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>

    <div class="title">Sugar M8</div>
    <div class="nav">
        <a href="#">Dashboard</a>
        <a href="#">History</a>
        <a href="#">Profile</a>
    </div>
    """,
    unsafe_allow_html=True
)

# 📅 Date in Sidebar
current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
st.sidebar.markdown(f"## 📅 {current_date}")

# 🧪 Glucose Data
def get_glucose_data():
    if 'glucose_data' not in st.session_state:
        st.session_state.glucose_data = {day: [90, 112, 98] for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']}
    return st.session_state.glucose_data

data = get_glucose_data()


# 📈 Graph Section
st.markdown("### Glucose Level Trend")
fig, ax = plt.subplots(figsize=(4, 2))
for i, meal in enumerate(['Breakfast', 'Lunch', 'Dinner']):
    ax.plot(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], [data[day][i] for day in data], marker='o', linestyle='-', label=meal)
ax.set_ylabel("Glucose Level (mg/dL)")
ax.set_ylim(min(min(data.values())) - 10, max(max(data.values())) + 10)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# ➕ Add Reading
st.markdown("### Add New Reading")
day = st.selectbox("Select Day", list(data.keys()))
breakfast = st.number_input("Breakfast (mg/dL)", min_value=1, step=1, format='%d')
lunch = st.number_input("Lunch (mg/dL)", min_value=1, step=1, format='%d')
dinner = st.number_input("Dinner (mg/dL)", min_value=1, step=1, format='%d')

if st.button("Add Readings", key='add', help='Click to add new glucose readings'):
    data[day] = [breakfast, lunch, dinner]
    st.rerun()

# ⚠️ Glucose Alerts
def check_glucose_levels(breakfast, lunch, dinner):
    messages = []
    if breakfast or lunch or dinner:
        for meal, value in zip(["Breakfast", "Lunch", "Dinner"], [breakfast, lunch, dinner]):
            if value < 70:
                messages.append(f"⚠️ Low {meal} sugar detected ({value} mg/dL). Check your glucose level before sleeping to prevent hypoglycemia.")
            elif value > 180:
                messages.append(f"⚠️ High {meal} sugar detected ({value} mg/dL).")
    return messages

alerts = check_glucose_levels(breakfast, lunch, dinner)
if alerts:
    for alert in alerts:
        st.warning(alert)

# 🧮 Estimated A1C
st.sidebar.markdown("<div class='a1c-section'>Estimated A1C</div>", unsafe_allow_html=True)
average_glucose = np.mean([val for sublist in data.values() for val in sublist])
estimated_a1c = (46.7 + average_glucose) / 28.7
st.sidebar.markdown(f"<div class='a1c-section'>Est. A1C: {estimated_a1c:.2f}%</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='a1c-section'>Avg Glucose: {average_glucose:.1f} mg/dL</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='a1c-section'>Highest: {max(max(data.values()))} mg/dL</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='a1c-section'>Lowest: {min(min(data.values()))} mg/dL</div>", unsafe_allow_html=True)

# 📝 Weekly Summary Generator
st.markdown("### Weekly Summary")

if st.button("Generate Weekly Summary"):
    glucose_data_text = ""
    for day, readings in data.items():
        glucose_data_text += f"{day}: Breakfast={readings[0]} mg/dL, Lunch={readings[1]} mg/dL, Dinner={readings[2]} mg/dL\n"

    prompt = f"""
    Here's a week's worth of glucose readings (in mg/dL). Generate a friendly, encouraging summary:

    {glucose_data_text}

    Include:
    - Average for the week
    - Highest and when
    - A compliment for the best day
    """

    with st.spinner("Generating summary using Gemini..."):
        response = model.generate_content(prompt)
        summary = response.text

    st.success("Here's your weekly glucose summary:")
    st.markdown(f"```markdown\n{summary}\n```")


