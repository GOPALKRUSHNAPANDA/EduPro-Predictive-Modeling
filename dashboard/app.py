import streamlit as st
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="EduPro Predictor", page_icon="🎓", layout="wide")

# ==================== STYLING ====================
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .header-banner { padding: 20px 0 16px 0; border-bottom: 2px solid #F0F0F0; margin-bottom: 24px; }
    .header-banner h1 { color: #1A1A2E; font-size: 1.9rem; font-weight: 800; margin: 0; }
    .header-banner p { color: #888888; font-size: 0.98rem; margin-top: 4px; }
    .section-label { font-size: 1.2rem; font-weight: 700; color: #1A1A2E; margin-top: 6px; }
    .section-desc { color: #999999; font-size: 0.9rem; margin-bottom: 14px; }
    .metric-card { background: #F8F7FF; border: 1px solid #E8E5FF; border-radius: 14px; padding: 24px; text-align: center; }
    .metric-card .label { color: #8A7FE0; font-size: 0.82rem; font-weight: 700; letter-spacing: 1px; }
    .metric-card .value { color: #1A1A2E; font-size: 2rem; font-weight: 800; margin-top: 6px; }
    section[data-testid="stSidebar"] { background-color: #FAFAFA; border-right: 1px solid #EEEEEE; }
    section[data-testid="stSidebar"] h2 { color: #1A1A2E; font-weight: 700; font-size: 1.1rem; }
    div.stButton > button { background-color: #6C5CE7; color: white; border: none; border-radius: 10px; padding: 10px 0; font-weight: 700; width: 100%; }
    div.stButton > button:hover { background-color: #5A4BD1; color: white; }
    hr { margin: 26px 0; border-color: #F0F0F0; }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS & DATA ====================
model_enrollment = joblib.load('model_enrollment.pkl')
model_revenue = joblib.load('model_revenue.pkl')
feature_columns = joblib.load('feature_columns.pkl')
importance_df = pd.read_csv('feature_importance.csv')
importance_df.columns = ['Feature', 'Importance']
category_df = pd.read_csv('category_summary.csv')

# ==================== HEADER ====================
st.markdown("""
    <div class="header-banner">
        <h1>🎓 EduPro Course Demand & Revenue Predictor</h1>
        <p>Set course details in the sidebar, then click Predict to see results.</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR INPUTS ====================
with st.sidebar:
    st.header("⚙️ Course Details")
    category = st.selectbox("📚 Course Category", sorted(category_df['CourseCategory'].unique()))
    price = st.slider("💰 Course Price ($)", 0, 500, 100)
    duration = st.slider("⏱️ Duration (hours)", 1, 50, 20)
    rating = st.slider("⭐ Expected Rating", 1.0, 5.0, 3.5)
    teacher_rating = st.slider("👩‍🏫 Instructor Rating", 1.0, 5.0, 3.5)
    teacher_experience = st.slider("📈 Instructor Experience (yrs)", 0, 25, 5)
    st.write("")
    predict_clicked = st.button("🔮 Predict")

# ==================== PREDICTIONS ====================
st.markdown('<p class="section-label">🎯 Predictions</p>', unsafe_allow_html=True)

if predict_clicked:
    input_dict = {col: 0 for col in feature_columns}
    input_dict['CoursePrice'] = price
    input_dict['CourseDuration'] = duration
    input_dict['CourseRating'] = rating
    input_dict['AvgTeacherRating'] = teacher_rating
    input_dict['AvgTeacherExperience'] = teacher_experience
    cat_col = 'CourseCategory_' + category
    if cat_col in input_dict:
        input_dict[cat_col] = 1
    input_df = pd.DataFrame([input_dict])[feature_columns]

    enrollment_pred = model_enrollment.predict(input_df)[0]
    revenue_pred = model_revenue.predict(input_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">PREDICTED ENROLLMENT</div>
                <div class="value">{int(enrollment_pred)} students</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">PREDICTED REVENUE</div>
                <div class="value">${revenue_pred:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown('<p class="section-desc">👈 Set course details and click Predict to see results here.</p>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== FEATURE IMPORTANCE ====================
st.markdown('<p class="section-label">🔍 What Drives Enrollment?</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">Top factors the model relies on most when predicting demand.</p>', unsafe_allow_html=True)
st.bar_chart(importance_df.set_index('Feature'), color="#6C5CE7")

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== CATEGORY COMPARISON (reacts to your sidebar selection) ====================
st.markdown('<p class="section-label">🏆 Category-Level Demand Comparison</p>', unsafe_allow_html=True)
st.markdown(f'<p class="section-desc">Your selected category (<b>{category}</b>) is highlighted below.</p>', unsafe_allow_html=True)

category_df['Highlight'] = category_df['CourseCategory'].apply(lambda c: 'Selected' if c == category else 'Other')

tab1, tab2, tab3 = st.tabs(["📈 Enrollment", "💰 Revenue", "📋 Full Table"])

with tab1:
    chart = alt.Chart(category_df).mark_bar().encode(
        x=alt.X('CourseCategory', sort='-y'),
        y='TotalEnrollment',
        color=alt.Color('Highlight', scale=alt.Scale(domain=['Selected','Other'], range=['#6C5CE7','#E0E0E0']), legend=None)
    ).properties(height=350)
    st.altair_chart(chart, use_container_width=True)

with tab2:
    chart2 = alt.Chart(category_df).mark_bar().encode(
        x=alt.X('CourseCategory', sort='-y'),
        y='TotalRevenue',
        color=alt.Color('Highlight', scale=alt.Scale(domain=['Selected','Other'], range=['#1A1A2E','#E0E0E0']), legend=None)
    ).properties(height=350)
    st.altair_chart(chart2, use_container_width=True)

with tab3:
    st.dataframe(category_df.drop(columns=['Highlight']), use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("EduPro Predictive Analytics · Random Forest & Gradient Boosting models")