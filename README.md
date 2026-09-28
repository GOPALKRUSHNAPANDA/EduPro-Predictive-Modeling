# EduPro Predictive Modeling for Course Demand & Revenue Forecasting

## Overview
This project builds predictive models to forecast course enrollment, course 
revenue, and category revenue for EduPro, an online learning platform. It 
replaces intuition-based course planning with data-driven forecasts to 
support decisions on new course launches, pricing, and instructor onboarding.

## Project Structure
- notebook — Full analysis notebook (Google Colab) data loading, 
  merging, feature engineering, model training, and evaluation
- dashboard — Interactive Streamlit web app for live predictions
- documents — Research paper and executive summary
- images — Screenshots of analysis outputs and the live dashboard

## How to Run the Dashboard
1. Open a terminal inside the `dashboard` folder
2. Install dependencies `pip install streamlit pandas scikit-learn joblib`
3. Run `python -m streamlit run app.py`
4. Open the local URL shown in the terminal (usually httplocalhost8501)

## Models Used
Linear Regression, Ridge Regression, Random Forest Regressor, and 
Gradient Boosting Regressor, trained on 60 courses with feature-engineered 
inputs (price bands, duration buckets, rating tiers, instructor experience).

## Key Findings
- Course revenue is highly predictable (R² ≈ 0.98) since it closely follows 
  price × enrollment for paid courses
- Enrollment count is harder to predict with only 60 courses of history 
  (R² ≈ 0.36), an honest limitation of dataset size
- Course rating, duration, and category are stronger enrollment drivers 
  than price alone

## Author
[Your Name] — Submitted to Unified Mentor