import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------
# Load dataset and models
# ---------------------------
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
sales_model = pickle.load(open(os.path.join(MODELS_DIR, "sales_model.pkl"), "rb"))
profit_model = pickle.load(open(os.path.join(MODELS_DIR, "profit_model.pkl"), "rb"))
le_region = pickle.load(open(os.path.join(MODELS_DIR, "region_encoder.pkl"), "rb"))
le_category = pickle.load(open(os.path.join(MODELS_DIR, "category_encoder.pkl"), "rb"))

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Global Superstore App", layout="wide")

# ---------------------------
# Custom Dark Mode CSS with white accents
# ---------------------------
st.markdown("""
<style>
/* Main wrapper for gradient */
.app-wrapper {
    background: linear-gradient(to right, #1e1e2f, #2c2c3e);
    padding: 50px;
    border-radius: 12px;
    color: #ffffff;
}

/* Navbar */
.navbar {
    display: flex;
    justify-content: space-between;
    padding: 20px;
    background-color: #2c2c3e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    border-radius: 8px;
}
.navbar a {
    margin: 0 15px;
    text-decoration: none;
    color: #ffffff; /* White links */
    font-weight: bold;
}
.navbar .logo {
    font-weight: bold;
    font-size: 24px;
    color: #ffffff; /* White logo */
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 80px 20px;
    background: linear-gradient(135deg, rgba(106,90,205,0.85), rgba(131,111,255,0.85));
    border-radius: 12px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    margin-bottom: 50px;
}
.hero h1 {
    font-size: 3em;
    margin-bottom: 10px;
    color: #ffffff; /* White */
}
.hero p {
    font-size: 1.3em;
    color: #e0e0e0;
}
.hero button {
    padding: 12px 30px;
    margin: 15px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 25px;
    border: none;
    cursor: pointer;
    color: white;
    background: #6a5acd;
    transition: background 0.3s, transform 0.2s;
}
.hero button:hover {
    background: #836fff;
    transform: translateY(-2px);
}

/* Feature Cards */
.feature-card {
    background: #2c2c3e;
    border-radius: 15px;
    padding: 25px;
    margin: 15px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    transition: transform 0.3s, box-shadow 0.3s;
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
}
.feature-card h3 {
    color: #ffffff; /* White titles */
    margin-bottom: 10px;
}
.feature-card p {
    color: #dcdcdc;
}

/* Buttons */
.stButton>button {
    background-color: #6a5acd;
    color: white;
    border-radius: 8px;
    padding: 10px 25px;
    border: none;
    transition: background-color 0.3s, transform 0.2s;
}
.stButton>button:hover {
    background-color: #836fff;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Navbar
# ---------------------------
st.markdown("""
<div class="navbar">
    <div class="logo">SuperStoreAI</div>
    <div>
        <a href='#'>Home</a>
        <a href='#'>Features</a>
        <a href='#'>Predict</a>
        <a href='#'>About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Hero Section
# ---------------------------
st.markdown("""
<div class="hero">
    <h1>Predict Your E-Commerce Sales & Profit</h1>
    <p>Interactive AI-powered predictions using Global Superstore data</p>
    <button onclick="window.scrollTo(0,document.body.scrollHeight);">Start Predicting 🚀</button>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Features Section
# ---------------------------
st.markdown("<h2 style='text-align:center; margin-top:50px; color:#ffffff;'>Key Features</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class='feature-card'>
        <h3>📊 Sales Prediction</h3>
        <p>Predict total sales per year, region, and category</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='feature-card'>
        <h3>💰 Profit Forecast</h3>
        <p>Estimate profits across regions and product categories</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='feature-card'>
        <h3>🌟 Interactive UI</h3>
        <p>Use dropdowns and buttons to select year, region, and category</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Prediction Input Section
# ---------------------------
st.markdown("<h2 style='text-align:center; margin-top:50px; color:#ffffff;'>Make a Prediction</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
year = col1.selectbox("Year", sorted(df['Year'].unique()))
region = col2.selectbox("Region", ['All Regions'] + sorted(df['Region'].unique()))
category = col3.selectbox("Category", ['All Categories'] + sorted(df['Category'].unique()))

# Filter dataset and make predictions
subset = df.copy()
if region != 'All Regions':
    subset = subset[subset['Region'] == region]
if category != 'All Categories':
    subset = subset[subset['Category'] == category]

if year > subset['Year'].max():
    base_year = subset['Year'].max()
    subset_year = subset[subset['Year'] == base_year].copy()
    subset_year['Year'] = year
else:
    subset_year = subset[subset['Year'] == year].copy()

total_quantity = subset_year['Quantity'].sum()
avg_discount = subset_year['Discount'].mean()
region_code = le_region.transform([region])[0] if region != 'All Regions' else 0
category_code = le_category.transform([category])[0] if category != 'All Categories' else 0

X_new = pd.DataFrame({
    'Year':[year],
    'Quantity':[total_quantity],
    'Discount':[avg_discount],
    'Region_Code':[region_code],
    'Category_Code':[category_code]
})
pred_sales = sales_model.predict(X_new)[0]
pred_profit = profit_model.predict(X_new)[0]

st.markdown(f"<h3 style='text-align:center; color:#ffffff;'>📈 Predicted Sales: ₹{pred_sales/1e5:.2f} lakh</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align:center; color:#ffffff;'>💰 Predicted Profit: ₹{pred_profit/1e5:.2f} lakh</h3>", unsafe_allow_html=True)
