import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# ---------------------------
# 1️⃣ Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# 2️⃣ Load preprocessed dataset
# ---------------------------
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

# ---------------------------
# 3️⃣ Encode categorical variables
# ---------------------------
region_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df['Region_Code'] = region_encoder.fit_transform(df['Region'])
df['Category_Code'] = category_encoder.fit_transform(df['Category'])

# ---------------------------
# 4️⃣ Convert Sales & Profit to numeric
# ---------------------------
df['Sales'] = df['Sales'].replace('[\$,]', '', regex=True).astype(float)
df['Profit'] = df['Profit'].replace('[\$,]', '', regex=True).astype(float)

# ---------------------------
# 5️⃣ Define features and targets
# ---------------------------
X = df[['Year', 'Quantity', 'Discount', 'Region_Code', 'Category_Code']]
y_sales = df['Sales']
y_profit = df['Profit']

# ---------------------------
# 6️⃣ Split dataset (optional)
# ---------------------------
X_train, X_test, y_sales_train, y_sales_test = train_test_split(X, y_sales, test_size=0.2, random_state=42)
_, _, y_profit_train, y_profit_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)

# ---------------------------
# 7️⃣ Train models
# ---------------------------
sales_model = LinearRegression()
sales_model.fit(X_train, y_sales_train)

profit_model = LinearRegression()
profit_model.fit(X_train, y_profit_train)

# ---------------------------
# 8️⃣ Save models and encoders
# ---------------------------
pickle.dump(sales_model, open(os.path.join(MODELS_DIR, "sales_model.pkl"), "wb"))
pickle.dump(profit_model, open(os.path.join(MODELS_DIR, "profit_model.pkl"), "wb"))
pickle.dump(region_encoder, open(os.path.join(MODELS_DIR, "region_encoder.pkl"), "wb"))
pickle.dump(category_encoder, open(os.path.join(MODELS_DIR, "category_encoder.pkl"), "wb"))

print("✅ Models trained and saved successfully in /models folder!")
