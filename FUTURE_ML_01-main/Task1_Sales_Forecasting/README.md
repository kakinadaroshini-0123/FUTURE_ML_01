# Machine Learning Project: Sales and Demand Forecasting 📈

## 📌 Project Overview
Sales forecasting is a critical component for modern businesses. This project implements an enterprise-grade **Sales and Demand Forecasting System** using historical business data. By leveraging Machine Learning techniques (Linear Regression), the system analyzes past sales trends, weekly seasonality, and overall growth to predict exact future daily sales for the next 30 days.

## 🎯 Objective
- Predict future sales based on historical data.
- Extract revealing trends and seasonality through timestamp features (`dayofweek`, `is_weekend`).
- Provide business-friendly visual outputs via an Interactive Web Dashboard.
- Build a robust model handling missing values natively using time-series interpolation.

## 📊 Business Insights
Accurate sales forecasting holds immense value across industries:
1. **Inventory Planning:** Prevents overstocking (tying up cash flow) and understocking (missing out on potential sales).
2. **Resource Allocation:** Enables strategic scheduling of warehouse staff and budgets during peak seasons (like weekends).
3. **Strategic Decision Making:** Provides actionable insights for marketing teams to design targeted promotions ahead of expected slow sales periods.

## ⚙️ Implementation Steps
1. **Load Data:** Imports the dataset (`sales_data.csv`). If it doesn't exist, a synthetic dataset is generated dynamically to simulate real-world conditions.
2. **Data Cleaning:** Identifies and handles missing values using time-series interpolation (`ffill` & `bfill`).
3. **Feature Engineering:** Extracts temporal features (`Day`, `Month`, `DayOfWeek`, `IsWeekend`) to help the ML model learn deeply embedded chronological patterns.
4. **Model Training:** Utilizes Scikit-learn's `LinearRegression` for accurate predictions on seasonal curves.
5. **Evaluation:** Computes regression metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
6. **Web Dashboard:** Deploys a complete full-stack web application using Streamlit to visualize the predictions interactively.

## 🛠 Tools & Technologies
- **Python 3.x**
- **Streamlit** (Web Application Framework)
- **Pandas Dataframes** (Data manipulation)
- **NumPy** (Numerical operations)
- **Scikit-learn** (Machine Learning model and metrics)
- **Matplotlib** (Data Visualization)

## 📂 Project Structure
```text
📁 Task1_Sales_Forecasting
│-- 📝 app.py                 # The interactive Streamlit Web Dashboard (contains all ML logic)
│-- 📊 sales_data.csv         # The dataset file (auto-generated if missing)
│-- 📈 forecast.png           # Output visualization graph of the predictions
│-- 📖 README.md              # Project documentation file (this file)
```

## 🚀 How to Run

### Interactive Web Dashboard (Recommended ✨)
We built a professional web application for this ML project!
1. Open your terminal in the `Task1_Sales_Forecasting` folder.
2. Run the following command:
```bash
streamlit run app.py
```
3. A browser window will automatically open with the fully interactive **Sales Forecasting Dashboard**. Just click the "Run Sales Forecast Model" button!

## 📈 Output
Once run, the model generates an informative graph capturing:
- **Blue Line:** Historical Actual Sales
- **Orange Dashed Line:** Model Fit (Testing Data)
- **Green Solid Line:** Future 30-Day Forecast Prediction
