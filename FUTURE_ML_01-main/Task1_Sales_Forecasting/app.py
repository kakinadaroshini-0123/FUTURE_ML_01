import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import datetime
import logging
from io import StringIO

# Configure logging to capture in Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILE = 'sales_data.csv'

def generate_dummy_data(file_path: str):
    """Generates a dummy sales dataset if one does not exist."""
    if os.path.exists(file_path):
        return

    logging.info(f"'{file_path}' not found. Generating a synthetic dataset...")
    
    # 2 years of daily data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    days = np.arange(len(dates))
    
    # Base sales + upward trend + annual seasonality + weekly seasonality + noise
    trend = days * 0.2
    yearly_seasonality = np.sin(days * (2 * np.pi / 365)) * 30
    weekly_seasonality = np.where(dates.weekday >= 5, 20, -10) # Weekends have higher sales
    noise = np.random.normal(0, 10, len(dates))
    
    sales = 150 + trend + yearly_seasonality + weekly_seasonality + noise
    sales = np.maximum(sales, 0) # Ensure no negative sales
    
    df_dummy = pd.DataFrame({'date': dates, 'sales': sales})
    
    # Introduce ~15 missing values randomly to demonstrate data cleaning capabilities
    missing_indices = np.random.choice(df_dummy.index, size=15, replace=False)
    df_dummy.loc[missing_indices, 'sales'] = np.nan
    
    # Format dates as strings typical of CSVs
    df_dummy['date'] = df_dummy['date'].dt.strftime('%Y-%m-%d')
    df_dummy.to_csv(file_path, index=False)
    logging.info(f"Synthetic dataset saved to '{file_path}'.")

def load_and_preprocess(file_path: str) -> pd.DataFrame:
    """Loads CSV, handles missing data, and extracts relevant time features."""
    logging.info("Loading and preprocessing dataset...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

    logging.info(f"Dataset shape: {df.shape}")
    
    # Handle missing values by forward filling, then backward filling (better for time series than median)
    missing_count = df['sales'].isnull().sum()
    if missing_count > 0:
        logging.info(f"Handling {missing_count} missing values using ffill/bfill interpolation...")
        df['sales'] = df['sales'].ffill().bfill()
        
    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort chronologically just in case
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def feature_engineering(df: pd.DataFrame, base_date: datetime.datetime) -> tuple:
    """Creates time-based predictive features from the Date column."""
    df_features = df.copy()
    
    # Expand temporal features
    df_features['month'] = df_features['date'].dt.month
    df_features['day'] = df_features['date'].dt.day
    df_features['dayofweek'] = df_features['date'].dt.dayofweek
    df_features['is_weekend'] = df_features['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Days since start captures the overall linear trend beautifully
    df_features['days_since_start'] = (df_features['date'] - base_date).dt.days
    
    # Define Feature Matrix X and Target Y
    feature_cols = ['days_since_start', 'month', 'day', 'dayofweek', 'is_weekend']
    X = df_features[feature_cols]
    
    if 'sales' in df_features.columns:
        y = df_features['sales']
        return X, y
    return X, None

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series):
    """Trains the Model using Linear Regression."""
    logging.info("Splitting data and training model...")
    # Time Series Split (No shuffling to maintain chronological integrity)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Using Linear Regression for stable predictions
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    logging.info("Linear Regression model successfully trained.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info("--- Model Evaluation Metrics ---")
    logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
    logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
    logging.info(f"R-squared Score (R2): {r2:.2f}")
    
    return model, y_pred, y_test

def predict_future(model, last_date: datetime.datetime, base_date: datetime.datetime, days: int = 30) -> pd.DataFrame:
    """Predicts sales for the specified number of future days."""
    logging.info(f"Forecasting sales for the next {days} days...")
    
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days + 1)]
    future_df = pd.DataFrame({'date': future_dates})
    
    # Engineer the same features for future dates
    X_future, _ = feature_engineering(future_df, base_date)
    
    # Make predictions
    future_df['predicted_sales'] = model.predict(X_future)
    future_df['predicted_sales'] = np.maximum(future_df['predicted_sales'], 0) # Prevent negative sales
    
    logging.info("First 5 days of future forecast:")
    logging.info(future_df[['date', 'predicted_sales']].head(5).to_string(index=False))
    
    return future_df

def plot_results(df: pd.DataFrame, test_dates: pd.Series, y_pred: np.ndarray, future_df: pd.DataFrame):
    """Generates a visualization of historical, test, and future predicted sales."""
    logging.info("Generating visualizations...")
    plt.figure(figsize=(15, 7))

    # Historical Actuality
    plt.plot(df['date'], df['sales'], label='Historical Sales (Actual)', color='#1f77b4', alpha=0.7)

    # Model Test Fit
    plt.plot(test_dates, y_pred, label='Linear Regression Fit (Test Data)', color='#ff7f0e', linestyle='--')

    # Future Forecast
    plt.plot(future_df['date'], future_df['predicted_sales'], label='30-Day Future Forecast', color='#2ca02c', linewidth=2.5)

    plt.title('Sales and Demand Forecasting using Linear Regression', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales Volume', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return plt.gcf()

st.set_page_config(page_title="Sales Forecasting", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Machine Learning Sales & Demand Forecasting")
st.markdown("""
Welcome to the interactive internship dashboard! 
This web application runs the **Linear Regression Model** on your historical data to predict exactly how much demand you will have over the next 30 days.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Control Panel")
    st.write("Click the button below to train the AI model on your dataset (`sales_data.csv`) and generate a fresh forecast.")
    
    if st.button("Run Sales Forecast Model", width='stretch'):
        with st.spinner("Training ML Model... Please wait..."):
            # Capture logs
            log_stream = StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(handler)
            
            try:
                generate_dummy_data(DATA_FILE)
                df = load_and_preprocess(DATA_FILE)
                
                # Base date for calculating 'days_since_start'
                base_date = df['date'].min()
                
                X, y = feature_engineering(df, base_date)
                
                # For plotting, we need the exact dates corresponding to the test set
                test_size = int(len(X) * 0.2)
                
                # Ensure test sizes match by recalculating exact lengths using train_test_split logic
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                test_dates = df['date'].iloc[X_test.index]

                model, y_pred, _ = train_and_evaluate_model(X, y)
                
                last_date = df['date'].max()
                future_df = predict_future(model, last_date, base_date, days=30)
                
                fig = plot_results(df, test_dates, y_pred, future_df)
                
                st.success("Model Trained Successfully!")
                st.session_state['run_complete'] = True
                st.session_state['logs'] = log_stream.getvalue()
                st.session_state['fig'] = fig
                st.session_state['future_df'] = future_df
                
            except Exception as e:
                st.error(f"There was an error: {str(e)}")
                st.session_state['run_complete'] = False

with col2:
    if st.session_state.get('run_complete'):
        st.header("30-Day Business Forecast")
        
        # Display the plot
        if 'fig' in st.session_state:
            st.pyplot(st.session_state['fig'])
            
        st.divider()
        st.subheader("Forecast Summary")
        if 'future_df' in st.session_state:
            st.dataframe(st.session_state['future_df'][['date', 'predicted_sales']].head(10))
            
        st.subheader("Console Output & Execution Logs")
        with st.expander("View raw terminal output"):
            st.code(st.session_state['logs'])
    else:
        st.info("Click the button on the left to execute the model and visualize the results!")
