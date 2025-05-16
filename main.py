
"""
Streamlit App for Future Sales/Stock Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

# Helper function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    if len(data) <= seq_length: # Not enough data to form even one sequence
        return np.array(X), np.array(y)
        
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    
    if X.shape[0] == 0: # Should be caught by initial length check, but as a safeguard
        return X, y
        
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

st.set_page_config(layout="wide")
st.title("Future Sales/Stock Price Prediction using LSTM")

st.markdown("""
Upload a CSV file with 'Date' and 'Close' price columns to train an LSTM model and predict future prices.
The model will be trained on 80% of your data and evaluated on the remaining 20%.
Then, you can specify how many days into the future you want to predict beyond the last date in your dataset.
""")

# Add a check for tensorflow at the beginning
try:
    import tensorflow
except ImportError:
    st.error("TensorFlow library not found. Please ensure it is installed in your environment. (e.g., pip install tensorflow)")
    st.stop()


uploaded_file = st.file_uploader("Upload your CSV file (e.g., TataMotors_1year_cleaned.csv)", type="csv")

if uploaded_file is not None:
    try:
        df_original = pd.read_csv(uploaded_file)
        df = df_original.copy()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    st.subheader("Data Preview (First 5 rows)")
    st.write(df.head())

    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("Error: CSV file must contain 'Date' and 'Close' columns.")
        st.stop()

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
    except Exception as e:
        st.error(f"Error processing Date column: {e}. Please ensure 'Date' column is in a recognizable format.")
        st.stop()

    if df.empty:
        st.error("The DataFrame is empty after processing. Please check your CSV file.")
        st.stop()
        
    if df['Close'].isnull().any():
        st.warning("Warning: 'Close' column contains missing values. Attempting to fill with forward fill (ffill).")
        df['Close'] = df['Close'].ffill()
        if df['Close'].isnull().any(): # If still null (e.g. first value(s) were null)
            df['Close'] = df['Close'].bfill() # Try backward fill as a last resort
            if df['Close'].isnull().any():
                st.error("Error: 'Close' column still contains missing values after ffill and bfill. Please clean your data.")
                st.stop()


    # --- Data for Prediction ---
    data_for_prediction = df[['Close']].copy()
    dataset_values = data_for_prediction.values # Keep as numpy array for scaling

    # --- Scaling ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset_values)

    seq_length = 60 # Look_back period

    # --- Train-Test Split for Model Evaluation ---
    train_data_len = int(len(scaled_data) * 0.8)
    
    train_data_scaled = scaled_data[:train_data_len]
    # For X_test, we need data from train_data_len - seq_length up to the end
    # y_test will correspond to scaled_data[train_data_len:]
    test_data_for_sequences_scaled = scaled_data[train_data_len - seq_length:]

    X_train, y_train = create_sequences(train_data_scaled, seq_length)
    X_test, y_test = create_sequences(test_data_for_sequences_scaled, seq_length)
    
    if X_train.shape[0] == 0:
        st.error(f"Not enough data to create training sequences. Need at least {seq_length + 1} data points in the 80% training split (found {len(train_data_scaled)}).")
        st.stop()
    if X_test.shape[0] == 0:
        st.error(f"Not enough data to create testing sequences. Need at least {seq_length + 1} data points for the test set portion used for sequences (found {len(test_data_for_sequences_scaled)}).")
        st.stop()

    # --- Model Building and Training ---
    st.subheader("Model Training")
    with st.spinner(f"Training LSTM model (Epochs: 20, Batch Size: 32, Sequence Length: {seq_length})... This may take a few minutes."):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50, return_sequences=False)) # Last LSTM before Dense
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    st.success("Model training complete!")

    # --- Evaluate on Test Set (Historical Data) ---
    st.subheader("Model Performance on Historical Test Data (last 20% of data)")
    test_predictions_scaled = model.predict(X_test)
    test_predictions_inversed = scaler.inverse_transform(test_predictions_scaled)
    
    actual_prices_inversed = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Dates for the test predictions. y_test corresponds to original data from index train_data_len onwards.
    test_dates = df.index[train_data_len : train_data_len + len(actual_prices_inversed)]
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("Performance Metrics on Test Set:")
        rmse = np.sqrt(mean_squared_error(actual_prices_inversed, test_predictions_inversed))
        mae = mean_absolute_error(actual_prices_inversed, test_predictions_inversed)
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

        errors = test_predictions_inversed.flatten() - actual_prices_inversed.flatten()
        fig_err, ax_err = plt.subplots()
        ax_err.hist(errors, bins=30, color='orange', edgecolor='black')
        ax_err.set_title('Prediction Error Distribution (Test Set)')
        ax_err.set_xlabel('Error (Predicted - Actual)')
        ax_err.set_ylabel('Frequency')
        ax_err.grid(True)
        st.pyplot(fig_err)

    with col2:
        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(actual_prices_inversed, test_predictions_inversed, alpha=0.6, color='red', label='Predicted vs Actual')
        min_val = min(actual_prices_inversed.min(), test_predictions_inversed.min()) if len(actual_prices_inversed)>0 else 0
        max_val = max(actual_prices_inversed.max(), test_predictions_inversed.max()) if len(actual_prices_inversed)>0 else 1
        ax_scatter.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='Perfect Prediction')
        ax_scatter.set_title('Predicted vs Actual (Test Set)')
        ax_scatter.set_xlabel('Actual Price')
        ax_scatter.set_ylabel('Predicted Price')
        ax_scatter.legend()
        ax_scatter.grid(True)
        ax_scatter.axis('equal')
        st.pyplot(fig_scatter)

    st.subheader("Historical Prices and Model Test Predictions")
    fig_test_preds, ax_test_preds = plt.subplots(figsize=(14, 7))
    ax_test_preds.plot(df.index, df['Close'], label='Full Historical Actual Price', color='blue', alpha=0.7)
    ax_test_preds.plot(test_dates, actual_prices_inversed, label='Actual Price (Test Set)', color='green')
    ax_test_preds.plot(test_dates, test_predictions_inversed, label='Predicted Price (Test Set)', color='red', linestyle='--')
    ax_test_preds.set_xlabel('Date')
    ax_test_preds.set_ylabel('Close Price')
    ax_test_preds.set_title('Historical Data & Test Set Predictions')
    ax_test_preds.legend()
    ax_test_preds.grid(True)
    st.pyplot(fig_test_preds)
    
    # --- Future Price Prediction ---
    st.subheader("Future Price Prediction")
    last_date_in_data = df.index[-1].strftime('%Y-%m-%d')
    st.markdown(f"Predictions will start from the day after the last date in your data ({last_date_in_data}).")
    
    num_future_days = st.number_input("Enter number of days to predict into the future:", min_value=1, max_value=365*5, value=90) # Predict up to 5 years

    last_sequence_scaled = scaled_data[-seq_length:]
    future_predictions_scaled_list = []
    current_sequence_for_future = last_sequence_scaled.reshape((1, seq_length, 1))

    with st.spinner(f"Predicting for {num_future_days} future days..."):
        for _ in range(num_future_days):
            next_pred_scaled = model.predict(current_sequence_for_future, verbose=0)[0,0]
            future_predictions_scaled_list.append(next_pred_scaled)
            new_entry_scaled = np.array([[next_pred_scaled]])
            current_sequence_for_future = np.append(current_sequence_for_future[:, 1:, :], new_entry_scaled.reshape(1,1,1), axis=1)
    
    future_predictions_inversed = scaler.inverse_transform(np.array(future_predictions_scaled_list).reshape(-1, 1))

    last_historical_date_obj = df.index[-1]
    future_dates = pd.to_datetime([last_historical_date_obj + datetime.timedelta(days=i) for i in range(1, num_future_days + 1)])

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions_inversed.flatten()})
    
    st.write(f"Predicted Future Prices for the next {num_future_days} days:")
    st.dataframe(future_df.set_index('Date').style.format({"Predicted Close": "{:.2f}"}))

    st.subheader("Combined Historical Data and Future Predictions Plot")
    fig_future, ax_future = plt.subplots(figsize=(15, 8))
    ax_future.plot(df.index, df['Close'], label='Historical Actual Price', color='blue')
    ax_future.plot(test_dates, test_predictions_inversed, label='Historical Predicted Price (on Test Set)', color='orange', linestyle='--')
    ax_future.plot(future_dates, future_predictions_inversed, label='Future Predicted Price', color='red', linestyle='-.')
    
    target_future_date_default = datetime.date(2025, 4, 24)
    target_future_date = st.date_input("Select a specific future date to highlight on the plot and list predictions after:", value=target_future_date_default)
    
    if target_future_date:
        target_future_date_dt = pd.to_datetime(target_future_date)
        ax_future.axvline(target_future_date_dt, color='purple', linestyle=':', lw=2, label=f'Target Date: {target_future_date_dt.strftime("%Y-%m-%d")}')
    
        predictions_after_target = future_df[future_df['Date'] > target_future_date_dt]
        if not predictions_after_target.empty:
             st.write(f"Predictions specifically after {target_future_date_dt.strftime('%Y-%m-%d')}:")
             st.dataframe(predictions_after_target.set_index('Date').style.format({"Predicted Close": "{:.2f}"}))
        elif not future_df.empty and future_df['Date'].iloc[-1] < target_future_date_dt :
             st.write(f"No predictions extend up to or beyond your target date of {target_future_date_dt.strftime('%Y-%m-%d')}. "
                      f"The current prediction window ends on {future_df['Date'].iloc[-1].strftime('%Y-%m-%d')}. "
                      f"Try increasing the 'number of days to predict into the future'.")
        elif future_df.empty:
            st.write("No future predictions were generated to compare against the target date.")
        else: # Predictions exist but none are strictly *after* the target date within the window
            st.write(f"The prediction window ends on {future_df['Date'].iloc[-1].strftime('%Y-%m-%d')}. "
                     f"There are no predicted values strictly *after* your target date of {target_future_date_dt.strftime('%Y-%m-%d')} within this window, "
                     f"or the target date is beyond the prediction window.")

    ax_future.set_xlabel('Date')
    ax_future.set_ylabel('Close Price')
    ax_future.set_title('Stock Price: Historical Data, Test Predictions, and Future Predictions')
    ax_future.legend(loc='upper left')
    ax_future.grid(True)
    st.pyplot(fig_future)
    
    if not future_dates.empty:
        st.info(f"The model predicts future values starting from {future_dates[0].strftime('%Y-%m-%d')}. "
                f"The displayed 'Target Date' is for visual reference. "
                f"Ensure you predict enough days to cover periods of interest.")

else:
    st.info("Awaiting CSV file upload to begin analysis...")

st.markdown("""
---
**Disclaimer:**
Stock market predictions are inherently speculative and subject to numerous market risks and uncertainties.
The predictions generated by this tool are based on historical data patterns and an LSTM model.
They should **not** be considered financial advice. Past performance is not indicative of future results.
Always conduct your own thorough research or consult with a qualified financial advisor before making any investment decisions.
""")