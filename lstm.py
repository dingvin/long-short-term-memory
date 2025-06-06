import os
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\asus\Documents\code\lstm\templates\img'

def create_and_fit_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model

def prediction_function(crypto_currency):
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime.now()

    df = yf.download(f'{crypto_currency}-USD', start=start_date, end=end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']].values)

    lookback = 60
    x_train, y_train = [], []

    for i in range(lookback, len(df_scaled)):
        x_train.append(df_scaled[i - lookback:i, 0])
        y_train.append(df_scaled[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_start = dt.datetime(2024, 1, 1)
    test_end = dt.datetime.now()

    df_test = yf.download(f'{crypto_currency}-USD', start=test_start, end=test_end)
    actual_prices = df_test['Close'].values

    df_total = pd.concat((df['Close'], df_test['Close']), axis=0)
    model_inputs = df_total[len(df_total) - len(df_test) - lookback:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test, y_test = [], []

    for i in range(lookback, len(model_inputs)):
        x_test.append(model_inputs[i - lookback:i, 0])
        y_test.append(model_inputs[i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = create_and_fit_model(x_train, y_train)

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    acc = r2_score(actual_prices, prediction_prices)
    mae = mean_absolute_error(actual_prices, prediction_prices)
    mse = mean_squared_error(actual_prices, prediction_prices)
    rmse = np.sqrt(mse)
    # Note: Precision, Recall, and F1-score are typically used for classification problems.
    # For regression, these might not be directly applicable.
    # Here, placeholders are added assuming binary classification for demonstration.
    # In practice, these metrics should be adapted to the specific problem.
    precision = precision_score((actual_prices > np.median(actual_prices)).astype(int),
                                (prediction_prices > np.median(prediction_prices)).astype(int))
    recall = recall_score((actual_prices > np.median(actual_prices)).astype(int),
                          (prediction_prices > np.median(prediction_prices)).astype(int))
    f1 = f1_score((actual_prices > np.median(actual_prices)).astype(int),
                  (prediction_prices > np.median(prediction_prices)).astype(int))

    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(prediction_prices, color='green', label='Predicted Prices')
    plt.title(f"{crypto_currency} Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc='upper left')
    prediction_image_filename = f"{crypto_currency}_prediction.png"
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], prediction_image_filename))
    plt.close()

    real_data = model_inputs[len(model_inputs) - lookback:len(model_inputs), 0]
    real_data = np.array(real_data).reshape(-1, 1)

    prediction = []

    for i in range(7):
        rd = np.reshape(real_data, (1, lookback, 1))
        t = model.predict(rd)
        price = scaler.inverse_transform(t)
        prediction.append(price[0][0])
        real_data = np.append(real_data[1:], t)

    prediction = np.array(prediction).reshape(-1, 1)
    final_prediction_prices = np.row_stack((prediction_prices, prediction))

    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(final_prediction_prices, color='green', label='Predicted Prices')
    plt.title(f"{crypto_currency} Price Prediction with 7 days forecast")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc='upper left')
    forecast_image_filename = f"{crypto_currency}_forecast.png"
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], forecast_image_filename))
    plt.close()

    price_today = actual_prices[-1]
    pred_price_today = prediction_prices[-1][0]
    max_price = max(prediction[:, 0])
    min_price = min(prediction[:, 0])
    upside = ((max_price - pred_price_today) * 100) / pred_price_today
    downside = ((min_price - pred_price_today) * 100) / pred_price_today

    return {
        'upside': upside,
        'downside': downside,
        'prediction_image': prediction_image_filename,
        'forecast_image': forecast_image_filename,
        'r2_score': acc,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crypto_currency = request.form['crypto_currency']
    result = prediction_function(crypto_currency)
    return render_template('result.html', result=result)

@app.route('/templates/img/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
