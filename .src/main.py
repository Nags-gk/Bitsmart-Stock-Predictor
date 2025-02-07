from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from datetime import timedelta
import swingTrading
#from keras.models import model_from_json
#from tensorflow.python.keras import optimizers

#Command to run in cmd(local)
#set FLASK_APP=main
#set FLASK_ENV=development
#flask run

app = Flask(__name__)
TIME_STEP = 30

filename='model/model_3.keras'
dataset = 'data/BTC_USD.csv'
loaded_model = tf.keras.models.load_model(filename)
#loaded_model = pickle.load(open(filename, 'rb'))

btc = pd.read_csv(dataset)
btc = btc.reset_index()
btc['Date'] = pd.to_datetime(btc['Date'], format='%Y-%m-%d')
start_date = '2020-01-01'
btc = btc.loc[(btc['Date'] >= start_date)]
closedf = btc[['Date','Close','Low','High']]
closedf = closedf.reset_index(drop=True)
scaler=MinMaxScaler(feature_range=(-1,1))
closedf[['Close','Low','High']]=scaler.fit_transform(closedf[['Close','Low','High']])

def get_past_data(input_date):
    date_object = dt.datetime.strptime(input_date, '%Y-%m-%d').date()
    start_window = date_object - timedelta(days=TIME_STEP)
    i = int(closedf.index[closedf['Date'] == str(start_window)].tolist()[0])
    X = []
    X.append(closedf[['Close','Low','High']].iloc[i+1:i+TIME_STEP+1])
    return np.array(X)



def predict_next_cycle(input_for_prediction, cycleLength):
  result = None
  for i in range(cycleLength):
    if(i != 0):
      array_reshaped = input_for_prediction.reshape(TIME_STEP, 3)
      array_appended = np.vstack([array_reshaped, predicted_value])
      array_final = array_appended[1:]
      input_for_prediction = array_final.reshape(1, TIME_STEP, 3)
    predicted_value = loaded_model.predict(input_for_prediction)
    ans = scaler.inverse_transform(predicted_value)
    #print(ans[0])
    if result is None:
      result = ans[0]
    else:
      result = np.vstack([result, ans[0]])
    #result.append((ans[0]))
  return result, sum(result[:,0])/7, min(result[:,1]), max(result[:,2])

def predict_next_7_days(input_date):
    input_for_prediction = get_past_data(input_date)
    result, avg_close, min_low, max_high = predict_next_cycle(input_for_prediction, 7)
    return result, round(avg_close, 2), round(min_low,2), round(max_high,2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_bitcoin', methods=['POST'])
def predict_bitcoin():
    #date = request.args.get('input_date')
    x = [str(x) for x in request.form.values()]
    date = x[0]
    max_date = "2024-05-19"
    date_object = dt.datetime.strptime(date, '%Y-%m-%d').date()
    max_date_object = dt.datetime.strptime(max_date, '%Y-%m-%d').date()
    if(date_object > max_date_object):
       date_validation = "You selected date greater than 2024-05-19, Please provide date before 2024-05-19"
       return render_template('index.html', date_validation=date_validation)
    #print(date)
    result, avg_close, min_low, max_high = predict_next_7_days(date)
    #return jsonify(result)
    initial_btc = 100000 / result[0,0]
    sell_date, buy_date, final_amt =  swingTrading.swing_trade(date, result, initial_btc, 0)
    return render_template('index.html',input_date=date, avg_close=avg_close, min_low=min_low, max_high=max_high, sell_date=sell_date, buy_date=buy_date, final_amt=final_amt)

if __name__ == '__main__':
    app.run(host='0.0.0.00',port=8080)
    #app.run(debug=True)