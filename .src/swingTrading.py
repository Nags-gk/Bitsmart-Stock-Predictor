import datetime as dt
from datetime import timedelta
import numpy as np

def swing_trade(date, predictions, initial_btc_amt, initial_balance):
    days = predictions.shape[0]
    assert days == 7, "Predictions should be for the next 7 days"

    cash_value = initial_balance
    btc_amt = initial_btc_amt

    close_prices = predictions[:, 0]
    low_prices = predictions[:, 1]
    high_prices = predictions[:, 2]

    max_high = np.max(high_prices)
    max_high_index = np.argmax(high_prices)

    min_low = np.inf
    min_low_index = -1

    for i in range(max_high_index + 1, days):
        if low_prices[i] < min_low:
            min_low = low_prices[i]
            min_low_index = i

    date_object = dt.datetime.strptime(date, '%Y-%m-%d').date()
    optimal_sell_date = 'NA'
    optimal_buy_date = 'NA'

    if max_high_index != -1:
        cash_value = btc_amt * max_high
        btc_amt = 0
        optimal_sell_date = str(date_object + timedelta(days=int(max_high_index)+1))

    if min_low_index != -1:
        print(f"Buy on day {min_low_index + 1} at low price {min_low}")
        btc_amt = cash_value / min_low
        cash_value = 0
        optimal_buy_date = str(date_object + timedelta(days=int(min_low_index)+1))

    final_value = btc_amt * close_prices[-1] if btc_amt > 0 else cash_value

    return optimal_sell_date, optimal_buy_date, round(final_value,2)