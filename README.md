# Bitsmart
BitcoinPricePrediction

## Implementation details
* Utilized Multivariate LSTM to predict bitcoin prices.
* Model looks back at past 30 days data to predict close, low and high prices of next 7 days.

## Execution Steps:
* just execute below commands from command prompt to run in local
* set FLASK_APP=main
* set FLASK_ENV=development
* flask run

## Libraries Required
* flask
* tensorflow==2.16.1
* numpy, pandas, matplotlib, sklearn etc.
