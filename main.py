# Bu kod hücresi main.py dosyasında bulunmalıdır.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from flask import Flask, render_template


app = Flask(__name__)


def read_data(data):
  """ Reads the data as a Pandas DataFrame. """
  df = pd.read_csv(data, nrows=600)
  # df.columns = Index(['Datetime', 'AEP_MW'], dtype='object')
  df['Datetime'] = pd.to_datetime(df['Datetime'])
  df.set_index('Datetime', inplace=True)
  return df

def initial_tests(data):
  """ Runs initial tests on the data. """
  result = adfuller(data[data.columns[0]])
  print('ADF Statistic:', result[0])
  print('p-value:', result[1])
  print('Critical Values:', result[4])

  if result[1] < 0.05 and result[0] < result[4]['5%']:
        print("Veri durağan.\n")
  else:
        print("Veri durağan değil.\n")


def lineplot(data):
  """ Visualizes the DataFrame and saves the plot as lineplot.png. """
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=data, x=data.index, y=data.columns[0])
  plt.xlabel('Date')
  plt.ylabel('Value')
  plt.title('Time Series Data')
  plt.savefig('lineplot.png')


@app.route('/')
def home():
    """ Renders the home page with the line plot. """
    df = read_data('AEP_hourly_mini.csv')
    initial_test_result = initial_tests(df)
    lineplot(df)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)

  