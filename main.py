# Bu kod hücresi main.py dosyasında bulunmalıdır.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import plotly.express as px



def read_data(data):
  """ Reads the data as a Pandas DataFrame. """
  df = pd.read_csv(data)
  df['Datetime'] = pd.to_datetime(df['Datetime'])
  df.set_index('Datetime', inplace=True)
  return df


def initial_tests(data):
  """ Runs initial tests on the data. """
  result = adfuller(data[data.columns[0]])
  col1, col2 = st.columns(2)
  with col1:
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
  with col2:
    st.write('Critical Values:', result[4])
  
  #st.write('ADF Statistic:', result[0])
  #st.write('p-value:', result[1])
  #st.write('Critical Values:', result[4])

  if result[1] < 0.05 and result[0] < result[4]['5%']:
      st.write("**⚠ The data is stationary.**")
  else:
      st.write("**⚠ The data is not stationary.**")


def prophet_train(df):
  st.write("Prophet Training has begun.")


def auto_arima_train(df):
  st.write("Auto ARIMA Training has begun.")


def arima_train(df):
  st.write("ARIMA Training has begun.")


def lstm_train(df):
  st.write("LSTM Training has begun.")


def ar_train(df):
  st.write("AR Training has begun.")


def var_train(df):
  st.write("VAR Training has begun.")


def sarima_train(df):
  st.write("SARIMA Training has begun.")


def sarimax_train(df):
  st.write("SARIMAX Training has begun.")

def lr_train(df, problem):
   st.write("Linear Regression Training has begun.")


def logr_train(df, problem):
   st.write("Logistic Regression Training has begun.")


def svm_train(df, problem):
   st.write("SVM Training has begun.")


def rf_train(df, problem):
   st.write("Random Forest Training has begun.")


def dt_train(df, problem):
   st.write("Decision Tree Training has begun.")


def nb_train(df, problem):
   st.write("Naive Bayes Training has begun.")


def nn_train(df, problem):
   st.write("Neural Network Training has begun.")


def lineplot(data):
  """ Visualizes the DataFrame and saves the plot. """
  fig = px.line(
    data, x=data.index, y=data.columns[0],
    labels={'x': 'Date', 'y': 'Value'},
    title='Time Series Data'
  )
  st.plotly_chart(fig, use_container_width=True)


def histogram(data):
  """ Visualizes the DataFrame with a histogram. """
  fig = px.histogram(
     data, x=data.columns[0],
     labels={'x': 'Value'},
     title='Histogram'
  ) 
  st.plotly_chart(fig, use_container_width=True)


def peek_data(df):
  """ Shows 50 lines of the dataset on Streamlit Page. """
  with st.expander("Tabular"):
    showData = st.multiselect(
       'Filter: ', df.columns, default=[]
    )
    st.write(df[showData].head(50))


def st_time_series_scenario(df):
    """ Builds the page if the data is in time-series format. """
    st.subheader("")
    st.subheader("ADF Test Results")
    with st.spinner("Tests are being completed..."):
          initial_tests(df)
          # Plots
          col1, col2 = st.columns(2)
          with col1:
            lineplot(df)
          with col2:
            histogram(df)

          # Data Content
          st.subheader("")
          st.subheader("Data Content")
          peek_data(df)
          st.write("\n\n")

          # Model Selection
          st.subheader("Forecasting")
          st.write("Please choose an algorithm:")
          selected_algorithm = st.selectbox(
            "", [
                "AR", "VAR", "ARMA", "ARIMA", "SARIMA",
                "Auto ARIMA", "SARIMAX", "Prophet", "LSTM"
            ]
          )
          st.write("You selected:", selected_algorithm)
          st.write("\n")
          # Model training
          if st.button("Train the model"):
            st.write("Model training is in progress...")
            if selected_algorithm == "ARIMA": arima_train(df)
            elif selected_algorithm == "AR": ar_train(df)
            elif selected_algorithm == "VAR": var_train(df)
            elif selected_algorithm == "Prophet": prophet_train(df)
            elif selected_algorithm == "SARIMA": sarima_train(df)
            elif selected_algorithm == "Auto ARIMA": auto_arima_train(df)
            elif selected_algorithm == "SARIMAX": sarimax_train(df)
            elif selected_algorithm == "LSTM": lstm_train(df)

          # Evaluation
          st.write("\n\n")
          st.subheader("Evaluation")


def st_normal_scenario(df):
  """ Builds the page if the data does not have a time-series format. """
  st.write("The data doesn't have a time-series format.")
  st.subheader("Data Content")
  peek_data(df)

  # Representation
  st.subheader("Representation")
  st.write("Choose the independent variable columns:")
  selected_cols = st.selectbox("", df.columns, key="ind")
  st.write("You selected:", selected_cols)
  #st.radiobutton()

  st.write("Choose the dependent variable column:")
  selected_col = st.selectbox("", df.columns, key="dep")
  st.write("You selected:", selected_col)

  algos = [
     "Linear Regession", "Logistic Regression", "SVM", "Random Forest",
     "Decision Tree", "Neural Network", "Naive Bayes"
  ]
  st.write("Choose an Algorithm")
  selected_alg = st.selectbox("", algos, key="alg")
  st.write("You selected:", selected_alg)

  problem = st.radio(
     "What type of problem are you looking to solve?", 
     ("Regression", "Classification")
  )

  # Model training
  if st.button("Train the model"):
    st.write("Model training is in progress...")
    if selected_alg == "Linear Regression": lr_train(df, problem)
    elif selected_alg == "Logistic Regression": logr_train(df, problem)
    elif selected_alg == "SVM": svm_train(df, problem)
    elif selected_alg == "Random Forest": rf_train(df, problem)
    elif selected_alg == "Decision Tree": dt_train(df, problem)
    elif selected_alg == "Neural Network": nn_train(df, problem)
    elif selected_alg == "Naive Bayes": nb_train(df, problem)



def streamlit_app():
    """ Builds a streamlit app with user interface. """
    st.subheader("Welcome to rott.ai")
    st.sidebar.image("rottie.jpg", caption="rott.ai")
    st.write("Please choose a file and press the Upload button.")
    uploaded_file = st.file_uploader("Dosya Seç", type=['csv'])

    if uploaded_file is not None:
      filename = uploaded_file.name
      df = read_data(uploaded_file)
      time_series = True

      if time_series:
        st_time_series_scenario(df)

      else:
         st_normal_scenario(df)


hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""


def main():
    st.set_page_config(page_title="Dashboard", page_icon="🐶", layout="wide")
    streamlit_app()


if __name__ == '__main__':
    main()
