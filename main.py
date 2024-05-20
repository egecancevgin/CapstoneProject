# Bu kod hücresi main.py dosyasında bulunmalıdır.
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



def read_data(data):
  """ Reads the data as a Pandas DataFrame. """
  df = pd.read_csv(data)
  try:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
  except:
    pass
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


def encode_features(df, cates, enc):
  """ Encodes the categorical features based on their encoding type. """
  df_encoded = df.copy()
  if enc == 'One-hot Encoding':
    df_encoded = pd.get_dummies(df_encoded, columns=cates)
  elif enc == 'Label Encoding':
    label_encoders = {}
  for col in cates:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
  return df_encoded


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


def lr_train(df, problem, dep, indeps, train_size):
   st.write("Linear Regression Training has begun.")


def logr_train(df, problem, dep, indeps, train_size):
   st.write("Logistic Regression Training has begun.")


def svm_train(df, problem, dep, indeps, train_size, cates=None, enc=None):
  """ Trains an SVM model. 
      :param: df
      :param: problem
      :param: dep
      :param: train_size
      :param: cates: Names of the categorical columns to encode
      :param: enc: Encoding type, 'One-hot Encoding' or 'Label Encoding'
  """
  st.write("SVM Training has begun.")
  if cates != None and enc !=None:
    df = encode_features(df, cates, enc)
  
  X = df[indeps]
  y = df[dep]
  train_size /= 100
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1-train_size, random_state=42
  )

  if problem == "Classification":
    st.write("SVC is running.")
    st.write("Dependent variables:", dep)
    st.write("Independent variable:", indeps)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Classification Accuracy: {accuracy}")

  elif problem == "Regression":
    st.Write("SVM is running.")
    st.write("Dependent variables:", dep)
    st.write("Independent variable:", indeps)
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Regression Mean Squared Error: {mse}")


def rf_train(df, problem, dep, indeps, train_size):
   st.write("Random Forest Training has begun.")


def dt_train(df, problem, dep, indeps, train_size):
   st.write("Decision Tree Training has begun.")


def nb_train(df, problem, dep, indeps, train_size):
   st.write("Naive Bayes Training has begun.")


def nn_train(df, problem, dep, indeps, train_size):
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
          drop_nan = st.checkbox('Drop NaN values')
          if drop_nan: df = df.dropna()
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
  
  drop_nan = st.checkbox('Drop NaN values')
  if drop_nan: df = df.dropna()
  
  st.subheader("Data Content")
  peek_data(df)

  # Representation
  st.subheader("Representation")
  st.write("Choose the independent variable columns:")
  indep_cols = st.multiselect("", df.columns, key="ind")
  st.write("You selected:", indep_cols)

  st.write("Choose the dependent variable column:")
  dep_col = st.selectbox("", df.columns, key="dep")
  st.write("You selected:", dep_col)

  st.write("Choose the categorical columns to encode:")
  cates = st.multiselect("", df.columns, key="cat")
  st.write("You selected:", cates)
  
  encodings = ["One-hot Encoding, Label Encoding"]
  st.write("Choose the encoding for categorical columns:")
  enc = st.selectbox("", encodings, key="enc")
  st.write("You selected:", enc)

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

  train_split = st.slider(
    'Determine the size of the training set in percent:',
    min_value=0,
    max_value=100,
    value=70,
    step=1
  )

  # Model training
  if st.button("Train the model"):
    st.write("Model training is in progress...")
    if selected_alg == "Linear Regression":
      lr_train(df, problem, dep_col, indep_cols, train_split, cates, enc)
    elif selected_alg == "Logistic Regression":
      logr_train(df, problem, dep_col, indep_cols, train_split, cates, enc)
    elif selected_alg == "SVM":
      svm_train(df, problem, dep_col, indep_cols, train_split, cates, enc)
    elif selected_alg == "Random Forest":
      rf_train(df, problem, dep_col, indep_cols, train_split, cates, enc)
    elif selected_alg == "Decision Tree":
      dt_train(df, problem, dep_col, indep_cols, train_split, cates, enc)
    elif selected_alg == "Neural Network":
      nn_train(df, problem, dep_col, indep_cols, train_split, cates, enc)
    elif selected_alg == "Naive Bayes":
      nb_train(df, problem, dep_col, indep_cols, train_split, cates, enc)


def streamlit_app():
    """ Builds a streamlit app with user interface. """
    st.subheader("Welcome to rott.ai")
    st.sidebar.image("rottie.jpg", caption="rott.ai")
    st.write("Please choose a file and press the Upload button.")
    uploaded_file = st.file_uploader("Dosya Seç", type=['csv'])

    if uploaded_file is not None:
      filename = uploaded_file.name
      df = read_data(uploaded_file)
      time_series = False

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
