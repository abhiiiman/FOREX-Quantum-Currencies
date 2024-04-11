import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from stocknews import StockNews
from datetime import datetime
from datetime import date
# from currency_converter import CurrencyConverter
from forex_python.converter import CurrencyRates
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from plotly import graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import joblib
from mango import Tuner
from sklearn.metrics import mean_squared_error, r2_score

global main_df

st.set_page_config(
    page_title="Quantum Currencies - ForEX",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/abhiiiman',
        'Report a bug': "https://www.github.com/abhiiiman",
        'About': "## Quantum Currencies ForEX Time Series Forecasting By The HardCoders 🦾"
    }
)

st.title("Quantum Currencies 💸")
st.markdown("##### ForEX Dashboard 🖥️")
st.markdown("##### Time Series Forecasting Model ⏳")

with st.sidebar:
    st.header("Dataset Configuration ⚙️")

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

upload_file = st.sidebar.file_uploader("Choose a File 📂")

if upload_file is None:
    st.info("Upload your Dataset file to get started ⬆️")
    st.stop()

st.markdown(f"`DataSet Loaded : {upload_file.name}`")

# loading the dataset here
df = load_data(upload_file)
st.dataframe(df)

with st.sidebar.popover("View Available Models "):
    st.write("Model Options :\n1. ARIMA 🤖\n2. LSTM 🤖\n3. FB Prophet 🤖")

# Set default dates
default_start_date = datetime(2001, 1, 1)
default_end_date = datetime(2016, 7, 1)

start_date = st.sidebar.date_input('Start Date 📅', value=default_start_date)
end_date = st.sidebar.date_input('End Date 📅', value=default_end_date)

# showing the well formatted dataset here

st.markdown('''
#### Data Preprocessing 🧠
* Imputed the `Null Values`.
* Extracted `Dates` frm the `Quarters`.
* Converted the `Dataframe` to a `suitable format`.
''')
format_df = pd.read_csv(r"Datasets\my_file.csv")
with st.expander("Data Preview"):
    st.dataframe(format_df)

# showing the df for INDIA here

st.markdown(
    "#### `INDIA - USD` Foreign Exchange Rate Data ⚖️"
)
india_df = pd.read_csv(r"Datasets\India.csv")
with st.expander("Data Preview"):
    st.dataframe(india_df)

# data visualization for india here
st.markdown(
    "#### Data Visualization"
)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(india_df['Date'], india_df['INDIA'], marker='o', linestyle='-')
ax.set_title('Exchange Rates for India Over a Period of 16 Years')
ax.set_xlabel('Quarter')
ax.set_ylabel('Exchange Rate')
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# creating the tabs here

visualize_rate, predict_rate, convert_currency, forex_news, about_us = st.tabs(['Visualize Exchange Rates 💹', 'Predict Exchange Rates 🔮', 'Currency Converter 🔁', 'ForEX News 📰', 'About Us⚡'])

with visualize_rate:
    st.title("Visualize Exchange Rates 💹")
    df = format_df
    # Extract the first column (Date_Quarter) to use as index
    df.set_index('Date_Quarter', inplace=True)

    # Extract country names from the DataFrame
    countries = df.columns.tolist()

    # Create a dropdown menu for selecting the country
    selected_country = st.selectbox('Select Country', countries)

    # Plotting time series graph for the selected country
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[selected_country], marker='o', linestyle='-')
    plt.title(f'Exchange Rates for {selected_country} Over Time')
    plt.xlabel('Quarter')
    plt.ylabel('Exchange Rate')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # show the plot
    st.pyplot(plt)

# creating the helper functions for the forecasting here.

main_df = pd.read_csv(r'Datasets\my_file.csv')

# Function to predict using ARIMA model
def arima_predictor(params):
    global data_points
    
    p, d, q = params['p'], params['d'], params['q']
    trend = params['trend']
    model = ARIMA(data_points, order=(p, d, q), trend=trend)
    model_fit = model.fit()
    output = model_fit.forecast()
    prediction = round(output[0], 3)
    return prediction, model_fit.fittedvalues

# Function to perform tuning using ARIMA model
def arima_mango_tuner(args_list):

    global data_points

    evaluated_params = []
    results = []
    for params in args_list:
        try:
            p, d, q = params['p'], params['d'], params['q']
            trend = params['trend']
            model = ARIMA(data_points, order=(p, d, q), trend=trend)
            predictions = model.fit()
            rmse = np.sqrt(mean_squared_error(data_points, predictions.fittedvalues))
            evaluated_params.append(params)
            results.append(rmse)
        except:
            evaluated_params.append(params)
            results.append(1e5)
    return evaluated_params, results

# Function to check for early stopping
def early_stopping(results):
    current_best_objective = results['best_objective']
    return current_best_objective <= 1.7

# Function to plot ARIMA graph
def plot_arima_graph(data_points, fitted_values, prediction, country):
    x_values = list(range(len(data_points)))
    df = pd.DataFrame({'Index': x_values, 'Data Points': data_points, 'Fitted Values': fitted_values})
    fig = px.line(df, x='Index', y=['Data Points', 'Fitted Values'], title=f'ARIMA-MAX Predicted Values vs. Historical ForEx Rates for {country.capitalize()}')
    fig.add_trace(go.Scatter(x=[len(data_points)], y=[prediction], mode='markers', name='Next Prediction'))
    fig.update_layout(xaxis_title=country, yaxis_title='ForEx Rates', legend_title='Legend')
    st.plotly_chart(fig)

def forecast_ARIMA_MAX(data_points, country):
    with st.spinner("Training ARIMA-MAX Model..."):
        # ARIMA parameter space
        param_space = dict(
            p=range(0, 20),
            d=range(0, 5),
            q=range(0, 20),
            trend=['n', 'c', 't', 'ct']  # n is no trend, c is constant term, t is linear trend, ct is both c and t
        )
        config_dict = dict(early_stopping=early_stopping, num_iteration=100)
        tuner = Tuner(param_space, arima_mango_tuner, config_dict)
        results = tuner.minimize()
        params = results['best_params']

        model_results = arima_predictor(params)
        predict, fitted_values = model_results

        st.balloons()

        st.write('Best Parameters:', params)
        st.markdown(f'##### Best Loss: `{results["best_objective"]}`')
        st.markdown(f'### Predicted Q4 Value for **{country.capitalize()}** is `{predict}`')
        st.markdown(f'#### R2 Score for {country.capitalize()}: `{round(r2_score(data_points, fitted_values) * 100, 2)}%`')

        st.subheader("Forecasting Actual VS Predicted Rates 📈")

        plot_arima_graph(data_points, fitted_values, predict, country)

def forecast_ARIMA(country):

    with st.spinner("Training ARIMA Model..."):
        # Data preprocessing here
        df_country = main_df[['Date_Quarter', country]]
        data = df_country[f'{country}']
        X = data.values
        size = int(len(X) * 0.7)
        train, test = X[0:size], X[size:len(X)]
        train2 = X
        data_history, predictions = [x for x in train], []
        history = train2.copy()

        # loading effect here
        progress_bar = st.progress(0)

        # training ARIMA on 5 lags, 1 differential and 3 MA window
        for t in range(len(test)):
            model = ARIMA(data_history, order=(5,1,3))
            model_fit = model.fit()
            output = model_fit.forecast()
            prediction = round(output[0], 3)
            predictions.append(prediction)
            actual = test[t]
            data_history.append(actual)
            progress_bar.progress((t + 1) / len(test))
        
        # Close progress bar
        progress_bar.empty()
        st.balloons()
    
        # loading the Model here
        loaded_model = joblib.load(r'Models\arima_model.pkl')

        # prediction for the next Quarter here
        next_model = ARIMA(history, order=(5,1,3))
        next_model_fit = next_model.fit()
        next_output = next_model_fit.forecast()
        next_prediction = round(next_output[0], 3)
        st.subheader("Predicted Exchange Rate for Next Quarter of 2016 🟢")
        st.success(next_prediction)

        # plotting the actual vs predicted graph here
        st.subheader("Forecasting Actual VS Predicted Rates 📈")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(test, 'r', label = "Actual Rates")
        plt.plot(predictions, 'g', label = 'Predicted Rates')
        plt.title(f"Exchange Rate for {country.capitalize()} over the period of 2001 - 2016")
        plt.xlabel("Time")
        plt.ylabel("Exchange Rate")
        plt.legend()
        st.pyplot(fig)

def forecast_LSTM(country):

    st.balloons()

    # Visualization 1 - normal
    st.subheader("1️⃣ Exchange Rate VS Time Chart")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(main_df[f'{country}'], 'b', label = "Exchange Rates")
    plt.title(f"Exchange Rate for {country} over the period of 2001 - 2016")
    plt.xlabel("Time")
    plt.ylabel("Exchange Rate")
    plt.legend()
    st.pyplot(fig)

    # visualization 2 - moving average of 10
    st.subheader("2️⃣ Exchange Rate VS Time Chart with 10 Moving Average")
    ma10 = main_df[f'{country}'].rolling(10).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(main_df[f'{country}'], 'b', label = "Exchange Rates")
    plt.plot(ma10, 'r', label = 'MA-10')
    plt.title(f"Exchange Rate for {country} over the period of 2001 - 2016")
    plt.xlabel("Time")
    plt.ylabel("Exchange Rate")
    plt.legend()
    st.pyplot(fig)

    # visualization 3 - moving average of 20
    st.subheader("3️⃣ Exchange Rate VS Time Chart with 10 & 20 Moving Average")
    ma20 = main_df[f'{country}'].rolling(20).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(main_df[f'{country}'], 'b', label = "Exchange Rates")
    plt.plot(ma10, 'r', label = 'MA-10')
    plt.plot(ma20, 'g', label = "MA-20")
    plt.title(f"Exchange Rate for {country} over the period of 2001 - 2016")
    plt.xlabel("Time")
    plt.ylabel("Exchange Rate")
    plt.legend()
    st.pyplot(fig)

    # Data preprocessing here
    df_country = main_df[['Date_Quarter', country]]

    # Splitting Data into Training and Testing -> 70-30 split
    data_training = pd.DataFrame(df_country[f'{country}'][0:int(len(df_country)*0.70)])
    data_testing = pd.DataFrame(df_country[f'{country}'][int(len(df_country)*0.70):int(len(df_country))])

    # Performing Min-Max Scaling here
    scaler = MinMaxScaler(feature_range = (0,1))
    data_training_array = scaler.fit_transform(data_training)

    # Loading the LSTM Model here
    model = load_model(r'Models\LSTM_Model_Max.h5')

    # Testing Part
    past_40_days = data_training.tail(40)
    final_df = pd.concat([past_40_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(40, input_data.shape[0]):
        x_test.append(input_data[i-40:i])
        y_test.append(input_data[i, 0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Making the Predictions here
    y_predicted = model.predict(x_test)

    # Inverse the Scaling here
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # visualization 4 - actual forecasting plot
    st.subheader(f"4️⃣ Future Time Series Forecasting | Original vs Predicted")
    fig = plt.figure(figsize = (12, 6))
    plt.plot(y_test, 'b', label = "Original Price")
    plt.plot(y_predicted, 'r', label = "Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Exchange Rate')
    plt.legend()
    st.pyplot(fig)

def forecast_FB(country, period):
    st.balloons()
    
    # setting up the dataframe for the model here.
    df_train = main_df[['Date_Quarter', country]]
    df_train = df_train.rename(columns = {'Date_Quarter' : 'ds', f'{country}' : 'y'})

    # setting up the model here
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    st.subheader('Forecast DataFrame ☑️')
    st.dataframe(forecast.tail())

    # plotting the forecast data here
    st.subheader('Forecast Data 🌟')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.subheader('Forecast Components ➕')
    fig2 = model.plot_components(forecast)
    st.write(fig2)

with predict_rate:
    # setting up the title here.
    st.title("Predict Exchange Rates 🔮")
    # creating a dropdown for the avaialable models here
    selected_model = st.selectbox('Select from Available Models 🤖', options=('ARIMA', 'ARIMA-MAX', 'LSTM', 'FB Prophet'))
    print(selected_model)
    # extracting the columns from the dataframe here
    countries = df.columns[1:]
    # Convert the extracted countries to a list
    country_list = list(countries)
    # Display the list of countries as a selectbox
    selected_country = st.selectbox('Select a country 🌍', country_list)
    #creating the slider for the period here
    n_years = st.slider("Years of Prediction 👇🏻 (Only for FB Prophet Model)", 1, 4)
    period = n_years * 365
    # creating the predict button here
    if st.button(f'Make Future Forecasting for {selected_country.capitalize()}'):
        try:
            if (selected_model == "ARIMA"):
                forecast_ARIMA(selected_country)
            elif (selected_model == "LSTM"):
                forecast_LSTM(selected_country)
            elif (selected_model == 'ARIMA-MAX'):
                data_points = list(main_df[selected_country])
                forecast_ARIMA_MAX(data_points, selected_country)
            else:
                forecast_FB(selected_country, period)
        except ValueError as e:
            st.error(str(e))

    # plotting the actual vs predicted graph here.

with about_us:
    st.title("About Us ⚡")
    st.header("Team HardCoders 🦾")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Assets/A-pfp.png")
        st.subheader("1️⃣ Abhijit Mandal")
        st.markdown('''
            * **`Github`** ⭐  
                https://github.com/abhiiiman
            * **`Linkedin`**  🔗 
                https://linkedin.com/in/abhiiiman
            * **`Portfolio`** 🌐
                https://abhiiiman.github.io/Abhijit-Mandal
        ''')

    with col2:
        st.image("Assets/H-pfp.png")
        st.subheader("2️⃣ Hardik Sharma")
        st.markdown('''
            * **`Github`** ⭐ 
                https://github.com/CodeStrate
            * **`Linkedin`**  🔗
                https://linkedin.com/in/hardik-sharma-0256cs
        ''')

    with col3:
        st.image("Assets/D-pfp.png")
        st.subheader("3️⃣ Divyanshi")
        st.markdown('''
            * **`Github`** ⭐
                https://github.com/Divvyanshiii
            * **`Linkedin`**  🔗 
                https://linkedin.com/in/divyanshi-shrivastav
        ''')

with convert_currency:
    # Initialize CurrencyConverter
    # c = CurrencyConverter()
    c = CurrencyRates()
    # Title
    st.title('Currency Converter 🔁')
    # Currency input fields
    amount = st.number_input('Enter amount to convert 👇', min_value=1, step=1, value=100)
    from_currency = st.text_input('Enter Source Currency 👇', value='INR')
    to_currency = st.text_input('Enter Target Currency 👇', value="USD")
    # Date input field
    conversion_date = st.date_input('Conversion Date 👇', value=date.today())
    # Convert currency when the user clicks the button
    if st.button('Convert'):
        try:
            # converted_amount = c.convert(amount, from_currency, to_currency, date=conversion_date)
            converted_amount = c.convert(from_currency, to_currency, amount)
            st.success(f'{amount} {from_currency} is approximately {converted_amount:.2f} {to_currency}')
        except ValueError as e:
            st.error(str(e))

with forex_news:
    st.title("Top 10 Finance News 🗞️")
    st.subheader("Domain : Stock Market 📉")
    sn = StockNews("finance", save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.markdown(f"### `News {i+1}`")
        st.markdown(f"* {df_news['published'][i]}")
        st.header(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        news_sentiment = df_news['sentiment_summary'][i]
        st.markdown(f'''
            * **Title Sentiment** `{title_sentiment}`  
            * **News Sentiment** `{news_sentiment}`
        ''')
