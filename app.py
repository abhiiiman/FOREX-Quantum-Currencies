import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from stocknews import StockNews
from datetime import datetime
from datetime import date
# from currency_converter import CurrencyConverter
from forex_python.converter import CurrencyRates
import requests
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import joblib

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
format_df = pd.read_csv("my_file.csv")
with st.expander("Data Preview"):
    st.dataframe(format_df)

# showing the df for INDIA here

st.markdown(
    "#### `INDIA - USD` Foreign Exchange Rate Data ⚖️"
)
india_df = pd.read_csv("India.csv")
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

main_df = pd.read_csv('my_file.csv')

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
        loaded_model = joblib.load('arima_model.pkl')

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
    model = load_model('LSTM_Model_Max.h5')

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
    selected_model = st.selectbox('Select from Available Models 🤖', options=('ARIMA', 'LSTM', 'FB Prophet'))
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
        st.image("A-pfp.png")
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
        st.image("H-pfp.png")
        st.subheader("2️⃣ Hardik Sharma")
        st.markdown('''
            * **`Github`** ⭐ 
                https://github.com/CodeStrate
            * **`Linkedin`**  🔗
                https://linkedin.com/in/hardik-sharma-0256cs
        ''')

    with col3:
        st.image("D-pfp.png")
        st.subheader("3️⃣ Divyanshi")
        st.markdown('''
            * **`Github`** ⭐
                https://github.com/Divvyanshiii
            * **`Linkedin`**  🔗 
                https://linkedin.com/in/divyanshi-shrivastav
        ''')

def fetch_exchange_rates(base_currency, target_currency, amount):
    api_key = 'b37bb6abd06165ca30037187'
    base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    try:
        # Send a GET request to the API
        response = requests.get(base_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract the conversion rates from the response
            conversion_rates = data["conversion_rates"]
            
            # Extract the exchange rate for the target currency
            exchange_rate = conversion_rates.get(target_currency)
            
            if exchange_rate is not None:
                # Print the exchange rate
                print(f"The exchange rate from {base_currency} to {target_currency} is: {exchange_rate}")
                return exchange_rate * amount
            else:
                st.error(f"Error: Unable to find exchange rate for {target_currency}")
                print(f"Error: Unable to find exchange rate for {target_currency}")
        else:
            # Print an error message if the request was not successful
            st.error(f"Error: Unable to fetch exchange rates. Status code: {response.status_code}")
            print(f"Error: Unable to fetch exchange rates. Status code: {response.status_code}")
    except Exception as e:
        # Handle any exceptions that occur during the request
        st.warning(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

with convert_currency:

    # Title
    st.title('Currency Converter 🔁')

    # Currency input fields
    amount = st.number_input('Enter amount to convert 👇', min_value=1, step=1, value=100)
    base_currency = st.text_input('Enter Source Currency 👇', value='USD')
    target_currency = st.text_input('Enter Target Currency 👇', value="INR")

    # Date input field
    conversion_date = st.date_input('Conversion Date 👇', value=date.today())
    # Convert currency when the user clicks the button
    if st.button('Convert'):
        try:
            exchange_rate = fetch_exchange_rates(base_currency=base_currency, target_currency=target_currency, amount=amount)
            st.success(f'{amount} {base_currency} is approximately {exchange_rate:.2f} {target_currency}')
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