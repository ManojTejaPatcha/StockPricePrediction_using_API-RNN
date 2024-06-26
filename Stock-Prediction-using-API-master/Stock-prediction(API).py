
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf

# Provide Inputs
api_key = input("Please enter your API key: ")
company = input("Please enter the company's ticker symbol: ")

data = yf.download(company,'2020-01-01','2022-12-31')

data = data.drop('Adj Close', axis=1)

data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Plot closing prices
data['Close'].plot()
plt.title(f'{company} Stock Price')
plt.show()

# Plot OHLC data as a candlestick chart
mpf.plot(data,type='candle',mav=(3,6,9),volume=True)