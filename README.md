Stock Price Prediction using LSTM with Python

Overview: This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks to predict stock prices. The LSTM model is trained on historical stock price data and then used to predict future prices. The project utilizes Python libraries such as yfinance, matplotlib, mplfinance, numpy, pandas, scikit-learn, and keras.

Technologies and Tactics Used:

Python: Python is the primary programming language used for this project due to its simplicity, versatility, and rich ecosystem of libraries for data analysis and machine learning.

yfinance: The yfinance library is used to fetch historical stock price data from Yahoo Finance. It provides an easy-to-use interface for accessing financial data.

Matplotlib: Matplotlib is a plotting library in Python used to visualize the historical and predicted stock prices.

mplfinance: mplfinance is used to plot the historical stock prices as candlestick charts, providing a more detailed view of Open, High, Low, and Close (OHLC) data.

NumPy: NumPy is used for numerical computations, particularly for array manipulation and mathematical operations on the data.

Pandas: Pandas is used for data manipulation and analysis, especially for handling the fetched historical stock price data and preparing it for training the LSTM model.

scikit-learn: The MinMaxScaler from scikit-learn is used to scale the stock price data to a range between 0 and 1, which is beneficial for training neural networks.

Keras with TensorFlow backend: Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. It's used to build and train the LSTM model for stock price prediction.

Long Short-Term Memory (LSTM): LSTM is a type of recurrent neural network (RNN) architecture that is well-suited for sequence prediction tasks. It can learn long-term dependencies and is widely used in time series forecasting, including stock price prediction.

Project Workflow:

Data Fetching: Historical stock price data is fetched using the yfinance library by providing the company's ticker symbol and the desired date range.

Data Visualization: Matplotlib is used to plot the closing prices of the stock over time. Additionally, mplfinance is used to visualize the OHLC data as candlestick charts.

Data Preparation: The fetched stock price data is preprocessed and scaled using the MinMaxScaler from scikit-learn to prepare it for training the LSTM model. Sequences of data are created for training the model.

LSTM Model Building: A Sequential model is constructed using Keras. It consists of two LSTM layers followed by a Dense layer. The model is compiled with the Adam optimizer and trained on the prepared data.

Testing and Prediction: Test data is prepared from the last portion of the historical data, and predictions are made using the trained LSTM model. Predictions are then inverse-scaled to obtain the actual stock price values.

Results Visualization and Evaluation: Matplotlib is used to visualize the actual and predicted stock prices. The Mean Squared Error (MSE) between the actual and predicted prices is calculated and printed for evaluating the performance of the model.

Conclusion: This project demonstrates the use of LSTM neural networks for stock price prediction, a common application of machine learning in finance. By following the provided README, users can understand the project's workflow, technologies used, and how to run the code to predict stock prices for a given company.
