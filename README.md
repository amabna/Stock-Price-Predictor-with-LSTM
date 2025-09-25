# Stock Price Predictor with LSTM and MACD Analysis

## Project Overview
This project is a **Stock Price Predictor** that leverages deep learning (LSTM neural networks) to forecast stock prices and technical analysis (MACD indicator) to generate buy/sell signals. Built with Python, it includes an interactive **Streamlit web application** for user-friendly stock price predictions and visualizations. The project supports multiple stocks, including international tickers (e.g., AMZN, TSLA) and Iranian market tickers (e.g., FOOLAD, KHODRO).

The codebase consists of two main components:
1. **Stock Price Prediction**: Uses an LSTM model to predict the next day's stock price based on historical closing prices.
2. **Technical Analysis**: Implements the MACD (Moving Average Convergence Divergence) indicator to identify potential buy and sell signals.

This project is ideal for data scientists, financial analysts, or enthusiasts interested in stock market forecasting and technical analysis.

---

## Features
- **LSTM-Based Prediction**: Uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on the past 60 days of closing prices.
- **MACD Indicator**: Generates buy/sell signals using the MACD indicator, visualized with interactive candlestick charts.
- **Streamlit Web App**: Provides an interactive interface to select stocks, view predictions, and visualize results.
- **Supported Stocks**: Includes datasets for AMZN, TSLA, FOOLAD, and KHODRO, with easy extensibility for other tickers.
- **Performance Metrics**: Displays Root Mean Squared Error (RMSE) to evaluate prediction accuracy.
- **Visualizations**: Plots historical data, predictions, and buy/sell signals using Matplotlib and Plotly.

---

## Installation
To run this project locally, follow these steps:

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (optional for cloning)

### Dependencies
Install the required Python libraries using the following command:

```bash
pip install numpy pandas matplotlib sklearn tensorflow streamlit pillow plotly
```

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Stock-Price-Predictor-LSTM.git
   cd Stock-Price-Predictor-LSTM
   ```

2. **Prepare Datasets**:
   - Place the stock data CSV files (`amazon.csv`, `tesla.csv`, `foolad.csv`, `khodro.csv`) in the appropriate directory (e.g., `stock_market_analysis/`).
   - Ensure the CSV files contain at least a `Date` column and a `Close` column for price data.

3. **Run the Streamlit App**:
   ```bash
   streamlit run webapppredictor.py
   ```

4. **Access the Web App**:
   - Open your browser and navigate to `http://localhost:8501`.
   - Select a stock symbol (AMZN, TSLA, FOOLAD, or KHODRO) from the sidebar to view predictions and visualizations.

---

## Project Structure
- **`webapppredictor.py`**: Main script for the Streamlit web application. Handles data loading, LSTM model training, prediction, and visualization of stock prices and predictions.
- **`predictor app.ipynb`**: Jupyter notebook containing the core LSTM prediction logic, used for development and testing.
- **`analysis.ipynb`**: Jupyter notebook for technical analysis, implementing the MACD indicator and generating buy/sell signals with visualizations.
- **Data Files**:
  - `amazon.csv`: Historical stock data for Amazon.
  - `tesla.csv`: Historical stock data for Tesla.
  - `foolad.csv`: Historical stock data for Foolad (Iranian stock).
  - `khodro.csv`: Historical stock data for Khodro (Iranian stock).

---

## How It Works
1. **Data Preparation**:
   - Loads stock data from CSV files and sets the `Date` column as the index.
   - Filters for the `Close` price column and normalizes data using `MinMaxScaler`.

2. **LSTM Model**:
   - Uses 80% of the data for training and 20% for testing.
   - Creates sequences of 60 days of closing prices to predict the next day's price.
   - Trains a Sequential LSTM model with two LSTM layers (50 units each) and two Dense layers.
   - Compiles with the Adam optimizer and mean squared error loss.

3. **Prediction**:
   - Predicts the next day's stock price using the last 60 days of data.
   - Inverse-transforms predictions to original price scale.
   - Calculates RMSE to evaluate model accuracy.

4. **MACD Analysis**:
   - Computes the MACD and signal line using exponential moving averages.
   - Generates buy signals when MACD crosses above the signal line and sell signals when it crosses below.
   - Visualizes signals with candlestick charts using Plotly.

5. **Visualization**:
   - Plots training data, actual prices, and predicted prices using Matplotlib.
   - Displays buy/sell signals on a candlestick chart in the `analysis.ipynb` notebook.

---

## Usage
1. **Run the Web App**:
   - Execute `streamlit run webapppredictor.py`.
   - Select a stock symbol from the sidebar (e.g., AMZN, TSLA, FOOLAD, KHODRO).
   - View the predicted price for the next day and the RMSE of the model.
   - Check the plot showing training data, actual prices, and predictions.

2. **Explore Technical Analysis**:
   - Open `analysis.ipynb` in Jupyter Notebook.
   - Run the cells to generate MACD-based buy/sell signals and visualize them on a candlestick chart.

3. **Extend the Project**:
   - Add new stock datasets by including additional CSV files and updating the `get_data` function in `webapppredictor.py`.
   - Experiment with different LSTM architectures or hyperparameters (e.g., number of units, epochs) in `predictor app.ipynb`.
   - Enhance the MACD analysis by adding other technical indicators (e.g., RSI, Bollinger Bands) in `analysis.ipynb`.

---

## Sample Output
- **Web App Interface**:
  - Select a stock (e.g., TSLA) from the sidebar.
  - See the predicted price for the next day and RMSE.
  - View a plot comparing training data, actual prices, and predictions.

- **MACD Analysis**:
  - Green triangles indicate buy signals, and red triangles indicate sell signals on the candlestick chart.
  - The chart shows the closing price trend alongside MACD-based signals.

---

## Limitations
- **Data Dependency**: The model relies on historical data quality and availability. Missing or noisy data may affect predictions.
- **Market Volatility**: LSTM models may struggle with sudden market changes or black-swan events.
- **Local File Paths**: The `get_data` function in `webapppredictor.py` uses hardcoded file paths, which need to be updated for your local environment.
- **Overfitting Risk**: The LSTM model may overfit if not tuned properly (e.g., insufficient regularization or excessive epochs).
- **MACD Lag**: The MACD indicator is a lagging indicator and may miss rapid price movements.

---

## Future Improvements
- **Dynamic Data Loading**: Replace hardcoded file paths with a dynamic data source (e.g., Yahoo Finance API).
- **Additional Indicators**: Incorporate more technical indicators like RSI, Bollinger Bands, or Stochastic Oscillator.
- **Model Tuning**: Experiment with advanced LSTM architectures (e.g., Bidirectional LSTM) or other models like GRU or Transformer.
- **Real-Time Data**: Integrate live stock data for real-time predictions.
- **Deployment**: Deploy the Streamlit app to a cloud platform (e.g., Heroku, Streamlit Cloud) for public access.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows PEP 8 guidelines and includes relevant tests.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Keras**: For providing the LSTM model implementation.
- **Streamlit**: For enabling an interactive web interface.
- **Plotly and Matplotlib**: For visualization capabilities.
- **Toplearn.com**: Inspiration for the project structure.
