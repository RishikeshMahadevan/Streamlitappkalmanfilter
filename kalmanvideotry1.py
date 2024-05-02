import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data['^DJI'], data['^GSPC'] = data['^GSPC'], data['^DJI']
    data.columns = [0, 1]
    return data

def process_data(EquityPrices):
    HedgeRatio = -1 * (EquityPrices.iloc[0, 0] / EquityPrices.iloc[0, 1])
    EquityPrices.iloc[:, 1] = EquityPrices.iloc[:, 1] * HedgeRatio
    prices = EquityPrices.values

    return prices

def kalman_filter(meas, x_prev, p_prev, F, H, Q, R):
    """
    Kalman filter implementation
    
    Args:
        meas (float): Current measurement
        x_prev (float): Previous state estimate
        p_prev (float): Previous state covariance
        F (float): State transition model
        H (float): Observation model
        Q (float): Process noise covariance
        R (float): Measurement noise covariance
        
    Returns:
        x_curr (float): Current state estimate
        p_curr (float): Current state covariance
    """
    # Predict
    x_pred = F * x_prev
    p_pred = F * p_prev * F + Q
    
    # Update
    K = p_pred * H / (H * p_pred * H + R)
    x_curr = x_pred + K * (meas - H * x_pred)
    p_curr = (1 - K * H) * p_pred
    
    return x_curr, p_curr

def calculate_portfolio(prices, EquityPrices):
    # Initialize parameters
    F = 1
    H = 1
    Q = 1
    R = 2

    # Initialize state
    NVec = 2  # Need two prices
    x_prev = prices[0, :]
    p_prev = np.full(NVec, Q)

    # Initialize arrays
    NTSteps = len(prices)
    KPrice = np.zeros((NTSteps, NVec))
    KCovPrice = np.zeros((NTSteps, NVec))

    # Kalman filter loop for both S&P 500 and Hedged DJI
    KPrice[0, :] = x_prev
    KCovPrice[0, :] = p_prev

    initial_port = 1000000
    tradevalue = 350000
    cashvalue = initial_port
    cash_shift = 250

    portvalue = initial_port
    port = []
    signal = []
    diffvalues = []
    diffvalues.append(0)
    nstocks = 0
    nstockslist = []
    spread = 0
    portk = initial_port

    obs_port = []
    kalman_port = []
    buysignal = []
    sellsignal = []

    cash = []
    signal = 0
    #cash.append(cashvalue)
    diffvalues.append(0)
    #port.append(initial_port)
    #nstockslist.append(nstocks)

    for k in range(1, NTSteps):
        meas = prices[k, :]
        for i in range(NVec):
            x_curr, p_curr = kalman_filter(meas[i], x_prev[i], p_prev[i], F, H, Q, R)
            KPrice[k, i] = x_curr
            KCovPrice[k, i] = p_curr
        
        x_prev = KPrice[k, :]
        p_prev = KCovPrice[k, :]
        
        obs_portfolio = (EquityPrices.iloc[k])[0] + (EquityPrices.iloc[k])[1] + cash_shift
        kalman_spread = (x_prev[0] + x_prev[1]) + cash_shift
        
        obs_port.append(obs_portfolio)
        kalman_port.append(kalman_spread)
        
        diff = obs_portfolio - kalman_spread
        diffvalues.append(diff)
        
        if signal == 1:
            if tradevalue < cashvalue:
                nstocks += tradevalue / (obs_portfolio)
                cashvalue -= tradevalue
        
        if signal == -1:
            cashvalue += nstocks * obs_portfolio
            nstocks = 0
        
        portk = cashvalue + nstocks * (obs_portfolio)
        cash.append(cashvalue)
        nstockslist.append(nstocks)
        
        signal = 0
        
        if diff < 0:
            if tradevalue < cashvalue:
                signal = 1
                buysignal.append(k)
        else:
            if nstocks > 0:
                signal = -1
                sellsignal.append(k)
        
        port.append(portk)

    return obs_port, kalman_port, port, cash, nstockslist, buysignal, sellsignal

def plot_graphs(idx, obs_port, kalman_port, port, cash, nstockslist, buysignal, sellsignal):
    # Clear previous plots
    plt.clf()
    
    # Set figure size
    plt.figure(figsize=(24, 24))
    
    # Plot actual portfolio price and Kalman-filtered portfolio price
    plt.subplot(4, 1, 1)
    plt.plot(obs_port[:idx], label='Actual Portfolio Price')
    plt.plot(kalman_port[:idx], 'r-', label='Kalman Portfolio')
    
    # Plot buy signals as green arrows
    if idx in buysignal:
        plt.arrow(idx, obs_port[idx-1], 0, 20, color='g', head_width=2, head_length=10)

    # Plot sell signals as red arrows
    if idx in sellsignal:
        plt.arrow(idx, obs_port[idx-1], 0, -20, color='r', head_width=2, head_length=10)

    plt.title('Actual vs Kalman Portfolio with Multiple Signals')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Price')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(port[:idx], color='purple', label='Portfolio Values', linestyle='-')
    plt.title('Portfolio values')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Values')
    plt.legend()

    # Plot cash value
    plt.subplot(4, 1, 3)
    plt.plot(cash[:idx], color='blue', label='Cash Value', linestyle='-')
    plt.title('Cash Value')
    plt.xlabel('Days')
    plt.ylabel('Cash Value')
    plt.legend()

    # Plot number of stocks
    plt.subplot(4, 1, 4)
    plt.plot(nstockslist[:idx], color='blue', label='Number of Stocks', linestyle='-')
    plt.title('Number of Stocks')
    plt.xlabel('Days')
    plt.ylabel('Number of Stocks')
    plt.legend()

    # Show the plot
    st.pyplot()


def main():
    st.title('Hedged Portfolio trading with Kalman Filter')
    
    # Get user input for tickers, start date, and end date
    tickers_input = st.text_input('Enter tickers (comma-separated)', '^GSPC,^DJI')
    start_date = st.date_input('Enter start date', key='start_date_input', value=pd.to_datetime('2023-04-16'))
    end_date = st.date_input('Enter end date', key='end_date_input', value=pd.to_datetime('2024-04-15'))
    
    submit_button = st.button('Submit')
    reset_button = st.button('Reset')
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if reset_button:
        st.session_state.data_loaded = False
        st.session_state.idx = 1
    
    if submit_button:
        tickers = tickers_input.split(',')
        
        # Get data
        EquityPrices = get_data(tickers, start_date, end_date)
        prices = process_data(EquityPrices)
        
        # Calculate portfolio
        obs_port, kalman_port, port, cash, nstockslist, buysignal, sellsignal = calculate_portfolio(prices, EquityPrices)
        
        st.session_state.obs_port = obs_port
        st.session_state.kalman_port = kalman_port
        st.session_state.port = port
        st.session_state.cash = cash
        st.session_state.nstockslist = nstockslist
        st.session_state.buysignal = buysignal
        st.session_state.sellsignal = sellsignal
        
        st.session_state.data_loaded = True
    
    if st.session_state.data_loaded:
        # Disable tickers, start date, and end date inputs
  
        # Add a slider to control the playback
        if 'idx' not in st.session_state:
            st.session_state.idx = 1
        idx = st.slider('Select Index', min_value=1, max_value=len(st.session_state.obs_port), value=st.session_state.idx)
        
        # Add a "Next" button to increment the index by one
        if st.button('Next'):
            st.session_state.idx += 1
            idx = min(st.session_state.idx, len(st.session_state.obs_port) - 1)
        
        # Update the slider value to reflect the new index
        st.session_state.idx = idx
        
        # Plot the graphs
        plot_graphs(idx, st.session_state.obs_port, st.session_state.kalman_port, st.session_state.port, st.session_state.cash, st.session_state.nstockslist, st.session_state.buysignal, st.session_state.sellsignal)

if __name__ == "__main__":
    main()


