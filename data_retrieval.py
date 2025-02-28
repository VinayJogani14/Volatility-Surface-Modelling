# data_retrieval.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def get_options_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetch options chain data for the given ticker from Yahoo Finance.
    Returns a DataFrame with columns: ['expiry', 'optionType', 'strike', 
    'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest'].
    Both call and put options for all available expirations are included.
    """
    # Initialize yfinance Ticker object
    ticker = yf.Ticker(ticker_symbol)
    
    # Get all option expiration dates for the ticker
    expirations = ticker.options
    
    all_options = []  # list to collect DataFrames for each expiration
    for exp in expirations:
        try:
            # Fetch option chain for this expiration
            opt_chain = ticker.option_chain(exp)
        except Exception as e:
            # Skip expiration if there's an issue fetching data
            print(f"Warning: could not fetch data for expiration {exp}: {e}")
            continue
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()
        # Tag option type and expiry date
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'
        calls['expiry'] = pd.to_datetime(exp)
        puts['expiry'] = pd.to_datetime(exp)
        # Append to list
        all_options.append(calls)
        all_options.append(puts)
    if not all_options:
        # Return empty DataFrame if no data fetched
        return pd.DataFrame()
    
    # Combine all expiration DataFrames
    data = pd.concat(all_options, ignore_index=True)
    
    # Calculate time to maturity in years for each option (for convenience in modeling)
    today = datetime.now().date()
    data['expiry'] = pd.to_datetime(data['expiry']).dt.date  # ensure date type
    data['T'] = data['expiry'].apply(lambda x: max((x - today).days, 0) / 365.0)
    
    # Sort data by expiry then strike
    data.sort_values(['expiry', 'strike', 'optionType'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# Example usage (for testing outside Streamlit):
# df = get_options_data("AAPL")
# print(df.head())
