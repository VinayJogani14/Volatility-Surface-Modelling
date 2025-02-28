# app.py
import streamlit as st
import pandas as pd
import yfinance as yf

# Import modules
from data_retrieval import get_options_data
from models import black_scholes_price, black_scholes_delta, heston_call_price
from calibration import calibrate_heston
from strategy import simulate_delta_hedge
from risk_analysis import compute_var_es, stress_test
from visualization import plot_vol_surface

# Streamlit App Title
st.title("Volatility Surface Modeling App")

# Sidebar inputs for ticker symbol and run controls
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (Yahoo Finance)", value="AAPL")
run_fetch = st.sidebar.button("Fetch Options Data")

if run_fetch:
    # Fetch options data using yFinance
    st.write(f"Fetching options data for **{ticker}** ...")
    options_df = get_options_data(ticker)
    if options_df.empty:
        st.error("Failed to retrieve options data. Please check the ticker symbol.")
    else:
        # Display summary of data
        underlying_price = yf.Ticker(ticker).history(period="1d")['Close'][0]
        st.write(f"**Underlying Price:** {underlying_price:.2f}")
        st.write(f"**Number of options loaded:** {len(options_df)} (calls and puts across all expiries)")
        st.dataframe(options_df.head(10))  # show a sample of the data
        
        # Volatility Surface Visualization
        st.subheader("Implied Volatility Surface")
        fig = plot_vol_surface(options_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Calibration (Heston)
        st.subheader("Heston Model Calibration")
        # Prepare market data for calibration: use one call per expiry (ATM strike for simplicity)
        market_data = []
        for expiry, grp in options_df[options_df['optionType']=='call'].groupby('expiry'):
            # pick strike closest to underlying price as a representative ATM option for that expiry
            df_grp = grp.copy()
            df_grp['moneyness'] = abs(df_grp['strike'] - underlying_price)
            atm_option = df_grp.sort_values('moneyness').iloc[0]
            market_data.append({
                "T": atm_option['T'],
                "strike": atm_option['strike'],
                "optionType": "call",
                "price": atm_option['lastPrice']
            })
        # Calibrate Heston parameters to this market data
        with st.spinner("Calibrating Heston model..."):
            heston_params = calibrate_heston(market_data, S0=underlying_price, r=0.01, q=0.0)
        v0, theta, kappa, sigma, rho = heston_params
        st.write("**Calibrated Heston Parameters:**")
        st.write(f"v0 (initial var): {v0:.4f}, θ (long-run var): {theta:.4f}, "
                 f"κ (mean reversion): {kappa:.4f}, σ (vol of vol): {sigma:.4f}, ρ (corr): {rho:.4f}")
        
        # Trading Strategy Simulation (Delta Hedging)
        st.subheader("Delta-Hedging Simulation")
        # User inputs to select an option to hedge
        expiries = sorted(options_df['expiry'].unique())
        chosen_expiry = st.selectbox("Select option expiry for hedging simulation", expiries)
        strikes = sorted(options_df[(options_df['expiry']==chosen_expiry) & (options_df['optionType']=='call')]['strike'].unique())
        atm_idx = int(len(strikes)/2)
        chosen_strike = st.selectbox("Select strike", strikes, index=atm_idx)
        # Get implied volatility for the chosen option (use mid of bid-ask if available, otherwise impliedVol provided)
        opt_row = options_df[(options_df['expiry']==chosen_expiry) & 
                             (options_df['strike']==chosen_strike) & 
                             (options_df['optionType']=='call')].iloc[0]
        implied_vol = opt_row['impliedVolatility']
        T_mat = opt_row['T']
        st.write(f"Chosen option: Expiry={chosen_expiry}, Strike={chosen_strike}, Implied Vol={implied_vol:.2%}, T={T_mat:.2f} years")
        # Run delta hedging simulation
        with st.spinner("Simulating delta-hedged strategy..."):
            pnl_distribution = simulate_delta_hedge(underlying_price, chosen_strike, T_mat, r=0.01, q=0.0, 
                                                    sigma=implied_vol, steps=252, n_paths=1000)
        # Compute risk metrics from simulation
        VaR95, ES95 = compute_var_es(pnl_distribution, confidence=0.95)
        st.write(f"**95% VaR:** {VaR95:.2f} (loss)  &nbsp;&nbsp; **95% Expected Shortfall:** {ES95:.2f} (loss)")
        # (We interpret VaR/ES as positive numbers representing potential loss)
        
        # Stress Testing
        st.subheader("Stress Testing Scenarios")
        # Compute current delta for the chosen option
        current_delta = black_scholes_delta(underlying_price, chosen_strike, T_mat, r=0.01, q=0.0, sigma=implied_vol, option_type='call')
        stress_results = stress_test(underlying_price, chosen_strike, T_mat, r=0.01, q=0.0, sigma=implied_vol, delta=current_delta)
        st.write(f"Current Delta of call (hedge ratio) = {current_delta:.3f}")
        for label, pnl in stress_results:
            if pnl < 0:
                result_str = f"Loss of {-pnl:.2f}"
            else:
                result_str = f"Profit of {pnl:.2f}"
            st.write(f"{label}: {result_str}")
