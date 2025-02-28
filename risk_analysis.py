# risk_analysis.py
import numpy as np
# Assume black_scholes_price and black_scholes_delta are imported for stress testing
from models import black_scholes_price, black_scholes_delta


def compute_var_es(pnl_distribution: np.ndarray, confidence: float = 0.95):
    """
    Compute Value-at-Risk (VaR) and Expected Shortfall (ES) at a given confidence level.
    pnl_distribution: array of profit/loss outcomes (positive = profit, negative = loss).
    confidence: confidence level for VaR/ES (e.g., 0.95 for 95% VaR).
    Returns: (VaR, ES) as positive numbers representing loss (so VaR = e.g. 5 means $5 loss at 95%).
    """
    if len(pnl_distribution) == 0:
        return None, None
    # Sort P&L outcomes in ascending order (worst to best)
    sorted_pnl = np.sort(pnl_distribution)
    N = len(sorted_pnl)
    # Index for VaR cutoff (for 95% confidence, we look at the 5th percentile loss)
    cutoff_index = int(np.floor((1 - confidence) * N))
    if cutoff_index < 0: 
        cutoff_index = 0
    # VaR is defined as the loss at the cutoff_index (positive number for loss)
    VaR = -sorted_pnl[cutoff_index]  # negate because if pnl is negative (loss), we want positive value
    # Expected Shortfall (ES) is the average loss in the worst (1-confidence)% cases
    ES = -np.mean(sorted_pnl[:cutoff_index+1])
    return VaR, ES

def stress_test(S: float, K: float, T: float, r: float, q: float, sigma: float, delta: float):
    """
    Perform stress tests on a delta-hedged call position for specified shocks.
    S: current underlying price
    K: strike price
    T: time to maturity
    r: risk-free rate
    q: dividend yield
    sigma: current implied volatility
    delta: current delta of the option (hedge ratio, number of shares short per option)
    Returns: list of tuples (scenario_description, P&L) for each stress scenario.
    """
    scenarios = [
        {"dS": -0.20, "dVol": 0.00, "label": "Underlying -20%"},
        {"dS": 0.20, "dVol": 0.00, "label": "Underlying +20%"},
        {"dS": -0.20, "dVol": 0.25, "label": "Underlying -20%, Vol +25%"},
        {"dS": 0.00, "dVol": 0.50, "label": "Volatility +50%"},
    ]
    results = []
    # Current option price for reference
    current_option_price = black_scholes_price(S, K, T, r, q, sigma, option_type='call')
    current_portfolio_val = current_option_price - delta * S  # value = call - delta * stock (since we are short delta shares)
    for sc in scenarios:
        # Apply shocks
        S_new = S * (1 + sc["dS"])
        vol_new = sigma * (1 + sc["dVol"])
        # New option value after shock (assuming immediate repricing with new vol)
        option_new = black_scholes_price(S_new, K, T, r, q, vol_new, option_type='call')
        # Portfolio value after shock: option value - (same delta * new underlying price)
        portfolio_new = option_new - delta * S_new
        # P&L from scenario = new portfolio value - original portfolio value
        pnl = portfolio_new - current_portfolio_val
        results.append((sc["label"], pnl))
    return results

# Example usage:
# pnl_dist = np.array([ ... ])  # some distribution from simulation
# VaR, ES = compute_var_es(pnl_dist, confidence=0.95)
# print("95% VaR:", VaR, "95% ES:", ES)
# stress = stress_test(S=100, K=100, T=0.5, r=0.01, q=0.0, sigma=0.2, delta=0.5)
# for scenario, pnl in stress:
#     print(scenario, "P&L:", pnl)
