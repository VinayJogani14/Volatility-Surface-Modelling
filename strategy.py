# strategy.py
import numpy as np
# Assume black_scholes_price and black_scholes_delta are imported from models.py
from models import black_scholes_price, black_scholes_delta


def simulate_delta_hedge(S0: float, K: float, T: float, r: float, q: float, sigma: float, 
                         steps: int = 252, n_paths: int = 1000) -> np.ndarray:
    """
    Simulate a delta-hedged strategy for a long call option.
    Generates n_paths scenarios for the underlying price over time T (years),
    hedges the call by shorting underlying shares equal to the call's delta, rebalancing at each time step.
    Returns an array of P&L outcomes for the strategy at option expiration.
    Parameters:
      S0: initial underlying price
      K: option strike price
      T: time to maturity (years)
      r: risk-free interest rate
      q: dividend yield
      sigma: assumed volatility for underlying price evolution (could use implied vol or realized vol)
      steps: number of hedging intervals (e.g., 252 for daily hedging in 1 year)
      n_paths: number of Monte Carlo price paths to simulate
    """
    # Time increment
    dt = T / steps
    # Precompute drift and diffusion terms for risk-neutral GBM simulation
    # Using risk-neutral drift = (r - q) for pricing P&L perspective
    mu = r - q
    # Initialize array to store final P&L for each path
    pnl_results = np.zeros(n_paths)
    
    # Monte Carlo simulation of underlying price paths
    # We simulate in log space for numerical stability
    # Each path will be an array of length (steps+1) for prices from t=0 to t=T
    Z = np.random.standard_normal((n_paths, steps))  # random shocks
    # Use vectorized simulation for efficiency
    # log(S_t) = log(S_{t-1}) + (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    log_S_paths = np.zeros((n_paths, steps+1))
    log_S_paths[:, 0] = np.log(S0)
    for t in range(1, steps+1):
        log_S_paths[:, t] = (log_S_paths[:, t-1] 
                              + (mu - 0.5 * sigma**2) * dt 
                              + sigma * np.sqrt(dt) * Z[:, t-1])
    S_paths = np.exp(log_S_paths)  # convert log prices to actual prices
    
    # Iterate over each simulated price path to apply hedging strategy
    for i in range(n_paths):
        prices = S_paths[i]
        # Initial option price and delta at t=0
        option_price = black_scholes_price(S0, K, T, r, q, sigma, option_type='call')
        delta = black_scholes_delta(S0, K, T, r, q, sigma, option_type='call')
        # Start with a portfolio consisting of +1 call and -delta shares, and a bank account for financing
        # Calculate initial cash from shorting delta shares (this could be used to partially fund the call purchase)
        cash = -option_price + delta * S0  # negative because we pay for the call, positive from short stock
        # Simulate hedging over each time step
        for t in range(1, steps):
            # Time remaining after this step
            t_remaining = T - t * dt
            # Grow the cash at the risk-free rate for dt
            cash *= np.exp(r * dt)
            # Underlying price at this step
            S_t = prices[t]
            # Compute new option delta at this time (using Black-Scholes with constant sigma assumption)
            new_delta = black_scholes_delta(S_t, K, t_remaining, r, q, sigma, option_type='call')
            # Rebalance the hedge: adjust stock position to new delta
            # The change in the short stock position = new_delta - old delta
            # If new_delta > delta, we need to short additional (new_delta - delta) shares (cash inflow)
            # If new_delta < delta, we buy back (delta - new_delta) shares (cash outflow)
            cash += (delta - new_delta) * S_t  # remove (delta - new_delta) shares from short position at price S_t
            # Update delta to new value for next iteration
            delta = new_delta
        # At maturity (t = T), compute final P&L
        # Last underlying price:
        S_T = prices[steps]
        # Option payoff at maturity (we are long one call)
        payoff = max(S_T - K, 0.0)
        # Close the remaining short stock position (buy back delta shares at price S_T)
        cash *= np.exp(r * dt)  # accrue interest on cash for the last interval
        cash -= delta * S_T
        # After closing stock and receiving option payoff, the net cash is the P&L
        pnl = cash + payoff
        pnl_results[i] = pnl
    return pnl_results

# Example usage:
# pnl_dist = simulate_delta_hedge(100, 100, 1.0, 0.01, 0.0, 0.2, steps=252, n_paths=1000)
# print("Mean P&L:", np.mean(pnl_dist))
