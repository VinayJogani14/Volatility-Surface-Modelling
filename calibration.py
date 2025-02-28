# calibration.py
import numpy as np
from scipy.optimize import differential_evolution
from models import heston_call_price

# We assume models.py functions are imported or accessible: black_scholes_price, heston_call_price

def calibrate_heston(market_data, S0: float, r: float = 0.01, q: float = 0.0):
    """
    Calibrate Heston model parameters (v0, theta, kappa, sigma, rho) to market data.
    market_data: pandas DataFrame or list of dicts with fields 'T', 'strike', 'optionType', 'price'.
                 (Use call options data for calibration to call prices.)
    S0: current underlying price
    r: risk-free rate
    q: dividend yield
    Returns: Optimized parameter tuple [v0, theta, kappa, sigma, rho]
    """
    # Filter to use only call options for calibration (put prices can be converted via put-call parity if needed)
    # Assume market_data entries have 'optionType' which can be 'call' or 'put'
    data_calls = [row for row in market_data if row.get('optionType','call') == 'call']
    if not data_calls:
        raise ValueError("No call data provided for calibration.")
    
    # Objective function: sum of squared price errors between Heston model and market
    def objective(params):
        v0, theta, kappa, sigma, rho = params
        # Impose some penalty if parameters are out of reasonable bounds (though bounds in optimizer should handle it)
        if v0 < 0 or theta < 0 or sigma < 0 or kappa < 0 or not (-1 < rho < 1):
            return np.inf
        error_sum = 0.0
        for opt in data_calls:
            K = opt['strike']
            T = opt['T']
            market_price = opt['price']  # market option price
            # Compute Heston model call price for given parameters
            model_price = heston_call_price(S0, K, T, r, q, v0, theta, kappa, sigma, rho)
            # Weighting: can weight by 1 or by option price, but here equal weight for simplicity
            error_sum += (model_price - market_price) ** 2
        return error_sum
    
    # Initial guess for parameters (v0, theta, kappa, sigma, rho)
    # v0 and theta initalized to roughly ATM implied vol^2, kappa ~1, sigma ~ initial vol, rho ~ -0.5 (typical for equity)
    initial_guess = [0.04, 0.04, 1.0, 0.5, -0.5]
    # Parameter bounds to ensure feasible values
    bounds = [
        (1e-6, 2.0),    # v0: initial variance
        (1e-6, 2.0),    # theta: long-run variance
        (1e-3, 10.0),   # kappa: mean reversion speed
        (1e-3, 2.0),    # sigma: vol of vol
        (-0.999, 0.999) # rho: correlation
    ]
    # Use differential evolution for global optimization (can be time-consuming but robust)
    result = differential_evolution(objective, bounds, tol=1e-6, maxiter=100, polish=True)
    best_params = result.x
    print(f"Calibration completed. Optimal parameters: v0={best_params[0]:.6f}, "
          f"theta={best_params[1]:.6f}, kappa={best_params[2]:.6f}, "
          f"sigma={best_params[3]:.6f}, rho={best_params[4]:.6f}")
    return best_params

# Note: One could also use CVXPY for certain convex optimization formulations (e.g., smoothing the vol surface),
# but here we use SciPy's DE for the non-convex Heston calibration.
#
# Example usage:
# market_data = [
#     {"T": 0.5, "strike": 100, "optionType": "call", "price": 10.5},
#     {"T": 1.0, "strike": 100, "optionType": "call", "price": 15.2},
#     # ... more points across strikes and maturities
# ]
# S0 = 100
# params = calibrate_heston(market_data, S0)
