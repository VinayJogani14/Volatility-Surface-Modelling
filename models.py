# models.py
import numpy as np
from math import log, sqrt, exp, pi
from scipy.stats import norm
from scipy.integrate import quad

def black_scholes_price(S: float, K: float, T: float, r: float, q: float, 
                        sigma: float, option_type: str = 'call') -> float:
    """
    Calculate the Black-Scholes price for a European call or put option.
    S: current underlying price
    K: strike price
    T: time to maturity in years
    r: risk-free interest rate (annualized)
    q: dividend yield (annualized)
    sigma: volatility of the underlying (annualized standard deviation)
    option_type: 'call' or 'put'
    """
    if T <= 0:
        # At expiration, option price is its intrinsic value
        if option_type == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    # Black-Scholes formula components
    d1 = (log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    # Discount factors for dividend and risk-free rate
    df_div = exp(-q * T)  # dividend discount factor
    df_rf = exp(-r * T)   # risk-free discount factor
    if option_type == 'call':
        price = S * df_div * norm.cdf(d1) - K * df_rf * norm.cdf(d2)
    else:  # put option
        price = K * df_rf * norm.cdf(-d2) - S * df_div * norm.cdf(-d1)
    return price

def black_scholes_delta(S: float, K: float, T: float, r: float, q: float, 
                        sigma: float, option_type: str = 'call') -> float:
    """
    Compute the Black-Scholes delta (hedge ratio) for a European call or put option.
    Delta is the first derivative of option price w.r.t. underlying price.
    """
    if T <= 0:
        # At expiry, delta for call is 1 if S>K (0 otherwise); for put, -1 if S<K (0 otherwise).
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    # Call delta = e^{-qT} * N(d1), Put delta = -e^{-qT} * N(-d1)
    if option_type == 'call':
        return exp(-q * T) * norm.cdf(d1)
    else:
        return -exp(-q * T) * norm.cdf(-d1)

# Heston Model Implementation
def _heston_characteristic_fn(u: complex, T: float, v0: float, theta: float, 
                              kappa: float, sigma: float, rho: float) -> complex:
    """
    Characteristic function for log-price in Heston model.
    u: complex argument for characteristic function
    T: time to maturity
    v0: initial variance (volatility^2 at t=0)
    theta: long-run variance (mean of variance under risk-neutral measure)
    kappa: rate of mean reversion of variance
    sigma: volatility of volatility
    rho: correlation between Brownian motions for asset and variance
    Returns phi(u), without the factor exp(i u ln(S0 e^{(r-q)T})) which is handled externally.
    """
    # Define commonly used terms
    b = kappa + 1j * rho * sigma * u
    d = np.sqrt(b**2 + (sigma**2) * (u * (u - 1j)))  # note: u*(u-1j) = u^2 - i u
    g = (b - d) / (b + d)
    # Avoid division by zero issues by adding a tiny epsilon if needed (not usually necessary for valid params)
    # Compute C(T) and D(T) for Heston characteristic exponent
    # C(T) corresponds to the constant term (dependent on theta) and D(T) the coefficient of v0.
    # Using closed-form solution for Heston (Heston 1993)
    G = np.exp(-d * T)
    # D(T) term:
    D = (1 - G) / (1 - g * G) * ((b - d) / (sigma**2))
    # C(T) term:
    C = (kappa * theta / (sigma**2)) * ((b - d) * T - 2 * np.log((1 - g * G) / (1 - g)))
    # Characteristic function (without the exp(i u ln S0) factor)
    return np.exp(C + D * v0)

def heston_call_price(S: float, K: float, T: float, r: float, q: float, 
                      v0: float, theta: float, kappa: float, sigma: float, rho: float) -> float:
    """
    Price a European call option using the Heston stochastic volatility model.
    S: current underlying price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    q: dividend yield
    v0: initial variance (current V(t) at t=0)
    theta: long-run variance (mean reversion level of variance)
    kappa: mean reversion speed of variance
    sigma: volatility of volatility (std dev of variance process)
    rho: correlation between asset returns and variance (in [-1,1])
    Returns the call option price under the Heston model.
    """
    # If T is very close to 0, use intrinsic value (no time for volatility to act)
    if T <= 1e-8:
        return max(S - K, 0.0)
    # Forward price F = S * exp((r - q)T)
    F = S * exp((r - q) * T)
    # x = ln(F/K)
    x = log(F / K)
    # Integration limits and function:
    # We'll integrate from 0 to an upper limit (e.g., 100) for numerical approximation
    integrand = lambda u: (2.0 * np.real(np.exp(-1j * u * x) * _heston_characteristic_fn(u + 0.5j, T, v0, theta, kappa, sigma, rho)) 
                           / (u**2 + 0.25))
    # Perform numerical integration (SciPy quad)
    # Note: quad returns (result, error)
    integration_limit = 100.0
    result, error = quad(integrand, 0.0, integration_limit, limit=100, epsabs=1e-8, epsrel=1e-8)
    # Heston call price formula using the integration result
    call_price = S * exp(-q * T) - K * exp(-r * T) * result
    return call_price

# Example usage (for testing):
# price_bs = black_scholes_price(100, 100, 0.5, 0.01, 0.0, 0.2, 'call')
# delta_bs = black_scholes_delta(100, 100, 0.5, 0.01, 0.0, 0.2, 'call')
# price_heston = heston_call_price(100, 100, 0.5, 0.01, 0.0, v0=0.04, theta=0.04, kappa=1.0, sigma=0.5, rho=-0.5)
