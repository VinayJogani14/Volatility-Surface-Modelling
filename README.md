# 📉 Volatility Surface Modelling

A comprehensive Python-based framework for modeling implied volatility surfaces, calibrating financial models (e.g., Black-Scholes, Heston), engineering arbitrage strategies, and performing portfolio risk analytics using real-world market data.

---

## 🧩 Overview

This repository presents a modular and extensible platform designed for quantitative finance professionals and researchers interested in:

- Constructing and calibrating **volatility surfaces** from market data
- Designing and backtesting **options trading strategies** informed by implied volatility
- Performing **risk analysis** and **portfolio diagnostics** using statistical metrics
- Visualizing market structure through **interactive 2D and 3D charts**

The project leverages **real-time and historical financial data** and applies **advanced numerical methods** for optimization, integration, and simulation.

---

## 🔍 Key Capabilities

### ✅ Volatility Modelling
- Supports **Black-Scholes** and **Heston stochastic volatility** models
- Model calibration via:
  - Root-finding methods (e.g., Brent, Newton-Raphson)
  - Convex/non-convex optimization (SciPy, CVXPY)
- Constructs complete **volatility surfaces** and **smiles** using spline interpolation and curve fitting

### 📈 Options Strategy Backtesting
- Long/short volatility and delta-neutral strategies
- Use of surface-informed **arbitrage signals**
- Tracks key metrics: Sharpe Ratio, drawdown, and P&L attribution

### ⚠️ Risk Analysis
- **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**
- Risk contribution and sensitivity reports
- Stress testing with user-defined shocks and market scenarios

### 📊 Visualization
- Interactive 3D plotting of volatility surfaces (Plotly)
- Comparative plots of implied vs. historical volatility
- Time-series analysis of model parameters and risk metrics

---

## 🗂️ Project Structure

```
Volatility-Surface-Modelling/
│
├── app.py                  # Launchable Streamlit app for dashboard interaction
├── models.py               # Model definitions (Black-Scholes, Heston)
├── calibration.py          # Surface calibration and optimization routines
├── strategy.py             # Options trading strategies and evaluation
├── risk_analysis.py        # VaR, Expected Shortfall, stress testing
├── data_retrieval.py       # Market data via yFinance / Bloomberg
├── visualization.py        # Surface and smile plotting utilities
├── README.md               # Documentation
└── requirements.txt        # All dependencies (auto-generated)
```

---

## 💾 Installation

```bash
git clone https://github.com/your-username/Volatility-Surface-Modelling.git
cd Volatility-Surface-Modelling
pip install -r requirements.txt
```

---

## 🚀 Quickstart

Run the Streamlit application (if interactive GUI is included):

```bash
streamlit run app.py
```

Run individual scripts for CLI-based usage:

```python
# Calibrate surface using historical options data
from calibration import calibrate_vol_surface

# Generate trading signals
from strategy import generate_signals

# Analyze risk
from risk_analysis import compute_var
```

---

## 🛠️ Dependencies

Auto-detected core dependencies:

```text
numpy
pandas
scipy
cvxpy
plotly
yfinance
streamlit
```

> Note: `cvxpy` may require additional solvers like `osqp`, `ecos`, or `scs`.

---

## 📊 Example Output

| Chart                         | Description                               |
|------------------------------|-------------------------------------------|
| Volatility Smile             | Cross-sectional IV across strike prices   |
| 3D Volatility Surface        | IV across strike × expiry space           |
| VaR vs Portfolio Returns     | Risk-adjusted backtest diagnostics        |
| Strategy P&L Curve           | Backtested profit & loss performance      |

---

## 🔍 Use Cases

- **Quant Research** – Model fitting, arbitrage detection
- **Trading Strategy Development** – Design and evaluate volatility-based strategies
- **Risk Management** – Portfolio risk quantification and stress testing
- **Education** – Learn implied volatility modeling and financial engineering

---

## 📄 License

This project is licensed under the MIT License.
