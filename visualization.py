# visualization.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

def plot_vol_surface(options_df: pd.DataFrame):
    """
    Generate a Plotly 3D surface plot for the implied volatility surface.
    options_df: DataFrame containing option data with columns 'T' (time to maturity in years), 
                'strike', and 'impliedVolatility'. Ideally include a single option type (calls).
    Returns: Plotly Figure object with the volatility surface.
    """
    if options_df.empty:
        raise ValueError("Options DataFrame is empty, cannot plot surface.")
    # Use only call data for surface (put and call vol should be similar via put-call parity)
    df = options_df[options_df['optionType'] == 'call'].copy()
    # Prepare data for interpolation
    T_vals = df['T'].values
    K_vals = df['strike'].values
    vol_vals = df['impliedVolatility'].values
    # Define grid ranges for T and K
    T_lin = np.linspace(T_vals.min(), T_vals.max(), num=50)
    K_lin = np.linspace(K_vals.min(), K_vals.max(), num=50)
    TT, KK = np.meshgrid(T_lin, K_lin)
    # Interpolate implied vol data onto grid (linear interpolation; use nearest for extrapolation)
    VV = griddata((T_vals, K_vals), vol_vals, (TT, KK), method='linear')
    # Fill any NaNs from interpolation by nearest neighbor (for extrapolation at edges)
    if np.any(np.isnan(VV)):
        VV_nearest = griddata((T_vals, K_vals), vol_vals, (TT, KK), method='nearest')
        VV = np.where(np.isnan(VV), VV_nearest, VV)
    # Create surface plot
    fig = go.Figure(data=[go.Surface(x=TT, y=KK, z=VV, colorscale='Viridis', opacity=0.8)])
    fig.update_layout(title="Implied Volatility Surface",
                      scene=dict(xaxis_title="Time to Maturity (years)",
                                 yaxis_title="Strike Price",
                                 zaxis_title="Implied Volatility"))
    return fig

# Example usage:
# fig = plot_vol_surface(options_df)
# fig.show()  # in a Jupyter environment, or use st.plotly_chart(fig) in Streamlit
