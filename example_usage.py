# Volatility Trading Algorithm - Example Usage

"""
This notebook demonstrates the key features of the volatility trading algorithm
with practical examples and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from volatility_trading_algorithm import (
    BlackScholesModel, VolatilitySurface, RealizedVolatilityEstimator,
    VolatilityTradingStrategy, Option, Position
)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)


# ==============================================================================
# EXAMPLE 1: Option Pricing and Greeks
# ==============================================================================

print("=" * 80)
print("EXAMPLE 1: Black-Scholes Pricing and Greeks Calculation")
print("=" * 80)

# Parameters
S = 100  # Spot price
K = 100  # Strike price
T = 0.25  # Time to expiration (3 months)
r = 0.05  # Risk-free rate
sigma = 0.25  # Volatility

# Calculate prices
bs = BlackScholesModel()
call_price = bs.call_price(S, K, T, r, sigma)
put_price = bs.put_price(S, K, T, r, sigma)

print(f"\nOption Pricing:")
print(f"  Call Price: ${call_price:.2f}")
print(f"  Put Price: ${put_price:.2f}")

# Calculate Greeks
call_delta = bs.delta(S, K, T, r, sigma, 'call')
put_delta = bs.delta(S, K, T, r, sigma, 'put')
gamma = bs.gamma(S, K, T, r, sigma)
vega = bs.vega(S, K, T, r, sigma)
call_theta = bs.theta(S, K, T, r, sigma, 'call')
put_theta = bs.theta(S, K, T, r, sigma, 'put')

print(f"\nGreeks:")
print(f"  Call Delta: {call_delta:.4f}")
print(f"  Put Delta: {put_delta:.4f}")
print(f"  Gamma: {gamma:.4f}")
print(f"  Vega: ${vega:.2f} per 1% volatility change")
print(f"  Call Theta: ${call_theta:.2f} per day")
print(f"  Put Theta: ${put_theta:.2f} per day")


# ==============================================================================
# EXAMPLE 2: Implied Volatility Calculation
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Implied Volatility Back-Calculation")
print("=" * 80)

# Given a market option price, calculate implied volatility
market_call_price = 6.50
implied_vol = bs.implied_volatility(market_call_price, S, K, T, r, 'call')

print(f"\nMarket Call Price: ${market_call_price:.2f}")
print(f"Implied Volatility: {implied_vol:.2%}")

# Verify by repricing
repriced_call = bs.call_price(S, K, T, r, implied_vol)
print(f"Repriced Call: ${repriced_call:.2f}")
print(f"Pricing Error: ${abs(market_call_price - repriced_call):.4f}")


# ==============================================================================
# EXAMPLE 3: Building a Volatility Surface
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Volatility Surface Construction")
print("=" * 80)

# Create a realistic options chain with volatility smile
spot = 100
strikes = np.arange(85, 116, 5)
expiries = [0.083, 0.25, 0.5, 1.0]  # 1M, 3M, 6M, 1Y

options_list = []

for expiry in expiries:
    for strike in strikes:
        moneyness = strike / spot
        # Create volatility smile (higher vol for OTM options)
        base_vol = 0.20
        smile_effect = 0.15 * (moneyness - 1.0) ** 2
        vol = base_vol + smile_effect + 0.02 * expiry  # Term structure
        
        # Price options
        call_px = bs.call_price(spot, strike, expiry, r, vol)
        put_px = bs.put_price(spot, strike, expiry, r, vol)
        
        options_list.append(Option(strike, expiry, 'call', call_px, spot))
        options_list.append(Option(strike, expiry, 'put', put_px, spot))

# Build volatility surface
vol_surface = VolatilitySurface(options_list, spot, r)

print(f"\nVolatility Surface Statistics:")
print(f"  Number of Options: {len(options_list)}")
print(f"  IV Range: {vol_surface.surface_data['iv'].min():.2%} - {vol_surface.surface_data['iv'].max():.2%}")
print(f"  Mean IV: {vol_surface.surface_data['iv'].mean():.2%}")

# Analyze skew for different expiries
print("\nVolatility Skew Analysis:")
for expiry in [0.25, 0.5, 1.0]:
    skew = vol_surface.calculate_volatility_skew(expiry)
    if skew:
        print(f"\n  {expiry*12:.0f}-Month Expiry:")
        print(f"    ATM Vol: {skew['atm_vol']:.2%}")
        print(f"    Put Skew: {skew['put_skew']:.2%}")
        print(f"    Call Skew: {skew['call_skew']:.2%}")
        print(f"    Asymmetry: {skew['skew_asymmetry']:.2%}")


# ==============================================================================
# EXAMPLE 4: Realized Volatility Estimation
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: Realized Volatility Estimation")
print("=" * 80)

# Generate sample price data
np.random.seed(42)
n_days = 252
dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')

# Simulate realistic OHLC data
returns = np.random.normal(0.0005, 0.02, n_days)
close_prices = 100 * np.exp(np.cumsum(returns))

# Add intraday volatility
intraday_vol = 0.015
open_prices = close_prices * np.exp(np.random.normal(0, intraday_vol/2, n_days))
high_prices = np.maximum(open_prices, close_prices) * np.exp(np.abs(np.random.normal(0, intraday_vol, n_days)))
low_prices = np.minimum(open_prices, close_prices) * np.exp(-np.abs(np.random.normal(0, intraday_vol, n_days)))

price_data = pd.DataFrame({
    'date': dates,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices
})

# Calculate realized volatility using different methods
rv_est = RealizedVolatilityEstimator()

rv_parkinson = rv_est.parkinson_volatility(high_prices, low_prices, window=20)
rv_gk = rv_est.garman_klass_volatility(open_prices, high_prices, low_prices, close_prices, window=20)
rv_yz = rv_est.yang_zhang_volatility(open_prices, high_prices, low_prices, close_prices, window=20)

print("\nRealized Volatility Comparison (20-day window):")
print(f"  Parkinson:    Mean = {np.nanmean(rv_parkinson):.2%}, Std = {np.nanstd(rv_parkinson):.2%}")
print(f"  Garman-Klass: Mean = {np.nanmean(rv_gk):.2%}, Std = {np.nanstd(rv_gk):.2%}")
print(f"  Yang-Zhang:   Mean = {np.nanmean(rv_yz):.2%}, Std = {np.nanstd(rv_yz):.2%}")


# ==============================================================================
# EXAMPLE 5: Complete Trading Strategy
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: Volatility Trading Strategy Execution")
print("=" * 80)

# Initialize strategy
strategy = VolatilityTradingStrategy(initial_capital=100000, risk_free_rate=0.05)

print(f"\nInitial Capital: ${strategy.capital:,.2f}")

# Scenario: IV is elevated, expect mean reversion
current_spot = 100
current_iv = 0.35
historical_iv = np.array([0.20, 0.22, 0.21, 0.23, 0.24, 0.22, 0.23, 0.21, 0.20, 0.22])
realized_vol = 0.20

print(f"\nMarket Conditions:")
print(f"  Spot Price: ${current_spot:.2f}")
print(f"  Current IV: {current_iv:.2%}")
print(f"  Historical IV Mean: {np.mean(historical_iv):.2%}")
print(f"  Realized Vol: {realized_vol:.2%}")
print(f"  IV - RV Spread: {(current_iv - realized_vol):.2%}")

# Generate signals
mr_signal = strategy.volatility_mean_reversion_signal(current_iv, historical_iv, threshold=1.5)
iv_rv_signal = strategy.iv_rv_spread_signal(current_iv, realized_vol, threshold=0.05)

print(f"\nTrading Signals:")
print(f"  Mean Reversion Signal: {mr_signal}")
print(f"  IV-RV Spread Signal: {iv_rv_signal}")

# Both signals suggest selling volatility - let's sell a strangle
if mr_signal == 'SELL_VOL' and iv_rv_signal == 'SELL_VOL':
    print("\n>>> EXECUTING: Sell OTM Strangle (Short Volatility)")
    
    # Sell OTM strangle
    put_strike = 95
    call_strike = 105
    expiry = 0.25
    quantity = 5
    
    call_pos, put_pos = strategy.create_strangle_position(
        current_spot, put_strike, call_strike, expiry, current_iv, quantity
    )
    
    # Flip signs for short position
    call_pos.quantity = -quantity
    put_pos.quantity = -quantity
    
    strategy.execute_trade(call_pos, 'OPEN')
    strategy.execute_trade(put_pos, 'OPEN')
    
    total_premium = (call_pos.entry_price + put_pos.entry_price) * quantity
    
    print(f"\n  Position Details:")
    print(f"    Short {abs(call_pos.quantity)} x {call_strike} Calls @ ${call_pos.entry_price:.2f}")
    print(f"    Short {abs(put_pos.quantity)} x {put_strike} Puts @ ${put_pos.entry_price:.2f}")
    print(f"    Total Premium Collected: ${total_premium:.2f}")
    print(f"    Max Profit: ${total_premium:.2f}")
    print(f"    Breakeven Points: ${put_strike - total_premium/quantity:.2f} and ${call_strike + total_premium/quantity:.2f}")

# Check portfolio Greeks
greeks = strategy.calculate_portfolio_greeks()

print(f"\n  Portfolio Greeks:")
print(f"    Delta: {greeks['delta']:.4f}")
print(f"    Gamma: {greeks['gamma']:.4f}")
print(f"    Vega: {greeks['vega']:.4f}")
print(f"    Theta: ${greeks['theta']:.2f} per day")

# Calculate delta hedge
hedge_shares = strategy.delta_hedge_position(current_spot)
print(f"\n  Delta Hedging:")
print(f"    Shares needed: {hedge_shares:.2f}")
print(f"    Hedge cost: ${abs(hedge_shares * current_spot):,.2f}")
print(f"    After-hedge delta: ~0")


# ==============================================================================
# EXAMPLE 6: Scenario Analysis
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: P&L Scenario Analysis")
print("=" * 80)

# Analyze P&L under different scenarios at expiration
scenarios = [
    ('Large Down Move', 85),
    ('Small Down Move', 92),
    ('No Move (ATM)', 100),
    ('Small Up Move', 108),
    ('Large Up Move', 115)
]

print("\nP&L at Expiration (assuming IV normalizes to 25%):")
print(f"{'Scenario':<20} {'Spot':<10} {'Call Value':<12} {'Put Value':<12} {'P&L':<12} {'Return':<12}")
print("-" * 80)

for scenario_name, spot_at_expiry in scenarios:
    # Calculate option values at expiration (zero time value)
    call_value = max(spot_at_expiry - call_strike, 0)
    put_value = max(put_strike - spot_at_expiry, 0)
    
    # P&L = Premium collected - options value (since we're short)
    pnl = total_premium - (call_value + put_value) * quantity
    pnl_pct = (pnl / total_premium) * 100
    
    print(f"{scenario_name:<20} ${spot_at_expiry:<9.2f} ${call_value:<11.2f} ${put_value:<11.2f} ${pnl:<11.2f} {pnl_pct:<11.1f}%")


# ==============================================================================
# EXAMPLE 7: Risk Management
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 7: Risk Management and Position Monitoring")
print("=" * 80)

print("\nPosition Summary:")
print(f"  Number of Positions: {len(strategy.positions)}")
print(f"  Capital Deployed: ${(100000 - strategy.capital):,.2f}")
print(f"  Available Capital: ${strategy.capital:,.2f}")

print("\nRisk Metrics:")
max_loss_down = put_strike * quantity - total_premium
max_loss_up = float('inf')  # Theoretically unlimited for short calls
practical_max_loss = (call_strike + 50 - current_spot) * quantity - total_premium

print(f"  Max Loss (Downside): ${max_loss_down:,.2f}")
print(f"  Max Loss (Upside): Unlimited (Short Calls)")
print(f"  Practical Max Loss (50pt move up): ${practical_max_loss:,.2f}")
print(f"  Max Profit: ${total_premium:.2f}")
print(f"  Profit Probability: ~{70:.0f}% (within breakevens)")

print("\nGreek Exposures:")
print(f"  Delta Exposure: {abs(greeks['delta']):.2f} shares equivalent")
print(f"  Gamma Risk: {greeks['gamma']:.4f} (negative = short gamma)")
print(f"  Vega Exposure: ${greeks['vega']:.2f} per 1% IV change")
print(f"  Theta Decay: ${greeks['theta']:.2f} per day (positive = earning time decay)")

print("\nMonitoring Triggers:")
print("  ⚠️  Close if underlying moves beyond breakevens")
print("  ⚠️  Consider adjusting if delta exceeds ±0.30")
print("  ⚠️  Monitor IV - if it rises significantly, consider closing")
print("  ✓  Ideal outcome: IV drops and time decays with underlying staying near ATM")


# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
This notebook demonstrated:

1. ✓ Black-Scholes pricing and Greeks calculation
2. ✓ Implied volatility back-calculation
3. ✓ Volatility surface construction and analysis
4. ✓ Multiple realized volatility estimators
5. ✓ Complete trading strategy execution
6. ✓ Scenario analysis and P&L projections
7. ✓ Risk management and position monitoring

Key Takeaways:
- Volatility trading requires sophisticated modeling and risk management
- Delta hedging isolates volatility P&L from directional moves
- Multiple signals (mean reversion, IV-RV spread) improve decision quality
- Greeks provide essential risk metrics for position management
- Always define maximum loss and have exit strategies

Next Steps:
- Backtest strategies on historical data
- Implement dynamic rebalancing
- Add transaction cost modeling
- Develop more sophisticated entry/exit rules
- Integrate real-time data feeds
""")

print("=" * 80)
print("Demo Complete!")
print("=" * 80)
