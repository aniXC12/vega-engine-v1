# Advanced Volatility Trading Algorithm

A sophisticated Python implementation of multiple volatility trading strategies including volatility surface modeling, delta hedging, mean reversion trading, and implied vs realized volatility arbitrage.

## 🎯 Overview

This algorithm combines several advanced concepts in options trading and volatility arbitrage:

- **Volatility Surface Modeling**: Build and analyze 3D volatility surfaces from options chains
- **Greek Calculation**: Full Black-Scholes Greeks (Delta, Gamma, Vega, Theta)
- **Delta Hedging**: Dynamic position hedging to maintain delta neutrality
- **Arbitrage Detection**: Identify calendar spread and butterfly arbitrage opportunities
- **Multiple Volatility Estimators**: Parkinson, Garman-Klass, and Yang-Zhang methods
- **Signal Generation**: IV mean reversion and IV-RV spread strategies
- **Backtesting Framework**: Complete framework for strategy evaluation

## 📊 Key Features

### 1. Volatility Surface Analysis
```python
vol_surface = VolatilitySurface(options, spot_price, risk_free_rate)
interpolated_vol = vol_surface.get_interpolated_vol(moneyness=1.05, expiry=0.5)
arbitrage_opps = vol_surface.detect_arbitrage_opportunities()
skew = vol_surface.calculate_volatility_skew(expiry=0.25)
```

### 2. Realized Volatility Estimation
Multiple estimators for more accurate volatility measurement:
- **Parkinson**: Uses high-low range (more efficient than close-to-close)
- **Garman-Klass**: Incorporates open, high, low, close
- **Yang-Zhang**: Handles overnight jumps and drift

### 3. Trading Strategies
- **Long/Short Straddles**: Bet on volatility level
- **Long/Short Strangles**: Cheaper volatility plays with OTM options
- **Delta-Hedged Portfolios**: Isolate volatility P&L from directional moves
- **Calendar Spreads**: Exploit term structure dislocations

### 4. Risk Management
- Real-time Greek calculation and monitoring
- Dynamic delta hedging recommendations
- Portfolio-level risk aggregation
- Position sizing based on available capital

## 🛠️ Technical Implementation

### Black-Scholes Model
Complete implementation including:
- Option pricing (calls and puts)
- Implied volatility calculation using Newton-Raphson
- All Greeks with analytical formulas

### Volatility Surface
- 3D interpolation using cubic and linear methods
- Moneyness-based representation
- Skew and smile analysis
- Temporal interpolation for any expiry

### Signal Generation

**Mean Reversion Strategy**
```python
signal = strategy.volatility_mean_reversion_signal(
    current_iv=0.30,
    historical_iv=np.array([0.22, 0.24, 0.23, ...]),
    threshold=1.5  # Z-score threshold
)
```

**IV-RV Spread Strategy**
```python
signal = strategy.iv_rv_spread_signal(
    implied_vol=0.30,
    realized_vol=0.20,
    threshold=0.05  # 5% spread threshold
)
```

## 📈 Usage Examples

### Basic Usage
```python
from volatility_trading_algorithm import *

# Initialize strategy
strategy = VolatilityTradingStrategy(initial_capital=100000)

# Create a long straddle
call_pos, put_pos = strategy.create_straddle_position(
    spot_price=100,
    strike=100,
    expiry=0.25,
    implied_vol=0.25,
    quantity=10
)

# Open positions
strategy.execute_trade(call_pos, 'OPEN')
strategy.execute_trade(put_pos, 'OPEN')

# Check portfolio Greeks
greeks = strategy.calculate_portfolio_greeks()
print(f"Portfolio Delta: {greeks['delta']}")
print(f"Portfolio Vega: {greeks['vega']}")

# Calculate hedge
hedge_shares = strategy.delta_hedge_position(spot_price=100)
print(f"Hedge with {hedge_shares} shares")
```

### Advanced: Building a Volatility Surface
```python
# Create options chain
options = []
for strike in [90, 95, 100, 105, 110]:
    for expiry in [0.25, 0.5, 1.0]:
        # Add your options data
        options.append(Option(...))

# Build surface
vol_surface = VolatilitySurface(options, spot_price=100, risk_free_rate=0.05)

# Analyze skew
skew_analysis = vol_surface.calculate_volatility_skew(expiry=0.25)
print(f"ATM Vol: {skew_analysis['atm_vol']:.2%}")
print(f"Put Skew: {skew_analysis['put_skew']:.2%}")

# Find arbitrage
arbitrage_opps = vol_surface.detect_arbitrage_opportunities()
for opp in arbitrage_opps:
    print(f"Opportunity: {opp['type']} - {opp['signal']}")
```

### Backtesting
```python
# Prepare data
price_data = pd.DataFrame({
    'date': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...]
})

# Run backtest
results = strategy.backtest_strategy(price_data, options_data)

# Generate performance metrics
metrics = strategy.generate_performance_metrics(results)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## 📊 Performance Metrics

The algorithm calculates comprehensive performance metrics:

- **Total Return**: Overall strategy return
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Greek Exposures**: Real-time risk monitoring

## 🧮 Mathematical Foundation

### Black-Scholes Formula

**Call Option:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Greeks Formulas

**Delta:**
- Call: Δ_call = N(d₁)
- Put: Δ_put = N(d₁) - 1

**Gamma:**
```
Γ = N'(d₁) / (S₀σ√T)
```

**Vega:**
```
ν = S₀N'(d₁)√T
```

**Theta:**
- Call: Θ_call = -S₀N'(d₁)σ/(2√T) - rKe^(-rT)N(d₂)
- Put: Θ_put = -S₀N'(d₁)σ/(2√T) + rKe^(-rT)N(-d₂)

### Yang-Zhang Volatility Estimator

More robust volatility estimator that handles overnight jumps:

```
σ²_YZ = σ²_open + k·σ²_close + (1-k)·σ²_RS

where:
- σ²_open: Open-to-close volatility
- σ²_close: Close-to-close volatility  
- σ²_RS: Rogers-Satchell volatility
- k: Optimal weighting factor
```

## 🎓 Strategy Logic

### 1. Mean Reversion Strategy
- Calculate z-score of current IV vs historical mean
- When IV is >1.5 std devs above mean → SELL volatility
- When IV is >1.5 std devs below mean → BUY volatility
- Rationale: Volatility tends to revert to its long-term average

### 2. IV-RV Spread Strategy
- Compare implied volatility to realized volatility
- When IV > RV by threshold → SELL volatility (overpriced)
- When IV < RV by threshold → BUY volatility (underpriced)
- Rationale: IV should approximate future realized volatility

### 3. Delta-Hedged Volatility Trading
- Take volatility positions (long/short options)
- Dynamically hedge delta exposure with underlying
- P&L driven by gamma/theta interaction and vega
- Rationale: Isolate volatility P&L from directional moves

## ⚠️ Risk Considerations

This is an educational implementation. Real-world trading requires:

1. **Transaction Costs**: Include bid-ask spreads and commissions
2. **Slippage**: Market impact on larger orders
3. **Margin Requirements**: Capital needed for short options
4. **Early Exercise**: American options can be exercised early
5. **Dividends**: Affect option pricing and hedging
6. **Interest Rates**: Can vary and affect forward prices
7. **Model Risk**: Black-Scholes assumptions may not hold
8. **Pin Risk**: Risk near expiration at strike price
9. **Gap Risk**: Overnight/weekend moves can violate continuous hedging assumptions

## 🔧 Requirements

```python
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
```

Install dependencies:
```bash
pip install numpy pandas scipy
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/volatility-trading-algorithm.git
cd volatility-trading-algorithm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the demo:
```bash
python volatility_trading_algorithm.py
```

## 📚 Further Reading

### Academic Papers
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security Price Volatilities from Historical Data"
- Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices"

### Books
- "Option Volatility and Pricing" by Sheldon Natenberg
- "The Volatility Surface" by Jim Gatheral
- "Dynamic Hedging" by Nassim Taleb

### Online Resources
- [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/)
- [Options Industry Council](https://www.optionseducation.org/)

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional volatility estimators (EWMA, GARCH)
- More sophisticated hedging strategies
- Machine learning for volatility forecasting
- Real-time data integration
- Advanced arbitrage strategies (variance swaps, volatility swaps)
- Transaction cost modeling

## 📄 License

MIT License - feel free to use this code for educational and commercial purposes.

## ⚖️ Disclaimer

**This software is for educational purposes only. Trading options and volatility involves substantial risk of loss. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making any investment decisions.**

## 👨‍💻 Author

- LinkedIn: (https://www.linkedin.com/in/anish-rudra/)
- Email: anishrudra1@gmail.com

## 🙏 Acknowledgments

- Thanks to the quantitative finance community for open-source contributions
- Inspired by professional market-making and volatility arbitrage strategies
- Built with insights from academic research and practitioner knowledge

---

**Star this repository if you find it useful! ⭐**
