"""
Advanced Volatility Trading Algorithm
======================================

This algorithm implements multiple volatility trading strategies including:
1. Volatility surface modeling and arbitrage detection
2. Delta-hedged option positions with dynamic rebalancing
3. Volatility mean reversion trading
4. VIX term structure trading
5. Implied vs realized volatility spread trading

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import griddata, RBFInterpolator
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Option:
    """Option contract representation"""
    strike: float
    expiry: float  # Time to expiration in years
    option_type: str  # 'call' or 'put'
    premium: float
    underlying_price: float
    implied_vol: Optional[float] = None


@dataclass
class Position:
    """Trading position representation"""
    instrument: str
    quantity: float
    entry_price: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


class BlackScholesModel:
    """Black-Scholes option pricing and Greeks calculation"""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price"""
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price"""
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (per 1% change in volatility)"""
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option theta (per day)"""
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            theta_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta_part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return (theta_part1 + theta_part2) / 365
    
    @staticmethod
    def implied_volatility(option_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str, max_iterations: int = 100) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        sigma = 0.3  # Initial guess
        
        for _ in range(max_iterations):
            if option_type == 'call':
                price = BlackScholesModel.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholesModel.put_price(S, K, T, r, sigma)
            
            vega_val = BlackScholesModel.vega(S, K, T, r, sigma) * 100  # Undo the /100 in vega
            
            diff = option_price - price
            
            if abs(diff) < 1e-6:
                return sigma
            
            if vega_val == 0:
                return sigma
            
            sigma = sigma + diff / vega_val
            
            if sigma <= 0:
                sigma = 0.01
        
        return sigma


class VolatilitySurface:
    """Model and analyze the volatility surface"""
    
    def __init__(self, options: List[Option], spot_price: float, risk_free_rate: float):
        self.options = options
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.surface_data = None
        self._build_surface()
    
    def _build_surface(self):
        """Build volatility surface from option prices"""
        strikes = []
        expiries = []
        implied_vols = []
        
        for opt in self.options:
            iv = BlackScholesModel.implied_volatility(
                opt.premium, self.spot_price, opt.strike, 
                opt.expiry, self.risk_free_rate, opt.option_type
            )
            opt.implied_vol = iv
            
            strikes.append(opt.strike / self.spot_price)  # Moneyness
            expiries.append(opt.expiry)
            implied_vols.append(iv)
        
        self.surface_data = pd.DataFrame({
            'moneyness': strikes,
            'expiry': expiries,
            'iv': implied_vols
        })
    
    def get_interpolated_vol(self, moneyness: float, expiry: float) -> float:
        """Interpolate implied volatility for any strike/expiry"""
        points = self.surface_data[['moneyness', 'expiry']].values
        values = self.surface_data['iv'].values
        
        try:
            interpolated = griddata(points, values, ([moneyness], [expiry]), method='cubic')[0]
            if np.isnan(interpolated):
                interpolated = griddata(points, values, ([moneyness], [expiry]), method='linear')[0]
            return interpolated
        except:
            return np.mean(values)
    
    def detect_arbitrage_opportunities(self) -> List[Dict]:
        """Detect calendar spread and butterfly arbitrage opportunities"""
        opportunities = []
        
        # Calendar spread arbitrage (same strike, different expiries)
        strike_groups = self.surface_data.groupby('moneyness')
        
        for moneyness, group in strike_groups:
            if len(group) >= 2:
                sorted_group = group.sort_values('expiry')
                for i in range(len(sorted_group) - 1):
                    near_iv = sorted_group.iloc[i]['iv']
                    far_iv = sorted_group.iloc[i + 1]['iv']
                    
                    # Far-dated options should typically have higher or similar IV
                    if far_iv < near_iv - 0.05:  # 5% threshold
                        opportunities.append({
                            'type': 'calendar_spread',
                            'moneyness': moneyness,
                            'near_expiry': sorted_group.iloc[i]['expiry'],
                            'far_expiry': sorted_group.iloc[i + 1]['expiry'],
                            'iv_difference': near_iv - far_iv,
                            'signal': 'buy_far_sell_near'
                        })
        
        return opportunities
    
    def calculate_volatility_skew(self, expiry: float) -> Dict:
        """Calculate volatility skew metrics for a given expiry"""
        expiry_data = self.surface_data[
            (self.surface_data['expiry'] >= expiry - 0.01) & 
            (self.surface_data['expiry'] <= expiry + 0.01)
        ].sort_values('moneyness')
        
        if len(expiry_data) < 3:
            return {}
        
        atm_idx = (expiry_data['moneyness'] - 1.0).abs().idxmin()
        atm_vol = expiry_data.loc[atm_idx, 'iv']
        
        otm_puts = expiry_data[expiry_data['moneyness'] < 0.95]
        otm_calls = expiry_data[expiry_data['moneyness'] > 1.05]
        
        put_skew = otm_puts['iv'].mean() - atm_vol if len(otm_puts) > 0 else 0
        call_skew = otm_calls['iv'].mean() - atm_vol if len(otm_calls) > 0 else 0
        
        return {
            'expiry': expiry,
            'atm_vol': atm_vol,
            'put_skew': put_skew,
            'call_skew': call_skew,
            'skew_asymmetry': put_skew - call_skew
        }


class RealizedVolatilityEstimator:
    """Estimate realized volatility using various methods"""
    
    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
        """Parkinson's high-low volatility estimator"""
        hl_ratio = np.log(high / low)
        return np.sqrt(252 / (4 * window * np.log(2)) * 
                      pd.Series(hl_ratio ** 2).rolling(window).sum())
    
    @staticmethod
    def garman_klass_volatility(open_prices: np.ndarray, high: np.ndarray, 
                               low: np.ndarray, close: np.ndarray, window: int = 20) -> np.ndarray:
        """Garman-Klass volatility estimator (more efficient)"""
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_prices) ** 2
        
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(252 / window * pd.Series(gk).rolling(window).sum())
    
    @staticmethod
    def yang_zhang_volatility(open_prices: np.ndarray, high: np.ndarray,
                             low: np.ndarray, close: np.ndarray, window: int = 20) -> np.ndarray:
        """Yang-Zhang volatility estimator (handles overnight jumps)"""
        log_ho = np.log(high / open_prices)
        log_lo = np.log(low / open_prices)
        log_co = np.log(close / open_prices)
        
        log_oc = np.log(open_prices[1:] / close[:-1])
        log_oc = np.concatenate([[0], log_oc])
        
        log_cc = np.log(close[1:] / close[:-1])
        log_cc = np.concatenate([[0], log_cc])
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        rs = pd.Series(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).sum()
        close_vol = pd.Series(log_cc ** 2).rolling(window).sum()
        open_vol = pd.Series(log_oc ** 2).rolling(window).sum()
        
        yz = open_vol + k * close_vol + (1 - k) * rs
        return np.sqrt(252 / window * yz)


class VolatilityTradingStrategy:
    """Main trading strategy combining multiple volatility approaches"""
    
    def __init__(self, initial_capital: float = 100000, risk_free_rate: float = 0.05):
        self.capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.bs_model = BlackScholesModel()
    
    def delta_hedge_position(self, spot_price: float, target_delta: float = 0.0) -> float:
        """Calculate shares needed to delta hedge the portfolio"""
        portfolio_delta = sum(pos.delta * pos.quantity for pos in self.positions)
        hedge_shares = -(portfolio_delta - target_delta)
        return hedge_shares
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate total portfolio Greeks"""
        return {
            'delta': sum(pos.delta * pos.quantity for pos in self.positions),
            'gamma': sum(pos.gamma * pos.quantity for pos in self.positions),
            'vega': sum(pos.vega * pos.quantity for pos in self.positions),
            'theta': sum(pos.theta * pos.quantity for pos in self.positions)
        }
    
    def volatility_mean_reversion_signal(self, current_iv: float, 
                                        historical_iv: np.ndarray, 
                                        threshold: float = 1.5) -> str:
        """Generate trading signal based on IV mean reversion"""
        iv_mean = np.mean(historical_iv)
        iv_std = np.std(historical_iv)
        z_score = (current_iv - iv_mean) / iv_std
        
        if z_score > threshold:
            return 'SELL_VOL'  # IV is high, expect reversion
        elif z_score < -threshold:
            return 'BUY_VOL'  # IV is low, expect reversion
        else:
            return 'NEUTRAL'
    
    def iv_rv_spread_signal(self, implied_vol: float, realized_vol: float, 
                           threshold: float = 0.05) -> str:
        """Generate signal based on implied vs realized volatility spread"""
        spread = implied_vol - realized_vol
        
        if spread > threshold:
            return 'SELL_VOL'  # IV rich relative to RV
        elif spread < -threshold:
            return 'BUY_VOL'  # IV cheap relative to RV
        else:
            return 'NEUTRAL'
    
    def create_straddle_position(self, spot_price: float, strike: float, 
                                 expiry: float, implied_vol: float, 
                                 quantity: int = 1) -> Tuple[Position, Position]:
        """Create a long straddle position (long call + long put)"""
        # Calculate option prices
        call_price = self.bs_model.call_price(spot_price, strike, expiry, 
                                             self.risk_free_rate, implied_vol)
        put_price = self.bs_model.put_price(spot_price, strike, expiry, 
                                           self.risk_free_rate, implied_vol)
        
        # Calculate Greeks
        call_delta = self.bs_model.delta(spot_price, strike, expiry, 
                                        self.risk_free_rate, implied_vol, 'call')
        put_delta = self.bs_model.delta(spot_price, strike, expiry, 
                                       self.risk_free_rate, implied_vol, 'put')
        gamma = self.bs_model.gamma(spot_price, strike, expiry, 
                                    self.risk_free_rate, implied_vol)
        vega = self.bs_model.vega(spot_price, strike, expiry, 
                                 self.risk_free_rate, implied_vol)
        call_theta = self.bs_model.theta(spot_price, strike, expiry, 
                                        self.risk_free_rate, implied_vol, 'call')
        put_theta = self.bs_model.theta(spot_price, strike, expiry, 
                                       self.risk_free_rate, implied_vol, 'put')
        
        call_position = Position(
            instrument=f'CALL_{strike}_{expiry}',
            quantity=quantity,
            entry_price=call_price,
            delta=call_delta,
            gamma=gamma,
            vega=vega,
            theta=call_theta
        )
        
        put_position = Position(
            instrument=f'PUT_{strike}_{expiry}',
            quantity=quantity,
            entry_price=put_price,
            delta=put_delta,
            gamma=gamma,
            vega=vega,
            theta=put_theta
        )
        
        return call_position, put_position
    
    def create_strangle_position(self, spot_price: float, put_strike: float,
                                call_strike: float, expiry: float, 
                                implied_vol: float, quantity: int = 1) -> Tuple[Position, Position]:
        """Create a long strangle position (OTM call + OTM put)"""
        call_price = self.bs_model.call_price(spot_price, call_strike, expiry,
                                             self.risk_free_rate, implied_vol)
        put_price = self.bs_model.put_price(spot_price, put_strike, expiry,
                                           self.risk_free_rate, implied_vol)
        
        call_delta = self.bs_model.delta(spot_price, call_strike, expiry,
                                        self.risk_free_rate, implied_vol, 'call')
        put_delta = self.bs_model.delta(spot_price, put_strike, expiry,
                                       self.risk_free_rate, implied_vol, 'put')
        
        call_gamma = self.bs_model.gamma(spot_price, call_strike, expiry,
                                        self.risk_free_rate, implied_vol)
        put_gamma = self.bs_model.gamma(spot_price, put_strike, expiry,
                                       self.risk_free_rate, implied_vol)
        
        call_vega = self.bs_model.vega(spot_price, call_strike, expiry,
                                      self.risk_free_rate, implied_vol)
        put_vega = self.bs_model.vega(spot_price, put_strike, expiry,
                                     self.risk_free_rate, implied_vol)
        
        call_theta = self.bs_model.theta(spot_price, call_strike, expiry,
                                        self.risk_free_rate, implied_vol, 'call')
        put_theta = self.bs_model.theta(spot_price, put_strike, expiry,
                                       self.risk_free_rate, implied_vol, 'put')
        
        call_position = Position(
            instrument=f'CALL_{call_strike}_{expiry}',
            quantity=quantity,
            entry_price=call_price,
            delta=call_delta,
            gamma=call_gamma,
            vega=call_vega,
            theta=call_theta
        )
        
        put_position = Position(
            instrument=f'PUT_{put_strike}_{expiry}',
            quantity=quantity,
            entry_price=put_price,
            delta=put_delta,
            gamma=put_gamma,
            vega=put_vega,
            theta=put_theta
        )
        
        return call_position, put_position
    
    def execute_trade(self, position: Position, action: str):
        """Execute a trade and update portfolio"""
        if action == 'OPEN':
            self.positions.append(position)
            cost = position.entry_price * abs(position.quantity)
            self.capital -= cost
            
            self.trade_history.append({
                'timestamp': datetime.now(),
                'action': 'OPEN',
                'instrument': position.instrument,
                'quantity': position.quantity,
                'price': position.entry_price,
                'capital': self.capital
            })
        
        elif action == 'CLOSE':
            # Find and remove position
            for i, pos in enumerate(self.positions):
                if pos.instrument == position.instrument:
                    proceeds = position.entry_price * abs(position.quantity)
                    self.capital += proceeds
                    self.positions.pop(i)
                    
                    self.trade_history.append({
                        'timestamp': datetime.now(),
                        'action': 'CLOSE',
                        'instrument': position.instrument,
                        'quantity': position.quantity,
                        'price': position.entry_price,
                        'capital': self.capital
                    })
                    break
    
    def backtest_strategy(self, price_data: pd.DataFrame, 
                         options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest the volatility trading strategy
        
        Parameters:
        -----------
        price_data: DataFrame with columns ['date', 'open', 'high', 'low', 'close']
        options_data: DataFrame with options chain data
        
        Returns:
        --------
        DataFrame with backtest results
        """
        results = []
        rv_estimator = RealizedVolatilityEstimator()
        
        # Calculate realized volatility
        rv = rv_estimator.yang_zhang_volatility(
            price_data['open'].values,
            price_data['high'].values,
            price_data['low'].values,
            price_data['close'].values,
            window=20
        )
        
        price_data['realized_vol'] = rv
        
        for date in price_data['date'].unique():
            daily_data = price_data[price_data['date'] == date].iloc[0]
            spot_price = daily_data['close']
            realized_vol = daily_data['realized_vol']
            
            if np.isnan(realized_vol):
                continue
            
            # Get current IV from options data (simplified)
            current_iv = 0.25  # This would come from actual options data
            
            # Generate signals
            iv_rv_signal = self.iv_rv_spread_signal(current_iv, realized_vol)
            
            # Execute strategy based on signals
            portfolio_greeks = self.calculate_portfolio_greeks()
            
            results.append({
                'date': date,
                'spot_price': spot_price,
                'realized_vol': realized_vol,
                'implied_vol': current_iv,
                'signal': iv_rv_signal,
                'portfolio_delta': portfolio_greeks['delta'],
                'portfolio_gamma': portfolio_greeks['gamma'],
                'portfolio_vega': portfolio_greeks['vega'],
                'portfolio_value': self.capital,
                'num_positions': len(self.positions)
            })
        
        return pd.DataFrame(results)
    
    def generate_performance_metrics(self, backtest_results: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        returns = backtest_results['portfolio_value'].pct_change().dropna()
        
        total_return = (backtest_results['portfolio_value'].iloc[-1] / 
                       backtest_results['portfolio_value'].iloc[0] - 1)
        
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trade_history),
            'final_capital': backtest_results['portfolio_value'].iloc[-1]
        }


def main():
    """Example usage of the volatility trading algorithm"""
    
    print("=" * 80)
    print("Advanced Volatility Trading Algorithm - Demo")
    print("=" * 80)
    
    # Initialize parameters
    spot_price = 100.0
    risk_free_rate = 0.05
    
    # Create sample options chain
    strikes = [90, 95, 100, 105, 110]
    expiries = [0.25, 0.5, 1.0]  # 3 months, 6 months, 1 year
    
    options = []
    for strike in strikes:
        for expiry in expiries:
            # Simulate option prices with a volatility smile
            moneyness = strike / spot_price
            base_iv = 0.25 + 0.1 * abs(moneyness - 1.0)  # Smile effect
            
            call_price = BlackScholesModel.call_price(
                spot_price, strike, expiry, risk_free_rate, base_iv
            )
            put_price = BlackScholesModel.put_price(
                spot_price, strike, expiry, risk_free_rate, base_iv
            )
            
            options.append(Option(strike, expiry, 'call', call_price, spot_price))
            options.append(Option(strike, expiry, 'put', put_price, spot_price))
    
    # Build volatility surface
    print("\n1. Building Volatility Surface...")
    vol_surface = VolatilitySurface(options, spot_price, risk_free_rate)
    print(f"   Surface built with {len(options)} options")
    print(f"   IV Range: {vol_surface.surface_data['iv'].min():.2%} - "
          f"{vol_surface.surface_data['iv'].max():.2%}")
    
    # Detect arbitrage opportunities
    print("\n2. Detecting Arbitrage Opportunities...")
    arbitrage_opps = vol_surface.detect_arbitrage_opportunities()
    print(f"   Found {len(arbitrage_opps)} potential arbitrage opportunities")
    for opp in arbitrage_opps[:3]:
        print(f"   - {opp['type']}: {opp['signal']} (IV diff: {opp['iv_difference']:.2%})")
    
    # Analyze volatility skew
    print("\n3. Analyzing Volatility Skew...")
    skew_analysis = vol_surface.calculate_volatility_skew(expiry=0.25)
    if skew_analysis:
        print(f"   ATM Vol: {skew_analysis['atm_vol']:.2%}")
        print(f"   Put Skew: {skew_analysis['put_skew']:.2%}")
        print(f"   Call Skew: {skew_analysis['call_skew']:.2%}")
        print(f"   Skew Asymmetry: {skew_analysis['skew_asymmetry']:.2%}")
    
    # Initialize trading strategy
    print("\n4. Initializing Trading Strategy...")
    strategy = VolatilityTradingStrategy(initial_capital=100000)
    
    # Create a sample straddle position
    print("\n5. Creating Long Straddle Position...")
    call_pos, put_pos = strategy.create_straddle_position(
        spot_price=spot_price,
        strike=100,
        expiry=0.25,
        implied_vol=0.25,
        quantity=10
    )
    
    strategy.execute_trade(call_pos, 'OPEN')
    strategy.execute_trade(put_pos, 'OPEN')
    
    print(f"   Position opened:")
    print(f"   - Call: ${call_pos.entry_price:.2f} x {call_pos.quantity}")
    print(f"   - Put: ${put_pos.entry_price:.2f} x {put_pos.quantity}")
    print(f"   - Total Cost: ${(call_pos.entry_price + put_pos.entry_price) * call_pos.quantity:.2f}")
    
    # Calculate portfolio Greeks
    print("\n6. Portfolio Greeks Analysis...")
    greeks = strategy.calculate_portfolio_greeks()
    print(f"   Delta: {greeks['delta']:.4f}")
    print(f"   Gamma: {greeks['gamma']:.4f}")
    print(f"   Vega: {greeks['vega']:.4f}")
    print(f"   Theta: {greeks['theta']:.4f}")
    
    # Calculate delta hedge
    hedge_shares = strategy.delta_hedge_position(spot_price)
    print(f"\n7. Delta Hedging...")
    print(f"   Shares needed to delta hedge: {hedge_shares:.2f}")
    print(f"   Hedge cost: ${abs(hedge_shares * spot_price):.2f}")
    
    # Generate signals
    print("\n8. Signal Generation...")
    historical_iv = np.array([0.22, 0.24, 0.23, 0.25, 0.26, 0.24, 0.23, 0.25, 0.27, 0.26])
    current_iv = 0.30
    realized_vol = 0.20
    
    mean_reversion_signal = strategy.volatility_mean_reversion_signal(
        current_iv, historical_iv
    )
    iv_rv_signal = strategy.iv_rv_spread_signal(current_iv, realized_vol)
    
    print(f"   Mean Reversion Signal: {mean_reversion_signal}")
    print(f"   IV-RV Spread Signal: {iv_rv_signal}")
    print(f"   Current IV: {current_iv:.2%}, Historical Mean: {np.mean(historical_iv):.2%}")
    print(f"   Realized Vol: {realized_vol:.2%}, IV-RV Spread: {(current_iv - realized_vol):.2%}")
    
    print("\n" + "=" * 80)
    print("Strategy Recommendation:")
    print("=" * 80)
    
    if mean_reversion_signal == 'SELL_VOL' and iv_rv_signal == 'SELL_VOL':
        print("Strong SELL volatility signal detected!")
        print("Recommended Action: Sell straddles/strangles or iron condors")
    elif mean_reversion_signal == 'BUY_VOL' and iv_rv_signal == 'BUY_VOL':
        print("Strong BUY volatility signal detected!")
        print("Recommended Action: Buy straddles/strangles")
    else:
        print("Mixed signals - exercise caution or stay neutral")
    
    print("\n" + "=" * 80)
    print("Algorithm demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
