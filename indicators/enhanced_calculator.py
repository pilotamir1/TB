"""
Enhanced Indicator Calculator with complete implementations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

class EnhancedIndicatorCalculator:
    """Enhanced calculator for all technical indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Indicator registry with metadata
        self.indicator_registry = {
            # Donchian Channels
            'Donchian_High_20': {'min_lookback': 20, 'category': 'volatility', 'function': 'calculate_donchian'},
            'Donchian_Low_20': {'min_lookback': 20, 'category': 'volatility', 'function': 'calculate_donchian'},
            'Donchian_High_55': {'min_lookback': 55, 'category': 'volatility', 'function': 'calculate_donchian'},
            'Donchian_Low_55': {'min_lookback': 55, 'category': 'volatility', 'function': 'calculate_donchian'},
            
            # Volume-based indicators
            'OBV': {'min_lookback': 1, 'category': 'volume', 'function': 'calculate_obv'},
            'CMF_10': {'min_lookback': 10, 'category': 'volume', 'function': 'calculate_cmf'},
            'CMF_20': {'min_lookback': 20, 'category': 'volume', 'function': 'calculate_cmf'},
            'CMF_50': {'min_lookback': 50, 'category': 'volume', 'function': 'calculate_cmf'},
            'AD_Line': {'min_lookback': 1, 'category': 'volume', 'function': 'calculate_ad_line'},
            'Force_Index_2': {'min_lookback': 2, 'category': 'volume', 'function': 'calculate_force_index'},
            
            # Volatility indicators
            'Chaikin_Volatility_14': {'min_lookback': 14, 'category': 'volatility', 'function': 'calculate_chaikin_volatility'},
            
            # Momentum indicators
            'QStick_20': {'min_lookback': 20, 'category': 'momentum', 'function': 'calculate_qstick'},
            'KST': {'min_lookback': 264, 'category': 'momentum', 'function': 'calculate_kst'},
            'TSI_25_13': {'min_lookback': 38, 'category': 'momentum', 'function': 'calculate_tsi'},
            'TSI_13_7': {'min_lookback': 20, 'category': 'momentum', 'function': 'calculate_tsi'},
            
            # Price transformation
            'Heikin_Ashi_Open': {'min_lookback': 2, 'category': 'price', 'function': 'calculate_heikin_ashi'},
            'Heikin_Ashi_High': {'min_lookback': 2, 'category': 'price', 'function': 'calculate_heikin_ashi'},
            'Heikin_Ashi_Low': {'min_lookback': 2, 'category': 'price', 'function': 'calculate_heikin_ashi'},
            'Heikin_Ashi_Close': {'min_lookback': 2, 'category': 'price', 'function': 'calculate_heikin_ashi'},
        }
    
    def get_indicator_info(self, name: str) -> Dict[str, Any]:
        """Get indicator metadata"""
        return self.indicator_registry.get(name, {})
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period required"""
        return max([info.get('min_lookback', 1) for info in self.indicator_registry.values()])
    
    def calculate_donchian(self, df: pd.DataFrame, name: str, period: int = None) -> pd.Series:
        """Calculate Donchian Channel High/Low"""
        if period is None:
            # Extract period from name
            if '20' in name:
                period = 20
            elif '55' in name:
                period = 55
            else:
                period = 20
        
        if 'High' in name:
            return df['high'].rolling(window=period).max()
        else:  # Low
            return df['low'].rolling(window=period).min()
    
    def calculate_obv(self, df: pd.DataFrame, name: str = 'OBV') -> pd.Series:
        """Calculate On Balance Volume"""
        obv = np.zeros(len(df))
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index, name=name)
    
    def calculate_cmf(self, df: pd.DataFrame, name: str, period: int = None) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        if period is None:
            # Extract period from name
            if '10' in name:
                period = 10
            elif '20' in name:
                period = 20
            elif '50' in name:
                period = 50
            else:
                period = 20
        
        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)  # Handle division by zero when high == low
        
        # Money Flow Volume
        mfv = mfm * df['volume']
        
        # Chaikin Money Flow
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        
        return cmf.fillna(0)
    
    def calculate_ad_line(self, df: pd.DataFrame, name: str = 'AD_Line') -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)  # Handle division by zero
        
        # Money Flow Volume
        mfv = mfm * df['volume']
        
        # Accumulation/Distribution Line (cumulative sum)
        ad_line = mfv.cumsum()
        
        return ad_line
    
    def calculate_force_index(self, df: pd.DataFrame, name: str, period: int = None) -> pd.Series:
        """Calculate Force Index"""
        if period is None:
            # Extract period from name
            if '2' in name:
                period = 2
            else:
                period = 2
        
        # Raw Force Index = Volume * (Close - Previous Close)
        raw_fi = df['volume'] * (df['close'] - df['close'].shift(1))
        
        if period == 1:
            return raw_fi
        else:
            # Smoothed Force Index using EMA
            return raw_fi.ewm(span=period).mean()
    
    def calculate_chaikin_volatility(self, df: pd.DataFrame, name: str, period: int = None) -> pd.Series:
        """Calculate Chaikin Volatility"""
        if period is None:
            # Extract period from name
            if '14' in name:
                period = 14
            else:
                period = 14
        
        # High-Low spread
        hl_spread = df['high'] - df['low']
        
        # EMA of High-Low spread
        ema_hl = hl_spread.ewm(span=period).mean()
        
        # Chaikin Volatility: % change of EMA over specified period
        cv = ((ema_hl - ema_hl.shift(period)) / ema_hl.shift(period)) * 100
        
        return cv.fillna(0)
    
    def calculate_qstick(self, df: pd.DataFrame, name: str, period: int = None) -> pd.Series:
        """Calculate QStick indicator"""
        if period is None:
            # Extract period from name
            if '20' in name:
                period = 20
            else:
                period = 20
        
        # QStick = SMA of (Close - Open)
        close_open_diff = df['close'] - df['open']
        qstick = close_open_diff.rolling(window=period).mean()
        
        return qstick
    
    def calculate_kst(self, df: pd.DataFrame, name: str = 'KST') -> pd.Series:
        """Calculate Know Sure Thing (KST) oscillator"""
        # Standard KST parameters
        roc1_period, roc1_ma = 10, 10
        roc2_period, roc2_ma = 15, 10
        roc3_period, roc3_ma = 20, 10
        roc4_period, roc4_ma = 30, 15
        
        # Calculate ROCs
        roc1 = ((df['close'] / df['close'].shift(roc1_period)) - 1) * 100
        roc2 = ((df['close'] / df['close'].shift(roc2_period)) - 1) * 100
        roc3 = ((df['close'] / df['close'].shift(roc3_period)) - 1) * 100
        roc4 = ((df['close'] / df['close'].shift(roc4_period)) - 1) * 100
        
        # Smooth ROCs with SMAs
        roc1_ma = roc1.rolling(window=roc1_ma).mean()
        roc2_ma = roc2.rolling(window=roc2_ma).mean()
        roc3_ma = roc3.rolling(window=roc3_ma).mean()
        roc4_ma = roc4.rolling(window=roc4_ma).mean()
        
        # Weighted KST
        kst = (roc1_ma * 1) + (roc2_ma * 2) + (roc3_ma * 3) + (roc4_ma * 4)
        
        return kst
    
    def calculate_tsi(self, df: pd.DataFrame, name: str, long_period: int = None, 
                     short_period: int = None) -> pd.Series:
        """Calculate True Strength Index (TSI)"""
        if long_period is None or short_period is None:
            # Extract periods from name
            if '25_13' in name:
                long_period, short_period = 25, 13
            elif '13_7' in name:
                long_period, short_period = 13, 7
            else:
                long_period, short_period = 25, 13
        
        # Price change
        price_change = df['close'] - df['close'].shift(1)
        
        # Absolute price change
        abs_price_change = price_change.abs()
        
        # Double smoothed price change
        pc_smooth1 = price_change.ewm(span=long_period).mean()
        pc_smooth2 = pc_smooth1.ewm(span=short_period).mean()
        
        # Double smoothed absolute price change
        apc_smooth1 = abs_price_change.ewm(span=long_period).mean()
        apc_smooth2 = apc_smooth1.ewm(span=short_period).mean()
        
        # TSI calculation
        tsi = 100 * (pc_smooth2 / apc_smooth2)
        
        return tsi.fillna(0)
    
    def calculate_heikin_ashi(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Calculate Heikin Ashi candlestick values"""
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Initialize arrays
        ha_open = np.zeros(len(df))
        ha_high = np.zeros(len(df))
        ha_low = np.zeros(len(df))
        
        # First candle
        ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        # Calculate Heikin Ashi values
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        
        # Heikin Ashi High and Low
        ha_high = np.maximum(df['high'], np.maximum(ha_open, ha_close))
        ha_low = np.minimum(df['low'], np.minimum(ha_open, ha_close))
        
        # Return requested component
        if 'Open' in name:
            return pd.Series(ha_open, index=df.index, name=name)
        elif 'High' in name:
            return pd.Series(ha_high, index=df.index, name=name)
        elif 'Low' in name:
            return pd.Series(ha_low, index=df.index, name=name)
        else:  # Close
            return pd.Series(ha_close, index=df.index, name=name)
    
    def calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional engineered features"""
        result_df = df.copy()
        
        # Returns features
        result_df['returns_1'] = df['close'].pct_change(1)
        result_df['returns_5'] = df['close'].pct_change(5)
        
        # Volatility features
        result_df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
        
        # Volume features
        avg_volume_20 = df['volume'].rolling(window=20).mean()
        result_df['volume_ratio'] = df['volume'] / avg_volume_20
        
        # Price z-score
        price_mean_20 = df['close'].rolling(window=20).mean()
        price_std_20 = df['close'].rolling(window=20).std()
        result_df['price_zscore_20'] = (df['close'] - price_mean_20) / price_std_20
        
        # Fill NaN values
        for col in ['returns_1', 'returns_5', 'volatility_10', 'volume_ratio', 'price_zscore_20']:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    def calculate_all_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all enhanced indicators"""
        result_df = df.copy()
        
        try:
            # Calculate each registered indicator
            for indicator_name, info in self.indicator_registry.items():
                try:
                    function_name = info.get('function')
                    
                    if function_name == 'calculate_donchian':
                        result_df[indicator_name] = self.calculate_donchian(df, indicator_name)
                    elif function_name == 'calculate_obv':
                        result_df[indicator_name] = self.calculate_obv(df, indicator_name)
                    elif function_name == 'calculate_cmf':
                        result_df[indicator_name] = self.calculate_cmf(df, indicator_name)
                    elif function_name == 'calculate_ad_line':
                        result_df[indicator_name] = self.calculate_ad_line(df, indicator_name)
                    elif function_name == 'calculate_force_index':
                        result_df[indicator_name] = self.calculate_force_index(df, indicator_name)
                    elif function_name == 'calculate_chaikin_volatility':
                        result_df[indicator_name] = self.calculate_chaikin_volatility(df, indicator_name)
                    elif function_name == 'calculate_qstick':
                        result_df[indicator_name] = self.calculate_qstick(df, indicator_name)
                    elif function_name == 'calculate_kst':
                        result_df[indicator_name] = self.calculate_kst(df, indicator_name)
                    elif function_name == 'calculate_tsi':
                        result_df[indicator_name] = self.calculate_tsi(df, indicator_name)
                    elif function_name == 'calculate_heikin_ashi':
                        result_df[indicator_name] = self.calculate_heikin_ashi(df, indicator_name)
                    
                    self.logger.debug(f"Successfully calculated {indicator_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {indicator_name}: {e}")
                    # Fill with NaN instead of failing completely
                    result_df[indicator_name] = np.nan
            
            # Calculate additional engineered features
            result_df = self.calculate_additional_features(result_df)
            
            self.logger.info(f"Enhanced indicators calculated. Total features: {len(result_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced indicators: {e}")
        
        return result_df
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate feature statistics"""
        stats = {}
        
        for column in df.columns:
            if column not in ['open', 'high', 'low', 'close', 'volume']:  # Skip OHLCV columns
                series = df[column]
                stats[column] = {
                    'missing_ratio': series.isna().sum() / len(series),
                    'dtype': str(series.dtype),
                    'variance': float(series.var()) if series.dtype in ['float64', 'int64'] else 0,
                    'min': float(series.min()) if series.dtype in ['float64', 'int64'] else None,
                    'max': float(series.max()) if series.dtype in ['float64', 'int64'] else None,
                    'mean': float(series.mean()) if series.dtype in ['float64', 'int64'] else None,
                    'std': float(series.std()) if series.dtype in ['float64', 'int64'] else None
                }
        
        return stats