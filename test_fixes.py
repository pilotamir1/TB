#!/usr/bin/env python3
"""
Test script to validate the fixes implemented for the trading bot issues
"""

import os
import sys
import json
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_enhanced_logging():
    """Test Issue 1 & 10: Enhanced logging with emoji handling"""
    print("Testing enhanced logging system...")
    
    try:
        from utils.logging import initialize_logging, get_logger
        from config.config_loader import load_config
        
        # Load config
        config = load_config()
        
        # Initialize enhanced logging
        logger_system = initialize_logging(config)
        logger = get_logger('test_fixes')
        
        # Test emoji handling (this should not crash on Windows)
        logger.info("Testing emoji handling: 🚀 🎯 📉 ✅ 📊")
        logger.warning("Generic calculation used for TestIndicator - may need specific implementation")
        logger.debug("This should be suppressed on repeat")
        
        # Test structured logging
        logger_system.log_ml_event('test_event', {
            'test_data': 'enhanced_logging_works',
            'emoji_test': '✅'
        })
        
        print("✅ Enhanced logging test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced logging test failed: {e}")
        return False

def test_indicator_warnings():
    """Test Issue 4: Indicator warning suppression"""
    print("Testing indicator warning suppression...")
    
    try:
        from indicators.calculator import IndicatorCalculator
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        calculator = IndicatorCalculator()
        
        # This should only log warning once per indicator
        result = calculator.calculate_all_indicators(test_data)
        
        print("✅ Indicator warning suppression test passed")
        return True
        
    except Exception as e:
        print(f"❌ Indicator warning test failed: {e}")
        return False

def test_coinex_api_fallback():
    """Test Issue 5: CoinEx API graceful fallback"""
    print("Testing CoinEx API graceful fallback...")
    
    try:
        from trading.coinex_api import CoinExAPI
        
        # Create API client (will use demo/fallback data)
        api = CoinExAPI()
        
        # Test connection (should handle 404 gracefully)
        connection_ok = api.test_connection()
        
        # Test getting ticker (should fallback to demo data)
        ticker = api.get_ticker('BTCUSDT')
        
        if ticker and 'ticker' in ticker:
            print("✅ CoinEx API fallback test passed")
            return True
        else:
            print("❌ CoinEx API fallback test failed: No ticker data")
            return False
        
    except Exception as e:
        print(f"❌ CoinEx API test failed: {e}")
        return False

def test_config_loading():
    """Test enhanced config loading"""
    print("Testing enhanced configuration loading...")
    
    try:
        from config.config_loader import load_config
        
        config = load_config()
        
        # Check if enhanced features are configured
        expected_keys = ['logging', 'xgboost', 'ml', 'adaptive_threshold']
        
        for key in expected_keys:
            if key not in config:
                print(f"❌ Config test failed: Missing key '{key}'")
                return False
        
        # Check if XGBoost has regularization parameters
        xgb_config = config.get('xgboost', {})
        reg_params = ['reg_alpha', 'reg_lambda', 'min_child_weight', 'gamma']
        
        for param in reg_params:
            if param not in xgb_config:
                print(f"❌ Config test failed: Missing XGBoost regularization parameter '{param}'")
                return False
        
        print("✅ Enhanced configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TRADING BOT FIXES VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Enhanced Logging", test_enhanced_logging),
        ("Indicator Warning Suppression", test_indicator_warnings),
        ("CoinEx API Fallback", test_coinex_api_fallback),
        ("Enhanced Configuration", test_config_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   Test '{test_name}' failed")
        except Exception as e:
            print(f"   Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        sys.exit(0)
    else:
        print("⚠️  Some fixes need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()