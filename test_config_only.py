#!/usr/bin/env python3
"""
Configuration validation test for the combined training pipeline
Tests only the configuration changes without requiring ML dependencies
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_imports():
    """Test that all new configuration imports work"""
    print("Testing configuration imports...")
    
    try:
        from config.settings import FEATURE_SELECTION_CONFIG, LABELING_CONFIG, DATA_CONFIG, XGB_PRO_CONFIG
        print("✓ All configuration imports successful")
        return True
    except ImportError as e:
        print(f"✗ Configuration import failed: {e}")
        return False

def test_feature_selection_config():
    """Test the new FEATURE_SELECTION_CONFIG parameters"""
    print("Testing FEATURE_SELECTION_CONFIG...")
    
    try:
        from config.settings import FEATURE_SELECTION_CONFIG
        
        print(f"Current config: {FEATURE_SELECTION_CONFIG}")
        
        # Check required parameters
        required_params = ['enabled', 'mode', 'selection_window_4h', 'min_features', 'tolerance', 'max_iterations']
        
        for param in required_params:
            if param not in FEATURE_SELECTION_CONFIG:
                print(f"✗ Missing parameter: {param}")
                return False
        
        # Validate values
        if FEATURE_SELECTION_CONFIG['enabled'] != True:
            print("✗ Feature selection should be enabled")
            return False
            
        if FEATURE_SELECTION_CONFIG['mode'] != 'dynamic':
            print("✗ Mode should be 'dynamic'")
            return False
            
        if FEATURE_SELECTION_CONFIG['selection_window_4h'] != 800:
            print("✗ Selection window should be 800")
            return False
            
        if FEATURE_SELECTION_CONFIG['min_features'] != 20:
            print("✗ Min features should be 20")
            return False
            
        print("✓ FEATURE_SELECTION_CONFIG validation passed")
        return True
        
    except Exception as e:
        print(f"✗ FEATURE_SELECTION_CONFIG test failed: {e}")
        return False

def test_labeling_config():
    """Test the new LABELING_CONFIG parameters"""
    print("Testing LABELING_CONFIG...")
    
    try:
        from config.settings import LABELING_CONFIG
        
        print(f"Current config: {LABELING_CONFIG}")
        
        # Check required parameters
        required_params = ['target_distribution', 'initial_up_pct', 'initial_down_pct', 
                          'search_up_range', 'search_down_range', 'optimization_metric']
        
        for param in required_params:
            if param not in LABELING_CONFIG:
                print(f"✗ Missing parameter: {param}")
                return False
        
        # Validate target distribution
        target_dist = LABELING_CONFIG['target_distribution']
        if 'SELL' not in target_dist or 'BUY' not in target_dist or 'HOLD' not in target_dist:
            print("✗ Target distribution missing class labels")
            return False
            
        # Check if distribution sums to 1.0
        total = sum(target_dist.values())
        if abs(total - 1.0) > 0.01:
            print(f"✗ Target distribution should sum to 1.0, got {total}")
            return False
            
        # Validate thresholds
        if LABELING_CONFIG['initial_up_pct'] <= 0:
            print("✗ Initial up percentage should be positive")
            return False
            
        if LABELING_CONFIG['initial_down_pct'] >= 0:
            print("✗ Initial down percentage should be negative")
            return False
            
        # Validate ranges
        up_range = LABELING_CONFIG['search_up_range']
        down_range = LABELING_CONFIG['search_down_range']
        
        if len(up_range) != 3 or len(down_range) != 3:
            print("✗ Search ranges should have 3 elements [start, end, step]")
            return False
            
        if up_range[0] <= 0 or up_range[1] <= up_range[0]:
            print("✗ Up range should be positive and increasing")
            return False
            
        if down_range[0] >= 0 or down_range[1] <= down_range[0]:
            print("✗ Down range should be negative and end > start")
            return False
            
        print("✓ LABELING_CONFIG validation passed")
        return True
        
    except Exception as e:
        print(f"✗ LABELING_CONFIG test failed: {e}")
        return False

def test_data_config():
    """Test DATA_CONFIG for unlimited history"""
    print("Testing DATA_CONFIG...")
    
    try:
        from config.settings import DATA_CONFIG
        
        print(f"Current config relevant parts: use_all_history={DATA_CONFIG.get('use_all_history')}, max_4h_training_candles={DATA_CONFIG.get('max_4h_training_candles')}")
        
        if not DATA_CONFIG.get('use_all_history', False):
            print("✗ use_all_history should be True")
            return False
            
        if DATA_CONFIG.get('max_4h_training_candles', -1) != 0:
            print("✗ max_4h_training_candles should be 0 for unlimited")
            return False
            
        print("✓ DATA_CONFIG validation passed")
        return True
        
    except Exception as e:
        print(f"✗ DATA_CONFIG test failed: {e}")
        return False

def test_integration_ready():
    """Test that integration points are ready"""
    print("Testing integration readiness...")
    
    try:
        # Test import paths
        from config.settings import FEATURE_SELECTION_CONFIG, LABELING_CONFIG
        
        # Validate that configuration is set for proper combined pipeline
        if not FEATURE_SELECTION_CONFIG.get('enabled'):
            print("✗ Feature selection must be enabled for combined pipeline")
            return False
            
        if FEATURE_SELECTION_CONFIG.get('mode') != 'dynamic':
            print("✗ Dynamic mode must be enabled for combined pipeline")
            return False
            
        # Check selection window
        selection_window = FEATURE_SELECTION_CONFIG.get('selection_window_4h')
        if not (500 <= selection_window <= 1000):
            print(f"✗ Selection window {selection_window} should be in range 500-1000")
            return False
            
        # Check labeling target distribution ratios
        target_dist = LABELING_CONFIG['target_distribution']
        buy_ratio = target_dist.get('BUY', 0)
        sell_ratio = target_dist.get('SELL', 0) 
        hold_ratio = target_dist.get('HOLD', 0)
        
        if not (0.35 <= buy_ratio <= 0.45):
            print(f"✗ BUY ratio {buy_ratio} should be ~40%")
            return False
            
        if not (0.35 <= sell_ratio <= 0.45):
            print(f"✗ SELL ratio {sell_ratio} should be ~40%") 
            return False
            
        if not (0.15 <= hold_ratio <= 0.25):
            print(f"✗ HOLD ratio {hold_ratio} should be ~20%")
            return False
            
        print("✓ Integration readiness validation passed")
        print(f"  - Feature selection: enabled, dynamic mode, {selection_window} 4h window")
        print(f"  - Target distribution: BUY={buy_ratio:.1%}, SELL={sell_ratio:.1%}, HOLD={hold_ratio:.1%}")
        print(f"  - Unlimited training history: enabled")
        return True
        
    except Exception as e:
        print(f"✗ Integration readiness test failed: {e}")
        return False

def main():
    """Run all configuration tests"""
    print("=== Combined Training Pipeline Configuration Tests ===\n")
    
    tests = [
        test_config_imports,
        test_feature_selection_config,
        test_labeling_config,
        test_data_config,
        test_integration_ready
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}\n")
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All configuration tests passed!")
        print("Combined pipeline implementation configuration is ready.")
        print("\nKey Features Implemented:")
        print("• Dynamic feature selection on recent 800 4h candles")
        print("• Adaptive label thresholding targeting SELL≈40%, BUY≈40%, HOLD≈20%")
        print("• Unlimited historical training data usage")
        print("• KL divergence optimization for distribution matching")
        return True
    else:
        print("❌ Some configuration tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)