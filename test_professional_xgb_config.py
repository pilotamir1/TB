#!/usr/bin/env python3
"""
Test script to validate professional XGBoost configuration changes
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/runner/work/TB/TB')

def test_config_imports():
    """Test that all new configurations can be imported"""
    print("=== Testing Configuration Imports ===")
    
    try:
        from config.settings import DATA_CONFIG, FEATURE_SELECTION_CONFIG, XGB_PRO_CONFIG
        
        print("✓ All configs imported successfully")
        
        # Test DATA_CONFIG
        print(f"DATA_CONFIG.use_all_history: {DATA_CONFIG.get('use_all_history')}")
        print(f"DATA_CONFIG.max_4h_training_candles: {DATA_CONFIG.get('max_4h_training_candles')}")
        
        # Test FEATURE_SELECTION_CONFIG
        print(f"FEATURE_SELECTION_CONFIG.enabled: {FEATURE_SELECTION_CONFIG.get('enabled')}")
        
        # Test XGB_PRO_CONFIG
        print(f"XGB_PRO_CONFIG.n_estimators: {XGB_PRO_CONFIG.get('n_estimators')}")
        print(f"XGB_PRO_CONFIG.early_stopping_rounds: {XGB_PRO_CONFIG.get('early_stopping_rounds')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

def test_model_initialization():
    """Test that TradingModel can be initialized with professional XGBoost"""
    print("\n=== Testing Model Initialization ===")
    
    try:
        from ml.model import TradingModel
        from config.settings import XGB_PRO_CONFIG
        
        # Test model initialization
        model = TradingModel(model_type='xgboost_professional')
        print("✓ Professional XGBoost model initialized successfully")
        
        # Check if the model has the expected parameters
        if hasattr(model.model, 'n_estimators'):
            expected_estimators = XGB_PRO_CONFIG.get('n_estimators', 8000)
            actual_estimators = model.model.n_estimators
            print(f"Model n_estimators: {actual_estimators} (expected: {expected_estimators})")
            
            if actual_estimators == expected_estimators:
                print("✓ Model configured with correct number of estimators")
            else:
                print("✗ Model estimators don't match configuration")
                return False
        
        # Check other key parameters
        expected_max_depth = XGB_PRO_CONFIG.get('max_depth', 12)
        actual_max_depth = model.model.max_depth
        print(f"Model max_depth: {actual_max_depth} (expected: {expected_max_depth})")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_trainer_imports():
    """Test that ModelTrainer can import all required configs"""
    print("\n=== Testing Trainer Configuration Imports ===")
    
    try:
        from ml.trainer import ModelTrainer
        
        trainer = ModelTrainer()
        print("✓ ModelTrainer initialized successfully")
        
        # Check if trainer has access to new configs
        from config.settings import FEATURE_SELECTION_CONFIG, XGB_PRO_CONFIG
        
        print(f"Feature selection enabled: {FEATURE_SELECTION_CONFIG.get('enabled')}")
        print(f"XGBoost trees: {XGB_PRO_CONFIG.get('n_estimators')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all configuration tests"""
    print("Professional XGBoost Configuration Test Suite")
    print("=" * 50)
    
    tests = [
        test_config_imports,
        test_model_initialization, 
        test_trainer_imports
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)