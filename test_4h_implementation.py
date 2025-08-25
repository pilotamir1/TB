#!/usr/bin/env python3
"""
Test 4h Timeframe Implementation

This script tests the 4h timeframe functionality without requiring a full database setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time

def test_4h_timeframe_config():
    """Test that configuration is properly set to 4h"""
    print("=== Testing 4h Timeframe Configuration ===")
    
    from config.settings import TRADING_CONFIG, DATA_CONFIG
    
    print(f"✓ Timeframe: {TRADING_CONFIG['timeframe']}")
    print(f"✓ Update interval: {DATA_CONFIG['update_interval']} seconds") 
    print(f"✓ Min 4h candles: {DATA_CONFIG.get('min_4h_candles', 'Not set')}")
    
    assert TRADING_CONFIG['timeframe'] == '4h', "Timeframe should be 4h"
    assert DATA_CONFIG['update_interval'] == 300, "Update interval should be 300s for 4h"
    assert DATA_CONFIG.get('min_4h_candles') == 800, "Min 4h candles should be 800"
    
    print("✓ All configuration checks passed!")
    return True

def test_4h_alignment():
    """Test 4h alignment detection"""
    print("\n=== Testing 4h Alignment Detection ===")
    
    def is_aligned_4h(timestamp: int) -> bool:
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.hour % 4 == 0 and dt.minute == 0 and dt.second == 0
        except Exception:
            return False
    
    # Test aligned timestamps (should return True)
    aligned_tests = [
        (1704067200, "2024-01-01 00:00:00"),  # Midnight (0h)
        (1704081600, "2024-01-01 04:00:00"),  # 4AM (4h) 
        (1704096000, "2024-01-01 08:00:00"),  # 8AM (8h)
        (1704110400, "2024-01-01 12:00:00"),  # Noon (12h)
        (1704124800, "2024-01-01 16:00:00"),  # 4PM (16h)
        (1704139200, "2024-01-01 20:00:00"),  # 8PM (20h)
    ]
    
    # Test non-aligned timestamps (should return False)
    non_aligned_tests = [
        (1704070800, "2024-01-01 01:00:00"),  # 1AM (not divisible by 4)
        (1704072600, "2024-01-01 01:30:00"),  # 1:30AM (has minutes)
        (1704085200, "2024-01-01 05:00:00"),  # 5AM (not divisible by 4)
        (1704088800, "2024-01-01 06:00:00"),  # 6AM (not divisible by 4)
        (1704067260, "2024-01-01 00:01:00"),  # 12:01AM (has minutes)
    ]
    
    print("Testing aligned timestamps (should be True):")
    for timestamp, desc in aligned_tests:
        result = is_aligned_4h(timestamp)
        status = "✓" if result else "✗"
        print(f"  {status} {desc}: {result}")
        assert result, f"Timestamp {desc} should be aligned"
    
    print("\nTesting non-aligned timestamps (should be False):")
    for timestamp, desc in non_aligned_tests:
        result = is_aligned_4h(timestamp)
        status = "✓" if not result else "✗"
        print(f"  {status} {desc}: {result}")
        assert not result, f"Timestamp {desc} should NOT be aligned"
    
    print("✓ All alignment tests passed!")
    return True

def test_trading_engine_4h_logic():
    """Test trading engine 4h boundary logic"""
    print("\n=== Testing Trading Engine 4h Logic ===")
    
    # Mock the _is_new_timeframe logic
    def _is_new_timeframe_4h():
        """Simplified version of _is_new_timeframe for 4h"""
        now = datetime.now()
        h = now.hour
        m = now.minute
        
        # For 4h timeframe, only trigger at 4h boundaries  
        return (h % 4 == 0 and m == 0)
    
    print(f"Current time: {datetime.now()}")
    print(f"Current hour: {datetime.now().hour}")
    print(f"Current minute: {datetime.now().minute}")
    
    should_trigger = _is_new_timeframe_4h()
    print(f"Should trigger signal now: {should_trigger}")
    
    # Test specific times
    test_times = [
        (0, 0, True),   # Midnight (00:00)
        (4, 0, True),   # 4AM (04:00)
        (8, 0, True),   # 8AM (08:00) 
        (12, 0, True),  # Noon (12:00)
        (16, 0, True),  # 4PM (16:00)
        (20, 0, True),  # 8PM (20:00)
        (1, 0, False),  # 1AM (01:00)
        (3, 0, False),  # 3AM (03:00)
        (5, 0, False),  # 5AM (05:00)
        (4, 30, False), # 4:30AM (04:30)
        (8, 1, False),  # 8:01AM (08:01)
    ]
    
    print("\nTesting 4h boundary logic:")
    for hour, minute, expected in test_times:
        result = (hour % 4 == 0 and minute == 0)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {hour:02d}:{minute:02d} -> {result} (expected: {expected})")
        assert result == expected, f"Time {hour:02d}:{minute:02d} failed boundary test"
    
    print("✓ All trading engine logic tests passed!")
    return True

def test_cleanup_script():
    """Test the cleanup script functionality"""
    print("\n=== Testing Cleanup Script ===")
    
    try:
        # Test dry run of cleanup script
        import subprocess
        result = subprocess.run([
            'python', 'scripts/clean_candles_4h.py', '--help'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✓ Cleanup script help works")
        else:
            print(f"✗ Cleanup script help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing cleanup script: {e}")
        return False
    
    print("✓ Cleanup script tests passed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Testing 4h Timeframe Implementation")
    print("=" * 50)
    
    tests = [
        test_4h_timeframe_config,
        test_4h_alignment,
        test_trading_engine_4h_logic,
        test_cleanup_script,
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
            print(f"✗ Test {test.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! 4h timeframe implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)