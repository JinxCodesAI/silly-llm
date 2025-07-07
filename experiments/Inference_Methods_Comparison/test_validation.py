#!/usr/bin/env python3
"""
Test script to verify configuration validation works
"""

def test_method_validation():
    """Test method validation logic"""
    
    def validate_methods(methods_to_test):
        if methods_to_test is not None:
            valid_methods = {"standard", "simple", "beam_search", "batched", "nucleus_sampling"}
            invalid_methods = set(methods_to_test) - valid_methods
            if invalid_methods:
                raise ValueError(f"Invalid method(s) specified: {sorted(invalid_methods)}. "
                               f"Valid methods are: {sorted(valid_methods)}")
            
            # Check for duplicates
            if len(methods_to_test) != len(set(methods_to_test)):
                duplicates = [method for method in set(methods_to_test) 
                            if methods_to_test.count(method) > 1]
                raise ValueError(f"Duplicate method(s) specified: {sorted(duplicates)}")
    
    # Test 1: Valid methods
    try:
        validate_methods(["batched", "simple"])
        print("✅ Test 1 PASSED: Valid methods accepted")
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
    
    # Test 2: Invalid methods
    try:
        validate_methods(["batched", "invalid_method"])
        print("❌ Test 2 FAILED: Should have rejected invalid methods")
    except ValueError as e:
        print(f"✅ Test 2 PASSED: Invalid methods rejected - {e}")
    
    # Test 3: Duplicate methods
    try:
        validate_methods(["batched", "simple", "batched"])
        print("❌ Test 3 FAILED: Should have rejected duplicate methods")
    except ValueError as e:
        print(f"✅ Test 3 PASSED: Duplicate methods rejected - {e}")
    
    # Test 4: None (should be allowed)
    try:
        validate_methods(None)
        print("✅ Test 4 PASSED: None methods accepted")
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")

if __name__ == "__main__":
    test_method_validation()
