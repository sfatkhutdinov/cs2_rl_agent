#!/usr/bin/env python
"""
Test runner for CS2 RL Agent tests.
Runs all tests in the testing directory.
"""

import os
import sys
import unittest
import argparse
import logging

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests(test_pattern=None, verbose=False):
    """
    Run all tests in the testing directory.
    
    Args:
        test_pattern (str, optional): Pattern to match test files. Defaults to None.
        verbose (bool, optional): Whether to show verbose output. Defaults to False.
    
    Returns:
        bool: True if all tests passed, False otherwise.
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Discover and run tests
    loader = unittest.TestLoader()
    
    if test_pattern:
        test_suite = loader.discover(os.path.dirname(__file__), pattern=test_pattern)
    else:
        test_suite = loader.discover(os.path.dirname(__file__))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CS2 RL Agent tests")
    parser.add_argument(
        "--pattern", 
        type=str, 
        default=None, 
        help="Pattern to match test files (e.g., 'test_*.py')"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    success = run_tests(args.pattern, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 