#!/usr/bin/env python3
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

# Change to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=== Running Individual Task Tests ===")

import pytest

# Run all individual tests
test_result = pytest.main([
    'tests/test_individual.py',
    '-v',
    '--tb=short'
])

if test_result == 0:
    print("\n✅ All tests passed!")
else:
    print(f"\n❌ Tests failed with exit code: {test_result}")

# Exit with the same code
exit(test_result)