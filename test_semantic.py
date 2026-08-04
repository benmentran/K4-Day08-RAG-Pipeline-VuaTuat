#!/usr/bin/env python3

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing task5_semantic_search import...")

# Check if we can import the semantic_search function
try:
    from src.task5_semantic_search import semantic_search
    print("✅ Successfully imported semantic_search from src.task5_semantic_search")
except Exception as e:
    print(f"❌ Error importing semantic_search: {e}")

# Now try to run a simple test
print("\nTesting semantic_search function...")

try:
    results = semantic_search("payment methods", top_k=3)
    print(f"✅ semantic_search executed successfully, returned {len(results)} results")
    
    if results:
        print("Sample results:")
        for i, result in enumerate(results[:2]):
            print(f"  Result {i+1}:")
            print(f"    Content: {result['content'][:100]}...")
            print(f"    Score: {result['score']}")
            print(f"    Metadata: {result['metadata']}")
    
except Exception as e:
    print(f"❌ Error calling semantic_search: {e}")
    import traceback
    traceback.print_exc()