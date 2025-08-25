#!/usr/bin/env python3
"""
Test script to verify optimization functions work correctly
"""

import time
import logging
from typing import List, Dict, Any
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_structures: int = 10) -> List[Dict[str, Any]]:
    """Create test data for optimization testing"""
    test_data = []
    
    for i in range(n_structures):
        # Create simple test structure
        test_structure = {
            'positions': '[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]',
            'atomic_numbers': '[1, 1]',  # Hydrogen atoms
            'cell': '[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]',
            'pbc': '[True, True, True]'
        }
        test_data.append(test_structure)
    
    return test_data

def test_basic_processing():
    """Test basic processing functions"""
    logger.info("Testing basic processing functions...")
    
    try:
        from db_interface_optimized import process_atom_batch
        
        test_data = create_test_data(5)
        logger.info(f"Created {len(test_data)} test structures")
        
        # Test processing
        start_time = time.time()
        results = process_atom_batch(test_data)
        processing_time = time.time() - start_time
        
        logger.info(f"Processing completed in {processing_time:.3f}s")
        logger.info(f"Generated {len(results)} embeddings")
        
        # Check results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Valid embeddings: {len(valid_results)}/{len(results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic processing test failed: {e}")
        return False

def test_parallel_processing():
    """Test parallel processing functions"""
    logger.info("Testing parallel processing functions...")
    
    try:
        from db_interface_optimized import process_batch_parallel
        
        test_data = create_test_data(20)
        logger.info(f"Created {len(test_data)} test structures")
        
        # Test parallel processing
        start_time = time.time()
        results = process_batch_parallel(test_data, n_workers=2)
        processing_time = time.time() - start_time
        
        logger.info(f"Parallel processing completed in {processing_time:.3f}s")
        logger.info(f"Generated {len(results)} embeddings")
        
        # Check results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Valid embeddings: {len(valid_results)}/{len(results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Parallel processing test failed: {e}")
        return False

def test_advanced_processing():
    """Test advanced processing functions"""
    logger.info("Testing advanced processing functions...")
    
    try:
        from db_interface_advanced import process_atom_batch_optimized
        
        test_data = create_test_data(15)
        logger.info(f"Created {len(test_data)} test structures")
        
        # Test advanced processing
        start_time = time.time()
        results = process_atom_batch_optimized(test_data, batch_size=8)
        processing_time = time.time() - start_time
        
        logger.info(f"Advanced processing completed in {processing_time:.3f}s")
        logger.info(f"Generated {len(results)} embeddings")
        
        # Check results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Valid embeddings: {len(valid_results)}/{len(results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Advanced processing test failed: {e}")
        return False

def run_all_tests():
    """Run all optimization tests"""
    logger.info("Starting optimization tests...")
    
    tests = [
        ("Basic Processing", test_basic_processing),
        ("Parallel Processing", test_parallel_processing),
        ("Advanced Processing", test_advanced_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "💥 CRASHED"
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25} {result}")
    
    # Overall status
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Optimizations are working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
