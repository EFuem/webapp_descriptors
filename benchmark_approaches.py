#!/usr/bin/env python3
"""
Benchmark script to compare different optimization approaches
"""

import time
import psutil
import os
from typing import Callable, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_function(func: Callable, name: str, *args, **kwargs) -> dict:
    """Benchmark a function and return performance metrics"""
    logger.info(f"Benchmarking {name}...")
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Get initial CPU usage
    initial_cpu = process.cpu_percent()
    
    # Time the function
    start_time = time.time()
    start_cpu = time.time()
    
    try:
        result = func(*args, **kwargs)
        success = True
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        result = None
        success = False
    
    end_time = time.time()
    end_cpu = time.time()
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory
    
    # Estimate CPU usage (rough approximation)
    cpu_time = end_cpu - start_cpu
    
    metrics = {
        'name': name,
        'success': success,
        'execution_time': execution_time,
        'memory_used_mb': memory_used,
        'final_memory_mb': final_memory,
        'cpu_time': cpu_time,
        'result_count': len(result) if result else 0
    }
    
    if success and result:
        metrics['rate_structures_per_sec'] = len(result) / execution_time if execution_time > 0 else 0
    
    logger.info(f"{name} completed in {execution_time:.2f}s, "
                f"Memory: {memory_used:+.1f}MB, "
                f"Rate: {metrics.get('rate_structures_per_sec', 0):.2f} structures/sec")
    
    return metrics

def run_benchmarks():
    """Run all benchmarks and compare results"""
    logger.info("Starting benchmark comparison...")
    
    # Import the different approaches
    try:
        from db_interface import main as original_main
        from db_interface_optimized import main_optimized, main_hybrid
        from db_interface_advanced import main_advanced, main_simple_parallel
        
        approaches = [
            (original_main, "Original Sequential"),
            (main_hybrid, "Optimized Hybrid"),
            (main_simple_parallel, "Simple Parallel"),
            (main_advanced, "Advanced Parallel")
        ]
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure all optimization scripts are in the same directory")
        return
    
    # Run benchmarks
    results = []
    for func, name in approaches:
        try:
            metrics = benchmark_function(func, name)
            results.append(metrics)
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    if not results:
        print("No successful benchmarks to compare")
        return
    
    # Find best performing approach
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_rate = max(successful_results, key=lambda x: x.get('rate_structures_per_sec', 0))
        best_time = min(successful_results, key=lambda x: x['execution_time'])
        best_memory = min(successful_results, key=lambda x: x['memory_used_mb'])
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   Highest rate: {best_rate['name']} ({best_rate.get('rate_structures_per_sec', 0):.2f} structures/sec)")
        print(f"   Fastest time: {best_time['name']} ({best_time['execution_time']:.2f}s)")
        print(f"   Lowest memory: {best_memory['name']} ({best_memory['memory_used_mb']:+.1f}MB)")
    
    print(f"\n📊 DETAILED COMPARISON:")
    print(f"{'Approach':<25} {'Time(s)':<10} {'Rate(/s)':<12} {'Memory(MB)':<12} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        rate = result.get('rate_structures_per_sec', 0)
        memory = result.get('memory_used_mb', 0)
        
        print(f"{result['name']:<25} {result['execution_time']:<10.2f} "
              f"{rate:<12.2f} {memory:<12.1f} {status:<10}")
    
    # Performance improvement analysis
    if len(successful_results) > 1:
        print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
        baseline = successful_results[0]
        for result in successful_results[1:]:
            if result['execution_time'] > 0 and baseline['execution_time'] > 0:
                speedup = baseline['execution_time'] / result['execution_time']
                print(f"   {result['name']} is {speedup:.2f}x faster than {baseline['name']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_benchmarks()
