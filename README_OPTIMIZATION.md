# Database Processing Optimization Guide

This guide explains how to speed up your database processing and MACE embedding generation using various optimization techniques.

## 🚀 Performance Improvements

The original sequential processing can be significantly improved using:

1. **Multiprocessing** - Parallel CPU processing
2. **Batch Processing** - Optimized MACE model inference
3. **Memory Management** - Controlled memory usage
4. **Async Operations** - Non-blocking database operations

## 📁 Files Overview

- `db_interface.py` - Original sequential implementation
- `db_interface_optimized.py` - Basic parallel processing
- `db_interface_advanced.py` - Advanced optimization with memory management
- `benchmark_approaches.py` - Performance comparison tool
- `requirements_optimization.txt` - Additional dependencies

## 🔧 Installation

Install the required dependencies:

```bash
pip install -r requirements_optimization.txt
```

## 🎯 Optimization Strategies

### 1. Basic Parallel Processing (`db_interface_optimized.py`)

**Best for**: Simple speedup without complex memory management

**Features**:
- Uses `ProcessPoolExecutor` for parallel processing
- Configurable number of workers
- Two approaches: `main_optimized()` and `main_hybrid()`

**Usage**:
```python
from db_interface_optimized import main_hybrid

# Use hybrid approach (balanced performance/memory)
embeddings = main_hybrid()
```

### 2. Advanced Optimization (`db_interface_advanced.py`)

**Best for**: Maximum performance with large datasets

**Features**:
- Producer-consumer pattern with queues
- Memory monitoring and cleanup
- Controlled concurrency
- Fallback to simpler approach if needed

**Usage**:
```python
from db_interface_advanced import main_advanced

# Use advanced processing
embeddings = main_advanced()
```

## ⚡ Performance Tuning

### Worker Count Optimization

```python
# For CPU-intensive tasks
n_workers = min(mp.cpu_count(), 8)

# For GPU-intensive tasks (MACE model)
n_workers = min(4, mp.cpu_count() // 2)
```

### Batch Size Tuning

```python
# Optimal batch size for MACE model
batch_size = 32  # Adjust based on your GPU memory
```

### Memory Management

```python
# Limit queue size to prevent memory buildup
max_queue_size = n_workers * 4

# Regular garbage collection
if completed_batches % 20 == 0:
    gc.collect()
```

## 📊 Benchmarking

Run the benchmark script to compare all approaches:

```bash
python benchmark_approaches.py
```

This will:
- Test all optimization strategies
- Measure execution time, memory usage, and throughput
- Provide performance comparison table
- Identify the best approach for your use case

## 🎛️ Configuration Options

### Database Configuration

```python
config = QueryConfig(
    limit_rows_per_sub_split=10_000,    # Rows per sub-split
    rows_per_split=1_000_000,           # Total rows per split
    num_sub_splits=10,                  # Number of sub-splits
)
```

### Processing Configuration

```python
# Conservative settings (recommended for most cases)
n_workers = min(4, mp.cpu_count() // 2)
batch_size = 32
max_queue_size = n_workers * 4

# Aggressive settings (for high-performance systems)
n_workers = min(8, mp.cpu_count())
batch_size = 64
max_queue_size = n_workers * 8
```

## 🔍 Monitoring and Debugging

### Memory Usage Monitoring

```python
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Log memory usage
logger.info(f"Memory usage: {get_memory_usage():.1f} MB")
```

### Progress Tracking

```python
# Progress updates every N batches
if batch_count % 10 == 0:
    elapsed = time.time() - start_time
    rate = len(all_embeddings) / elapsed
    logger.info(f"Rate: {rate:.2f} structures/sec")
```

## 🚨 Common Issues and Solutions

### 1. Memory Issues

**Problem**: Out of memory errors
**Solution**: Reduce `n_workers` or `batch_size`

```python
# More conservative settings
n_workers = 2
batch_size = 16
```

### 2. GPU Memory Issues

**Problem**: CUDA out of memory
**Solution**: Reduce batch size and worker count

```python
# GPU-friendly settings
n_workers = 2  # Fewer workers to avoid GPU contention
batch_size = 16  # Smaller batches
```

### 3. Database Connection Issues

**Problem**: Connection timeouts
**Solution**: Add retry logic and connection pooling

```python
# Add retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        # Database operation
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)  # Exponential backoff
```

## 📈 Expected Performance Improvements

Based on typical configurations:

| Approach | Speedup | Memory Usage | Complexity |
|----------|---------|--------------|------------|
| Original | 1x | Low | Low |
| Basic Parallel | 2-4x | Medium | Low |
| Advanced | 4-8x | High | High |

**Note**: Actual performance depends on:
- CPU cores available
- GPU memory and performance
- Database connection speed
- Dataset size and complexity

## 🎯 Recommendations

### For Small Datasets (< 10k structures)
- Use `main_hybrid()` from `db_interface_optimized.py`
- Simple and reliable

### For Medium Datasets (10k - 100k structures)
- Use `main_simple_parallel()` from `db_interface_advanced.py`
- Good balance of performance and reliability

### For Large Datasets (> 100k structures)
- Use `main_advanced()` from `db_interface_advanced.py`
- Maximum performance with memory management

### For Production Systems
- Start with conservative settings
- Monitor memory and CPU usage
- Gradually increase workers/batch size
- Implement proper error handling and logging

## 🔧 Customization

### Adding Custom Processing Logic

```python
def custom_processing_function(atoms_list):
    """Custom processing logic"""
    results = []
    for atoms in atoms_list:
        # Your custom logic here
        result = process_atoms(atoms)
        results.append(result)
    return results

# Use in worker process
def worker_process(batch_queue, result_queue, worker_id):
    while True:
        batch_data = batch_queue.get()
        if batch_data is None:
            break
        
        # Use custom processing
        results = custom_processing_function(batch_data)
        result_queue.put(results)
```

### Custom Error Handling

```python
def robust_processing(batch_data):
    """Robust processing with error handling"""
    results = []
    for item in batch_data:
        try:
            result = process_item(item)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process item: {e}")
            results.append(None)  # Placeholder for failed items
    return results
```

## 📚 Additional Resources

- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)
- [MACE Model Documentation](https://github.com/ACEsuit/mace)
- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/)

## 🤝 Contributing

To improve these optimizations:

1. Test with different dataset sizes
2. Benchmark on different hardware configurations
3. Add new optimization strategies
4. Improve error handling and monitoring
5. Optimize for specific use cases

## 📄 License

This optimization code follows the same license as your original project.
