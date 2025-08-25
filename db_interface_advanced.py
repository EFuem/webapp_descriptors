import vastdb
from vastdb.config import QueryConfig
import os
import sys
from ase import Atoms
from ast import literal_eval
from generate_descriptors import get_embeddings
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import asyncio
import numpy as np
from functools import partial
import time
from typing import List, Dict, Any, Tuple
import gc
import psutil
import threading
from queue import Queue
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_vastdb_session():
    endpoint = "http://10.32.38.210"
    with open(f"/home/{os.environ['USER']}/.vast-dev/access_key_id", "r") as f:
        access_key = f.read().rstrip("\n")
    with open(f"/home/{os.environ['USER']}/.vast-dev/secret_access_key", "r") as f:
        secret_key = f.read().rstrip("\n")
    return vastdb.connect(endpoint=endpoint, access=access_key, secret=secret_key)

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def process_atom_batch_optimized(batch_data: List[Dict[str, Any]], batch_size: int = 32) -> List[np.ndarray]:
    """Process a batch of atom structures with optimized batching for MACE"""
    from generate_descriptors import get_embeddings
    from ase import Atoms
    from ast import literal_eval
    
    embeddings = []
    
    # Process in smaller batches to optimize MACE model inference
    for i in range(0, len(batch_data), batch_size):
        sub_batch = batch_data[i:i + batch_size]
        
        # Convert to Atoms objects
        atoms_list = []
        for b in sub_batch:
            try:
                atoms = Atoms(
                    positions=literal_eval(b['positions']), 
                    numbers=literal_eval(b['atomic_numbers']), 
                    cell=literal_eval(b['cell']), 
                    pbc=literal_eval(b['pbc'])
                )
                atoms_list.append(atoms)
            except Exception as e:
                logger.warning(f"Error creating Atoms object: {e}")
                atoms_list.append(None)
        
        # Process valid atoms
        valid_atoms = [a for a in atoms_list if a is not None]
        if valid_atoms:
            try:
                # Process all valid atoms in this sub-batch
                batch_embeddings = process_mace_batch(valid_atoms)
                
                # Map back to original positions
                sub_embeddings = []
                atom_idx = 0
                for atoms in atoms_list:
                    if atoms is not None:
                        sub_embeddings.append(batch_embeddings[atom_idx])
                        atom_idx += 1
                    else:
                        sub_embeddings.append(None)
                
                embeddings.extend(sub_embeddings)
            except Exception as e:
                logger.error(f"Error processing MACE batch: {e}")
                embeddings.extend([None] * len(sub_batch))
        else:
            embeddings.extend([None] * len(sub_batch))
    
    return embeddings

def process_mace_batch(atoms_list: List[Atoms]) -> List[np.ndarray]:
    """Process multiple atoms structures through MACE in a single batch"""
    from generate_descriptors import get_embeddings
    
    # For now, process individually since MACE might not support true batching
    # In the future, you could modify the MACE calculator to support batch processing
    embeddings = []
    for atoms in atoms_list:
        try:
            e = get_embeddings(atoms)
            embeddings.append(e)
        except Exception as e:
            logger.warning(f"Error in MACE processing: {e}")
            embeddings.append(None)
    
    return embeddings

def worker_process(batch_queue: Queue, result_queue: Queue, worker_id: int, batch_size: int = 32):
    """Worker process for parallel processing"""
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get batch from queue
            batch_data = batch_queue.get(timeout=5)
            if batch_data is None:  # Sentinel value to stop
                break
            
            # Process batch
            batch_embeddings = process_atom_batch_optimized(batch_data, batch_size)
            
            # Put results in result queue
            result_queue.put((batch_data, batch_embeddings))
            
            # Clear batch data to free memory
            del batch_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            result_queue.put((None, [None] * len(batch_data) if 'batch_data' in locals() else []))
    
    logger.info(f"Worker {worker_id} finished")

def main_advanced():
    """Advanced processing with optimal memory management and parallel processing"""
    session = get_vastdb_session()
    
    config = QueryConfig(
        limit_rows_per_sub_split=10_000,
        rows_per_split=1_000_000,
        num_sub_splits=10,
    )
    dataset_id = "DS_h0mshvvbxlai_0"
    
    # Configuration
    n_workers = min(4, mp.cpu_count() // 2)  # Conservative worker count
    batch_size = 32  # Optimal batch size for MACE
    max_queue_size = n_workers * 4  # Prevent memory buildup
    
    logger.info(f"Using {n_workers} workers with batch size {batch_size}")
    logger.info(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Create queues for inter-process communication
    batch_queue = Queue(maxsize=max_queue_size)
    result_queue = Queue()
    
    # Start worker processes
    workers = []
    for i in range(n_workers):
        worker = mp.Process(
            target=worker_process, 
            args=(batch_queue, result_queue, i, batch_size)
        )
        worker.start()
        workers.append(worker)
    
    all_embeddings = []
    start_time = time.time()
    
    try:
        with session.transaction() as tx:
            co_table = (
                tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
            )
            co_data = co_table.select(
                predicate=co_table["dataset_id"] == dataset_id, config=config
            )
            
            # Producer: Feed batches to workers
            batch_count = 0
            for i, co_batch in enumerate(co_data):
                batch_list = co_batch.to_pylist()
                if len(batch_list) > 0:
                    # Wait if queue is full
                    while batch_queue.qsize() >= max_queue_size:
                        time.sleep(0.1)
                    
                    batch_queue.put(batch_list)
                    batch_count += 1
                    
                    # Progress update
                    if batch_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = len(all_embeddings) / elapsed if elapsed > 0 else 0
                        memory = get_memory_usage()
                        logger.info(f"Queued {batch_count} batches, Processed {len(all_embeddings)} structures, "
                                  f"Rate: {rate:.2f} structures/sec, Memory: {memory:.1f} MB")
            
            # Send sentinel values to stop workers
            for _ in range(n_workers):
                batch_queue.put(None)
            
            # Consumer: Collect results from workers
            completed_batches = 0
            with tqdm(total=batch_count, desc="Processing batches") as pbar:
                while completed_batches < batch_count:
                    try:
                        batch_data, batch_embeddings = result_queue.get(timeout=30)
                        if batch_data is not None:
                            all_embeddings.extend(batch_embeddings)
                            completed_batches += 1
                            pbar.update(1)
                            
                            # Memory cleanup
                            if completed_batches % 20 == 0:
                                gc.collect()
                                memory = get_memory_usage()
                                logger.info(f"Memory usage: {memory:.1f} MB")
                    except Exception as e:
                        logger.error(f"Error collecting results: {e}")
                        break
    
    finally:
        # Cleanup
        for worker in workers:
            worker.terminate()
            worker.join()
        
        # Clear queues
        while not batch_queue.empty():
            try:
                batch_queue.get_nowait()
            except:
                pass
        
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                pass
    
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Total structures processed: {len(all_embeddings)}")
    logger.info(f"Average rate: {len(all_embeddings) / total_time:.2f} structures/sec")
    logger.info(f"Final memory usage: {final_memory:.1f} MB")
    
    return all_embeddings

def main_simple_parallel():
    """Simpler parallel processing approach"""
    session = get_vastdb_session()
    
    config = QueryConfig(
        limit_rows_per_sub_split=10_000,
        rows_per_split=1_000_000,
        num_sub_splits=10,
    )
    dataset_id = "DS_h0mshvvbxlai_0"
    
    n_workers = min(4, mp.cpu_count() // 2)
    logger.info(f"Using {n_workers} workers for simple parallel processing")
    
    all_embeddings = []
    start_time = time.time()
    
    with session.transaction() as tx:
        co_table = (
            tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
        )
        co_data = co_table.select(
            predicate=co_table["dataset_id"] == dataset_id, config=config
        )
        
        # Collect all batches first
        all_batches = []
        for co_batch in co_data:
            batch_list = co_batch.to_pylist()
            if len(batch_list) > 0:
                all_batches.append(batch_list)
        
        logger.info(f"Collected {len(all_batches)} batches to process")
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_atom_batch_optimized, batch) for batch in all_batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error in batch: {e}")
                    # Estimate batch size for placeholder
                    all_embeddings.extend([None] * 100)
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Total structures processed: {len(all_embeddings)}")
    logger.info(f"Average rate: {len(all_embeddings) / total_time:.2f} structures/sec")
    
    return all_embeddings

if __name__ == "__main__":
    logger.info("Starting advanced processing...")
    
    # Choose your optimization strategy
    try:
        # Option 1: Advanced processing with memory management (most efficient)
        embeddings = main_advanced()
    except Exception as e:
        logger.error(f"Advanced processing failed: {e}")
        logger.info("Falling back to simple parallel processing...")
        
        # Option 2: Simple parallel processing (more reliable)
        embeddings = main_simple_parallel()
    
    logger.info(f"Processing complete. Generated {len(embeddings)} embeddings.")
