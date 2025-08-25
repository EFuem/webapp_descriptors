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
from typing import List, Dict, Any

def get_vastdb_session():
    endpoint = "http://10.32.38.210"
    with open(f"/home/{os.environ['USER']}/.vast-dev/access_key_id", "r") as f:
        access_key = f.read().rstrip("\n")
    with open(f"/home/{os.environ['USER']}/.vast-dev/secret_access_key", "r") as f:
        secret_key = f.read().rstrip("\n")
    return vastdb.connect(endpoint=endpoint, access=access_key, secret=secret_key)

def process_atom_batch(batch_data: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Process a batch of atom structures using multiprocessing"""
    from generate_descriptors import get_embeddings
    from ase import Atoms
    from ast import literal_eval
    
    embeddings = []
    for b in batch_data:
        try:
            atoms = Atoms(
                positions=literal_eval(b['positions']), 
                numbers=literal_eval(b['atomic_numbers']), 
                cell=literal_eval(b['cell']), 
                pbc=literal_eval(b['pbc'])
            )
            e = get_embeddings(atoms)
            embeddings.append(e)
        except Exception as e:
            print(f"Error processing atom: {e}")
            embeddings.append(None)
    
    return embeddings

def process_batch_parallel(batch_list: List[Dict[str, Any]], n_workers: int = None) -> List[np.ndarray]:
    """Process a batch using multiprocessing"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Limit to avoid overwhelming the system
    
    # Split batch into chunks for each worker
    chunk_size = max(1, len(batch_list) // n_workers)
    chunks = [batch_list[i:i + chunk_size] for i in range(0, len(batch_list), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_atom_batch, chunk) for chunk in chunks]
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                print(f"Error in worker: {e}")
                results.extend([None] * len(chunks[0]))  # Placeholder for failed chunks
    
    return results

def process_batch_sequential(batch_list: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Process a batch sequentially (fallback)"""
    return process_atom_batch(batch_list)

def main_optimized():
    """Main function with optimized processing"""
    session = get_vastdb_session()
    
    config = QueryConfig(
        limit_rows_per_sub_split=10_000,
        rows_per_split=1_000_000,
        num_sub_splits=10,
    )
    dataset_id = "DS_h0mshvvbxlai_0"
    
    # Determine optimal number of workers
    n_workers = min(mp.cpu_count(), 8)  # Limit to avoid overwhelming the system
    print(f"Using {n_workers} workers for parallel processing")
    
    all_embeddings = []
    start_time = time.time()
    
    with session.transaction() as tx:
        co_table = (
            tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
        )
        co_data = co_table.select(
            predicate=co_table["dataset_id"] == dataset_id, config=config
        )
        
        for i, co_batch in tqdm(enumerate(co_data), desc="Processing batches"):
            batch_list = co_batch.to_pylist()
            
            if len(batch_list) > 0:
                # Process batch in parallel
                batch_embeddings = process_batch_parallel(batch_list, n_workers)
                all_embeddings.extend(batch_embeddings)
                
                # Progress update
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(all_embeddings) / elapsed if elapsed > 0 else 0
                    print(f"Processed {len(all_embeddings)} structures, Rate: {rate:.2f} structures/sec")
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total structures processed: {len(all_embeddings)}")
    print(f"Average rate: {len(all_embeddings) / total_time:.2f} structures/sec")
    
    return all_embeddings

def main_hybrid():
    """Hybrid approach: parallel processing with controlled concurrency"""
    session = get_vastdb_session()
    
    config = QueryConfig(
        limit_rows_per_sub_split=10_000,
        rows_per_split=1_000_000,
        num_sub_splits=10,
    )
    dataset_id = "DS_h0mshvvbxlai_0"
    
    # Use fewer workers to avoid overwhelming GPU memory
    n_workers = min(4, mp.cpu_count() // 2)
    print(f"Using {n_workers} workers for hybrid processing")
    
    all_embeddings = []
    start_time = time.time()
    
    with session.transaction() as tx:
        co_table = (
            tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
        )
        co_data = co_table.select(
            predicate=co_table["dataset_id"] == dataset_id, config=config
        )
        
        # Process with controlled concurrency
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for i, co_batch in enumerate(co_data):
                batch_list = co_batch.to_pylist()
                if len(batch_list) > 0:
                    future = executor.submit(process_atom_batch, batch_list)
                    futures.append((future, len(batch_list)))
                
                # Limit concurrent futures to avoid memory issues
                if len(futures) >= n_workers * 2:
                    # Wait for some to complete
                    for future, batch_size in futures[:n_workers]:
                        try:
                            batch_embeddings = future.result()
                            all_embeddings.extend(batch_embeddings)
                        except Exception as e:
                            print(f"Error in batch: {e}")
                            all_embeddings.extend([None] * batch_size)
                    
                    # Remove completed futures
                    futures = futures[n_workers:]
            
            # Process remaining futures
            for future, batch_size in futures:
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error in batch: {e}")
                    all_embeddings.extend([None] * batch_size)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total structures processed: {len(all_embeddings)}")
    print(f"Average rate: {len(all_embeddings) / total_time:.2f} structures/sec")
    
    return all_embeddings

if __name__ == "__main__":
    # Choose your optimization strategy
    print("Starting optimized processing...")
    
    # Option 1: Full parallel processing (faster but more memory intensive)
    # embeddings = main_optimized()
    
    # Option 2: Hybrid approach (balanced performance and memory usage)
    embeddings = main_hybrid()
    
    print(f"Processing complete. Generated {len(embeddings)} embeddings.")
