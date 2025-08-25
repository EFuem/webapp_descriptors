import vastdb
from vastdb.config import QueryConfig
import os
import sys
from ase import Atoms
from ast import literal_eval
from generate_descriptors import get_embeddings
from tqdm import tqdm

def get_vastdb_session():
    endpoint = "http://10.32.38.210"
    with open(f"/home/{os.environ['USER']}/.vast-dev/access_key_id", "r") as f:
        access_key = f.read().rstrip("\n")
    with open(f"/home/{os.environ['USER']}/.vast-dev/secret_access_key", "r") as f:
        secret_key = f.read().rstrip("\n")
    return vastdb.connect(endpoint=endpoint, access=access_key, secret=secret_key)

def process_single_atom(atom_data):
    """Process a single atom structure - can be used with multiprocessing.map"""
    try:
        atoms = Atoms(
            positions=literal_eval(atom_data['positions']), 
            numbers=literal_eval(atom_data['atomic_numbers']), 
            cell=literal_eval(atom_data['cell']), 
            pbc=literal_eval(atom_data['pbc'])
        )
        e = get_embeddings(atoms)
        return e
    except Exception as e:
        print(f"Error processing atom: {e}")
        return None

def process_batch_atoms(batch_data):
    """Process a batch of atom structures - can be used with multiprocessing.map"""
    results = []
    for atom_data in batch_data:
        result = process_single_atom(atom_data)
        results.append(result)
    return results

def main():
    """Main function for database processing and embedding generation"""
    session = get_vastdb_session()
    
    config = QueryConfig(
        limit_rows_per_sub_split=10_000,
        rows_per_split=1_000_000,
        num_sub_splits=10,
    )
    dataset_id = "DS_h0mshvvbxlai_0"
    
    all_embeddings = []
    
    with session.transaction() as tx:
        co_table = (
            tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
        )
        co_data = co_table.select(
            predicate=co_table["dataset_id"] == dataset_id, config=config
        )
        for i, co_batch in tqdm(enumerate(co_data)):
            batch_list = co_batch.to_pylist()
            for b in batch_list:
                atoms = Atoms(positions = literal_eval(b['positions']), numbers = literal_eval(b['atomic_numbers']), cell = literal_eval(b['cell']), pbc = literal_eval(b['pbc']))
                e = get_embeddings(atoms)
                all_embeddings.append(e)
    
    return all_embeddings

def main_parallel():
    """Main function using multiprocessing for parallel processing"""
    import multiprocessing as mp
    
    session = get_vastdb_session()
    
    config = QueryConfig(
        limit_rows_per_sub_split=10_000,
        rows_per_split=1_000_000,
        num_sub_splits=10,
    )
    dataset_id = "DS_h0mshvvbxlai_0"
    
    all_embeddings = []
    
    with session.transaction() as tx:
        co_table = (
            tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
        )
        co_data = co_table.select(
            predicate=co_table["dataset_id"] == dataset_id, config=config
        )
        
        # Collect all atom data first
        all_atom_data = []
        for i, co_batch in enumerate(co_data):
            batch_list = co_batch.to_pylist()
            all_atom_data.extend(batch_list)
        
        print(f"Processing {len(all_atom_data)} structures with multiprocessing...")
        
        # Use multiprocessing to process atoms in parallel
        n_workers = min(mp.cpu_count(), 4)  # Limit workers to avoid overwhelming the system
        print(f"Using {n_workers} workers")
        
        with mp.Pool(processes=n_workers) as pool:
            # Process atoms in parallel using map
            results = list(tqdm(
                pool.imap(process_single_atom, all_atom_data, chunksize=100),
                total=len(all_atom_data),
                desc="Processing atoms"
            ))
            
            all_embeddings = [r for r in results if r is not None]
    
    return all_embeddings

if __name__ == "__main__":
    # Choose your processing method
    print("Choose processing method:")
    print("1. Sequential processing")
    print("2. Parallel processing with multiprocessing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("Using parallel processing...")
        embeddings = main_parallel()
    else:
        print("Using sequential processing...")
        embeddings = main()
    
    print(f"Processing complete. Generated {len(embeddings)} embeddings.")

