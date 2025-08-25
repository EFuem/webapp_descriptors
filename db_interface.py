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

if __name__ == "__main__":
    embeddings = main()
    print(f"Processing complete. Generated {len(embeddings)} embeddings.")

