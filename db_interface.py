import vastdb
from vastdb.config import QueryConfig
import os

def get_vastdb_session():
    endpoint = "http://10.32.38.210"
    with open(f"/home/{os.environ['USER']}/.vast-dev/access_key_id", "r") as f:
        access_key = f.read().rstrip("\n")
    with open(f"/home/{os.environ['USER']}/.vast-dev/secret_access_key", "r") as f:
        secret_key = f.read().rstrip("\n")
    return vastdb.connect(endpoint=endpoint, access=access_key, secret=secret_key)

session = get_vastdb_session()

config = QueryConfig(
    limit_rows_per_sub_split=10_000,
    rows_per_split=1_000_000,
    num_sub_splits=10,
)
dataset_id = ""

with session.transaction() as tx:
    co_table = (
        tx.bucket("colabfit-prod").schema("prod").table("co_po_merged_innerjoin")
    )
    logger.info(f"Querying co_po_merged_innerjoin for dataset_id: {dataset_id}")
    co_data = co_table.select(
        predicate=co_table["dataset_id"] == dataset_id, config=config
    )
    for i, co_batch in enumerate(co_data):
        for co in co_batch:
        # convert to ase Atoms object
        # compute descriptors
        # save to file