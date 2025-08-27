import numpy as np
from ase.io import read
from mace.calculators import MACECalculator, mace_mp
from tqdm import tqdm
#import faiss

# ----------------------------------------------------
# 1. Load MACE model
# ----------------------------------------------------
# Example: "large" pretrained model from MACE (foundation model)
calculator = mace_mp(model="medium")
# ----------------------------------------------------
# 2. Load structures
# ----------------------------------------------------
# Assume two datasets stored in .xyz files (can be CIF/POSCAR etc.)
dataset_A = read("DS_h0mshvvbxlai_0_0.xyz", index=":100")
#dataset_B = read("DS_qph0akhjv9kv_0_0.xyz", index=":25")
#dataset_B = read("DS_z3s0qui5vg5c_0_0.xyz", index=":25")
# ----------------------------------------------------
# 3. Generate embeddings
# ----------------------------------------------------
def get_embeddings(atoms):
    """Return list of structure embeddings (mean-pooled atom embeddings)."""
    #embeddings = []
    #for atoms in tqdm(structures):
    # get dictionary with energy, forces, per-atom features
    results = calculator.get_descriptors(atoms)
    # results["atomic_features"]: shape (n_atoms, d)
    atom_embs, idx = results
    #print(atom_embs.shape, results[-2].shape)
    # aggregate per-structure (mean-pooling)
    emb = []
    for n in range(len(atoms)):
        indices = [i for i, x in enumerate(idx) if x == n]
        struct_emb = atom_embs[indices]
        print (struct_emb.shape)
        emb.append(struct_emb.mean(axis=0))
    #embeddings.append(struct_emb)
    return emb
    #return np.vstack(embeddings).astype(np.float32)
for i in range(10):
    emb_A = get_embeddings(dataset_A[i*10:(i+1)*10])   # shape (nA, d)
sys.exit()
#emb_B = get_embeddings(dataset_B)   # shape (nB, d)

#print("Dataset A:", emb_A.shape, "Dataset B:", emb_B.shape)

# ----------------------------------------------------
# 4. Dataset-level fingerprints
# ----------------------------------------------------
# (a) Simple mean embedding
#fp_A = emb_A.mean(axis=0)
#fp_B = emb_B.mean(axis=0)

# (b) Gaussian approx (mean + covariance diag)
#mu_A, mu_B = fp_A, fp_B
#cov_A = emb_A.var(axis=0)
#cov_B = emb_B.var(axis=0)

# ----------------------------------------------------
# 5. Compare datasets
# ----------------------------------------------------
def cosine_similarity(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

def frechet_distance(mu1, cov1, mu2, cov2):
    # simplified Fréchet distance (diagonal covariances)
    diff = np.sum((mu1 - mu2) ** 2)
    cov_diff = np.sum(cov1 + cov2 - 2 * np.sqrt(cov1 * cov2))
    return diff + cov_diff

#print("Cosine similarity:", cosine_similarity(fp_A, fp_B))
#print("Fréchet distance:", frechet_distance(mu_A, cov_A, mu_B, cov_B))

# ----------------------------------------------------
# 6. Per-structure similarity search (FAISS demo)
# ----------------------------------------------------
# dim = emb_A.shape[1]
# faiss.normalize_L2(emb_A)  # for cosine search
# faiss.normalize_L2(emb_B)

#index = faiss.IndexFlatIP(emb_A.shape[1])  # inner product = cosine after L2 norm
#index.add(emb_B)                # search in dataset B
#sims, idxs = index.search(emb_A[:5], k=3)  # nearest neighbors for first 5 in A, default: k=3
#print("Example per-structure neighbors (from dataset B):")
#print(sims)
