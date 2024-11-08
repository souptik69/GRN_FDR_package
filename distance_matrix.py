import pandas as pd
import numpy as np
import gc
import time
from dask.distributed import Client, LocalCluster
from scipy.stats import wasserstein_distance
import dask.bag as db

def compute_wasserstein_distances_hexa_split(expression_matrix: pd.DataFrame, 
                                             batch_size: int = 5000, 
                                             n_workers: int = 15, 
                                             memory_limit: str = '200GB') -> tuple[pd.DataFrame, float]:
    """
    Compute a gene-to-gene Wasserstein distance matrix for a preprocessed gene expression matrix.

    This function calculates pairwise Wasserstein distances between genes in the provided expression matrix.
    It divides the computation into six parts to leverage Dask parallel processing, enabling efficient
    computation on large datasets.

    Steps:
    1. Set up a local Dask cluster with the specified number of workers and memory limit.
    2. Create the lower-triangular index pairs for all gene combinations in the expression matrix.
    3. Split these index pairs into six parts for separate batch processing, improving memory management.
    4. Compute Wasserstein distances for each pair of genes in parallel using Dask.
    5. Aggregate results into a symmetric distance matrix and return it as a pandas DataFrame along with the
       computation time in seconds.

    Parameters:
    ----------
    expression_matrix : pd.DataFrame
        A preprocessed gene expression matrix where rows represent samples and columns represent genes.
    batch_size : int, optional, default=5000
        Number of gene pairs to process in each Dask partition batch. Adjust based on available memory and performance.
    n_workers : int, optional, default=15
        Number of workers for the Dask LocalCluster. More workers can speed up processing but may increase memory usage.
    memory_limit : str, optional, default='200GB'
        Memory limit per worker for the Dask LocalCluster. Format should be a string with a unit suffix (e.g., '200GB').

    Returns:
    -------
    tuple
        A tuple containing:
        - pd.DataFrame: Gene-to-gene Wasserstein distance matrix with genes as row and column indices.
        - float: Total computation time in seconds.


    Example:
    --------
    >>> import pandas as pd
    >>> from distance_matrix import compute_wasserstein_distances_rna_hexa_split
    >>> expression_matrix = pd.read_csv('filtered_expression_matrix.csv')
    >>> distance_matrix_df = compute_wasserstein_distances_rna_hexa_split(expression_matrix, batch_size=5000, n_workers=10, memory_limit='150GB')
    >>> print(distance_matrix_df.head())
    >>> print(f"Computation Time: {computation_time} seconds")

    Notes:
    ------
    - This function requires sufficient computational resources to handle large gene expression datasets.
    - Adjust `batch_size`, `n_workers`, and `memory_limit` to optimize memory usage and processing time depending on dataset size.
    - Garbage collection is forced at the end of the function to free up memory.
    """

    start_time = time.time()

    # Set up a local Dask cluster for parallel computing
    local_cluster = LocalCluster(n_workers=n_workers, memory_limit=memory_limit)
    custom_client = Client(local_cluster)

    # Define function to compute Wasserstein distance for a pair of gene indices
    def compute_wasserstein_pair(pair):
        i, j = pair
        return i, j, wasserstein_distance(expression_matrix.iloc[:, i], expression_matrix.iloc[:, j])

    genes = expression_matrix.columns
    num_genes = len(genes)

    # Generate indices for lower triangle of the distance matrix
    indices = np.tril_indices(num_genes, -1)
    index_pairs = list(zip(indices[0], indices[1]))

    # Split the index pairs into six parts
    part_size = len(index_pairs) // 6
    index_parts = [
        index_pairs[:part_size],
        index_pairs[part_size:2 * part_size],
        index_pairs[2 * part_size:3 * part_size],
        index_pairs[3 * part_size:4 * part_size],
        index_pairs[4 * part_size:5 * part_size],
        index_pairs[5 * part_size:]
    ]

    def compute_distances(index_pairs):
        batched_index_pairs = [index_pairs[i:i + batch_size] for i in range(0, len(index_pairs), batch_size)]
        return db.from_sequence(batched_index_pairs, npartitions=len(batched_index_pairs)).map(
            lambda batch: [compute_wasserstein_pair(pair) for pair in batch]).compute()

    results = []


    for i in range(6):
        results.append(compute_distances(index_parts[i]))

    # Initialize distance matrix
    distances_scipy = np.zeros((num_genes, num_genes))

    # Assign results to distance matrix
    for part_results in results:
        for batch_result in part_results:
            for i, j, distance in batch_result:
                distances_scipy[i, j] = distance
                distances_scipy[j, i] = distance


    # Convert to DataFrame with gene names as row and column labels
    distance_matrix_df = pd.DataFrame(distances_scipy, index=genes, columns=genes)

    custom_client.close()
    local_cluster.close()

    # Force garbage collection to free up memory
    gc.collect()

    # Calculate total time taken
    time_scipy = time.time() - start_time

    # Return the computed distance matrix
    return distance_matrix_df, time_scipy