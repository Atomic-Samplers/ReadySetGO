import numpy as np
from ase import Atoms

from readysetgo.structure_clustering.clustering_algorithms.classic_clustering_algorithm import ClassicClusteringAlgorithm
from readysetgo.structure_clustering.clustering_algorithms.hashing_clustering_algorithm import HashingClusteringAlgorithm
from readysetgo.structure_clustering.clustering_algorithms.core import ClusteringAlgorithm

def tune_tolerance_value(
    cluster_object: ClusteringAlgorithm,
    tuning_atoms_list: list[Atoms],
    target_structures: int | None = None,    
    bin_search_steps: int = 20,
    target_upper: bool = False,
    verbose: int = 0,
) -> float:
    """
    Get all clustering tolerances for the specified atoms list.
    If use_default_values is True, return the default_ct_dict.
    Otherwise, tune the clustering tolerances for each clustering algorithm
    in the cluster_object_dict and return a dictionary of clustering tolerances.
    Args:
        cluster_object (ClusteringAlgorithm): The clustering algorithm object to use for tuning.
        tuning_atoms_list (list[Atoms]): A list of Atoms objects
            to be used for tuning the clustering tolerances.
        bin_search_steps (int): The number of steps to use in the binary search
            for tuning the clustering tolerances.
        target_structures (int): The target number of unique structures to aim for
            during tuning. 
        target_upper (bool): Find the upper threshold that gives target_structures, if 
            false find the lower threshold.
        verbose (int): Verbosity level. If greater than 0, progress will be printed
    Returns:
        float: The tuned clustering tolerance value.
    """
    assert len(tuning_atoms_list) > 0, "tuning_atoms_list must not be empty."
    if target_structures is not None:
        assert target_structures > 0, "target_structures must be greater than 0."
        assert (
            target_structures <= len(tuning_atoms_list)
        ), "target_structures must be less than or equal to the number of structures in tuning_atoms_list."
    assert bin_search_steps > 0, "bin_search_steps must be greater than 0."
    
    if target_structures is None:
        target_structures = len(tuning_atoms_list)
    
    clustering_tolerance = target_structure_binary_search(
        steps=bin_search_steps,
        cluster_object=cluster_object,
        atoms_list=tuning_atoms_list,
        target_structures=target_structures,
        verbose=verbose,
        target_upper=target_upper,
    )
    
    return clustering_tolerance


def target_structure_binary_search(
    cluster_object: ClusteringAlgorithm,
    atoms_list: list[Atoms],
    steps: int = 20,
    target_structures: int = 100,
    start_min: float = 0.0,
    start_max: float = 5.0,
    verbose: int = 0,
    target_upper: bool = True,
):
    """
    Perform a binary search to find the optimal tolerance for the clustering algorithm
    that results in a number of unique structures close to the target number.
    """
    cluster_object.set_attribute("atoms_list", atoms_list)
    tolerance = start_max / 2
    best_min = start_min
    best_max = start_max
    exact_list = []
    all_diff_dict = {}
    for _ in range(steps):
        cluster_object.set_attribute("tolerance", tolerance)
        if isinstance(cluster_object, HashingClusteringAlgorithm):
            unique_structures = cluster_object.group()[1]
        elif isinstance(cluster_object, ClassicClusteringAlgorithm):
            unique_structures = len(cluster_object.group())

        all_diff_dict[tolerance] = np.abs(unique_structures - target_structures)
        # more tolerance means less unique structures
        if (
            unique_structures < target_structures
        ):  # if unique structures is less than target we need to decrease tolerance
            # decrease tolerance by moving max down
            if verbose > 1:
                print(
                    f"{unique_structures} found for tolerance: {tolerance}. Decreasing tolerance"
                )
            best_max = tolerance
        elif (
            unique_structures > target_structures
        ):  # if unique structures is more than target we need to increase tolerance
            # increase tolerance by moving min up
            if verbose > 1:
                print(
                    f"{unique_structures} found for tolerance: {tolerance}. Increasing tolerance"
                )
            best_min = tolerance
        else:  # we found the exact number of unique structures save and continue searching
            exact_list.append(tolerance)
            if (
                (target_structures == 1) or not target_upper
            ):  # if target is 1 lower the tolerance until we find more than 1
                if verbose > 1:
                    print(
                        f"{unique_structures} found for tolerance: {tolerance}. Decreasing tolerance"
                    )
                best_max = tolerance
            elif (
                target_structures == len(atoms_list) or target_upper
            ):  # if target is max increase tolerance until we find less than max
                if verbose > 1:
                    print(
                        f"{unique_structures} found for tolerance: {tolerance}. Increasing tolerance"
                    )
                best_min = tolerance
            else:  # target structures is not one or the total length default to reducing the tolerance further
                if verbose > 1:
                    print(
                        f"{unique_structures} found for tolerance: {tolerance}. Decreasing tolerance"
                    )

                    best_max = tolerance
        if verbose > 0:
            print(
                f"Current tolerance: {tolerance}, unique structures: {unique_structures}, target: {target_structures}. iter {_}/{steps}",
                end="\r",
            )
        tolerance = (best_min + best_max) / 2
    # print('exact_list', exact_list)
    if len(exact_list) > 0:
        if target_upper:
            tolerance = max(exact_list)
        else:
            tolerance = min(exact_list)
    else:
        tolerance = [
            tol
            for tol, diff in all_diff_dict.items()
            if diff == min(all_diff_dict.values())
        ][0]
    if verbose > 0:
        print(
            f"Final tolerance: {tolerance} with a difference of {all_diff_dict[tolerance]} from target structures ({target_structures}) found"
        )
    return tolerance

# def approximate_temperature_from_rattled_structures(rattled_structures):
#     calc = LennardJones()
#     ev_per_atom_list = []
#     for atoms in rattled_structures:
#         atoms.calc = calc
#         atoms.get_potential_energy()
#         ev_per_atom = atoms.get_total_energy() / len(atoms)
#         ev_per_atom_list.append(ev_per_atom)
    
#     std_dev_ev_per_atom = np.std(np.array(ev_per_atom_list))
#     temperature = (2 / 3) * (std_dev_ev_per_atom / (8.617333e-5))
#     return temperature