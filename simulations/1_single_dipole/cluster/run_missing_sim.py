import sys
import os
import json
import itertools


def indices_to_slurm_array_spec(indices):
    """Convert a list of indices to a compact Slurm array spec string.
    e.g. [0,1,2,5,6,10] -> '0-2,5-6,10'
    """
    if not indices:
        return ""
    indices = sorted(set(indices))
    ranges = []
    start = indices[0]
    end = indices[0]
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = i
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def check_completion(out_path, vertices, type_sim, modulated_params):
    """
    Check which simulations are complete and which are missing.

    Parameters
    ----------
    out_path         : str   Directory containing the output .pickle files.
    vertices         : list  List of vertex indices.
    type_sim         : str   Simulation type tag used in filenames.
    modulated_params : dict  Mapping of {param_name: [values, ...]}.
                              One or several params can be given; every
                              combination of their values is checked, e.g.
                              {"snr": [-50, -35], "patchsize": [2.5, 10]}
                              -> 4 combos per vertex.

    Returns
    -------
    missing_sim_idx : list
        List of flat indices (into vertices x param-combinations) that are missing.
    """
    param_names = list(modulated_params.keys())
    # cartesian product of all param value lists, e.g. [(-50, 2.5), (-50, 10), ...]
    param_combos = list(itertools.product(*[modulated_params[name] for name in param_names]))

    missing = []
    missing_sim_idx = []
    all_sim_idx = []

    for vertex in vertices:
        for combo in param_combos:
            key = (vertex,) + combo
            all_sim_idx.append(key)

            # build the filename suffix from each param name/value pair, in order
            param_str = "_".join(f"{name}{value}" for name, value in zip(param_names, combo))
            output_file = os.path.join(
                out_path,
                f"vx{vertex}_{type_sim}_{param_str}.pickle"
            )
            if not os.path.exists(output_file):
                missing.append(key)
                missing_sim_idx.append(all_sim_idx.index(key))

    completed_sim = len(all_sim_idx) - len(missing)
    print(f'Simulations in {out_path}: {completed_sim}/{len(all_sim_idx)}')

    # create a mapping file
    mapping_file = "missing_tasks_map.txt"
    with open(mapping_file, 'w') as f:
        for idx in missing_sim_idx:
            f.write(f"{idx}\n")
    print(f'Missing sim_idx saved to txt file: use #SBATCH --array=@{mapping_file}].txt%100')

    # if mapping file not supported by slurm
    array_spec = indices_to_slurm_array_spec(missing_sim_idx)
    array_spec_file = "missing_tasks_array_spec.txt"
    with open(array_spec_file, 'w') as f:
        f.write(array_spec)

    if array_spec:
        print(f"Use in your SBATCH script:")
        print(f"  #SBATCH --array={array_spec}%100")
    else:
        print("No missing simulations — nothing to requeue.")

    return missing_sim_idx


if __name__ == '__main__':
    try:
        json_file = sys.argv[1]
        print("USING:", json_file)
    except Exception:
        json_file = "settings_cluster.json"
        print("USING:", json_file)

    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    out_folder = 'sim_snr'

    # Fixed params
    subject_id = 'sub-001'
    session_id = 'ses-01'
    vertices = parameters["vertices"]

    # type of sim
    type_sim = '1_source'
    # type_sim = '2_simult_sources'
    # type_sim = '2_consec_sources'

    # Modulated params — pass as many as you like, each as a name -> list of values.
    modulated_params = {
        "snr": [-50, -35, -20, -10, -5, 0, 5],
        # "err": [0.5, 1, 2, 3, 4, 5],
        # "patchsize": [2.5, 10],
    }

    check_completion(out_folder, vertices, type_sim, modulated_params)