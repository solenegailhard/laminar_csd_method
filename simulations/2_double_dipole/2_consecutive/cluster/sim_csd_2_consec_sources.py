'''
Systematic simulations of 2 sources with with varying temporal distance: compare ebb_layer versus ebb_layer_sliding window

Retrieve the base data MEG file and subjects multilayer surfaces mesh, coregister, invert to compute forward model
Then simulate two simultaneous 25ms gaussians with decreasing distance in between the two sources.

- coergister to multilayer surface then 
    - invert using EEB layer with specific priors, extract source time series in each layer
    - invert using sliding window EBB layer, extract source time series in each layer
- coregister to each layer surface, computes model evidence (free energy) per layer model (necessary??)

Here we aim to assess how well the two methods can dissociate two sources in time.

Ran in 200 vertices sampled from quintiles of cortical thickness, leadfield differences, orientation, distance to scalp
'''

import glob
import shutil
import sys

import json
import os
import os.path as op
import traceback
import numpy as np
import pickle

import mne
from mne import read_epochs

from scipy.spatial.transform import Rotation as R # only used when coreg error

from lameg.invert import coregister, invert_ebb, invert_ebb_layer, invert_sliding_window_ebb_layer, load_source_time_series
from lameg.laminar import sliding_window_model_comparison
from lameg.simulate import run_current_density_simulation
from lameg.util import load_meg_sensor_data, get_fiducial_coords, spm_context
from lameg.surf import LayerSurfaceSet

def run(
    json_file,
    out_folder,
    output_sim_fname,
    subject_id,
    session_id,
    n_layers,
    sim_layers,
    sim_vertex,
    err_level,
    snr,
    tmp_dist,
    dipole_moment,
    win_size,
    patch_size,
    sim_patch_size,
    n_temp_modes_ebb,
    hann_ebb=False
    ):

    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    out_path = os.path.join(parameters['output_path'], out_folder)
    os.makedirs(out_path, exist_ok=True)  

    output_file = os.path.join(out_path, f"{output_sim_fname}.pickle")

    if os.path.exists(output_file):
        print(f"Skipping existing simulation: {output_file}")
        return 

    print(f'Running simulation: {output_file}')

    path = parameters["dataset_path"]
    der_path = op.join(path, "derivatives")
    proc_path = op.join(der_path, "processed")
    sub_path = op.join(proc_path, subject_id)
    ses_path = op.join(sub_path, session_id)

    fid_coords = get_fiducial_coords(subject_id, os.path.join(path, 'raw', 'participants.tsv'))
    orig_fid = np.array([fid_coords['nas'], fid_coords['lpa'], fid_coords['rpa']])
    mean_fid = np.mean(orig_fid, axis=0)
    zero_mean_fid = np.hstack([(orig_fid - mean_fid), np.ones((3, 1))])

    fs_mri_dir = os.path.join(proc_path, 'fs', subject_id, 'mri')
    fs_laminar_dir = os.path.join(proc_path, 'fs', subject_id, 'surf', 'laminar')

    # Make tmp dir for simulation
    tmp_dir = op.join(out_path, output_sim_fname)
    os.makedirs(tmp_dir, exist_ok=True)

    # Copy necessary data files to tmp directory 
    # create folder structure fo surf_set
    fs_tmp_dir = os.path.join(tmp_dir, 'fs')
    os.makedirs(fs_tmp_dir, exist_ok=True)

    mri_dir = os.path.join(fs_tmp_dir, subject_id, 'mri')
    os.makedirs(mri_dir, exist_ok=True)  

    laminar_surf_dir = os.path.join(fs_tmp_dir, subject_id, 'surf', 'laminar')
    os.makedirs(laminar_surf_dir, exist_ok=True)  

    # copy the mri surface for coregistration 
    shutil.copy(os.path.join(fs_mri_dir, 'orig.nii'),
                os.path.join(mri_dir,'orig.nii'))

    # copy the meshes and smoothed meshes (multilayer and single layers) 
    orientation_method = 'link_vector.fixed'

    shutil.copy(os.path.join(fs_laminar_dir, 'multilayer.11.ds.link_vector.fixed.gii'),
                os.path.join(laminar_surf_dir,'multilayer.11.ds.link_vector.fixed.gii'))
    shutil.copy(os.path.join(fs_laminar_dir, f'FWHM{patch_size:.2f}_multilayer.11.ds.link_vector.fixed.mat'),
            os.path.join(laminar_surf_dir, f'FWHM{patch_size:.2f}_multilayer.11.ds.link_vector.fixed.mat'))
    shutil.copy(os.path.join(fs_laminar_dir, f'FWHM{sim_patch_size:.2f}_multilayer.11.ds.link_vector.fixed.mat'),
            os.path.join(laminar_surf_dir, f'FWHM{sim_patch_size:.2f}_multilayer.11.ds.link_vector.fixed.mat'))

    layers = np.linspace(1, 0, n_layers)
    for layer in layers:

        if layer == 1:
            name = f'pial.ds.{orientation_method}'
        elif layer == 0:
            name = f'white.ds.{orientation_method}'
        else:
            name = f'{layer:.3f}.ds.{orientation_method}'

        shutil.copy(os.path.join(fs_laminar_dir, f'{name}.gii'),
                    os.path.join(laminar_surf_dir, f'{name}.gii'))
        shutil.copy(os.path.join(fs_laminar_dir, f'FWHM{patch_size:.2f}_{name}.mat'),
                os.path.join(laminar_surf_dir, f'FWHM{patch_size:.2f}_{name}.mat'))
    
    # copy surface for get_cortical_thickness
    shutil.copy(os.path.join(fs_laminar_dir, f'pial.ds.gii'),
                os.path.join(laminar_surf_dir, f'pial.ds.gii'))
    shutil.copy(os.path.join(fs_laminar_dir, f'white.ds.gii'),
                os.path.join(laminar_surf_dir, f'white.ds.gii'))

    # Copy data files to tmp directory
    data_file = os.path.join(ses_path,
        f'spm/ppspm_converted_{subject_id}-{session_id}-motor-epo.mat'
    )
    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]
    
    for ext in ['mat', 'dat']:
        shutil.copy(
            os.path.join(data_path, f'{data_base}.{ext}'),
            os.path.join(tmp_dir,   f'{data_base}.{ext}')
        )

    base_fname = os.path.join(tmp_dir, f'{data_base}.mat')

    try:

        # inititalize the suface object
        surf_set = LayerSurfaceSet(subject_id, n_layers, fs_tmp_dir)
        # save thickness for csd
        thickness = surf_set.get_cortical_thickness()[sim_vertex]

        n_spatial_modes = 'auto'

        # Gaussian signal
        signal_width = 25  # 25ms
        _, time, _ = load_meg_sensor_data(base_fname)
        zero_time  = time[int((len(time) - 1) / 2 + 1)]
        sim_signal1 = np.exp(-((time - zero_time - tmp_dist/2) ** 2) / (2 * signal_width ** 2)).reshape(1, -1)
        sim_signal2 = np.exp(-((time - zero_time + tmp_dist/2) ** 2) / (2 * signal_width ** 2)).reshape(1, -1)

        # get the layer vertices for each reconstructed layer
        all_layers_vertices = [surf_set.get_multilayer_vertex(i, sim_vertex) for i in range(n_layers)]

        # get the layer vertices where we simulate the sources 
        sim_vertices = [
            [all_layers_vertices[l1], all_layers_vertices[l2]]
            for l1, l2 in sim_layers
        ]
        sim_signal = np.vstack([sim_signal1, sim_signal2]) #same signal for both sources in the pair
        dipole_moment = [dipole_moment, dipole_moment] #same dipole moment for both sources in the pair

        sim_vx_res = {
            "sim_vertex": sim_vertex,
            "n_layers": n_layers,
            "sim_layers": sim_layers,
            "patch_size": patch_size,
            "err_level": err_level,
            "sim_patch_size": sim_patch_size,
            "n_temp_modes_ebb": n_temp_modes_ebb,
            "n_spatial_modes": n_spatial_modes,
            "hann_windowing_ebb": hann_ebb,
            "snr": snr,
            "dipole_moment": dipole_moment,
            "sim_signal": sim_signal,
            "win_size": win_size,
            "sfreq": parameters["downsample_dataset"],
            "thickness": thickness,
            "time": time,
            "ts_ebb_layer": np.zeros((len(sim_vertices), n_layers, len(time))),
            "ts_slidwd_ebb_layer": np.zeros((len(sim_vertices), n_layers, len(time))),
            "fs_slidwd": np.zeros((len(sim_vertices), n_layers, len(time))),
        }

        with spm_context() as spm:

            # Coregister once on base data
            coregister(
                fid_coords, 
                base_fname, 
                surf_set, 
                spm_instance=spm
            )

            # Run initial inversion
            [_,_] = invert_ebb(
                base_fname, 
                surf_set, 
                patch_size=patch_size,
                n_temp_modes=n_temp_modes_ebb, 
                spm_instance=spm
            )

            # Apply coregistration error if requested
            if err_level > 0:
                while True:
                    shift_vec = np.random.randn(3)
                    shift_vec = shift_vec / np.linalg.norm(shift_vec) * np.random.randn() * err_level

                    rotation_rad = err_level * np.pi / 180.0
                    rot_vec = np.random.randn(3)
                    rot_vec = rot_vec / np.linalg.norm(rot_vec) * np.random.randn() * rotation_rad

                    P = np.concatenate((shift_vec, rot_vec))
                    rotation = R.from_rotvec(P[3:])
                    A = np.eye(4)
                    A[:3, :3] = rotation.as_matrix()
                    A[:3, 3] = P[:3]

                    new_fid_homogeneous = (A @ zero_mean_fid.T).T
                    new_fid = new_fid_homogeneous[:, :3] + mean_fid
                    max_dist = np.max(np.sqrt(np.sum((new_fid - orig_fid) ** 2, axis=-1)))
                    if np.abs(err_level - max_dist) < .05:
                        break

                coreg_fid_coords = {
                    'nas': new_fid[0, :],
                    'lpa': new_fid[1, :],
                    'rpa': new_fid[2, :]
                }
            else:
                coreg_fid_coords = fid_coords

            sim_vx_res['fid_coords'] = coreg_fid_coords

            for sim_idx, layers in enumerate(sim_layers):
                prefix = f'{output_sim_fname}_layer{str(layers).zfill(2)}_'
                sim_l_vx = sim_vertices[sim_idx] #either single or pair of vertices depending on the sim_layer_pairs
                print(f"Simulating {prefix}...")
                sim_l_fname = run_current_density_simulation(
                    base_fname, 
                    prefix, 
                    sim_l_vx, 
                    sim_signal,
                    dipole_moment, 
                    sim_patch_size, 
                    snr,
                    average_trials=True,
                    spm_instance=spm
                )

                coregister(
                    coreg_fid_coords, 
                    sim_l_fname, 
                    surf_set, 
                    spm_instance=spm)

                # inversion with ebb_layer (METHOD 1-bis)
                [_, _, MU] = invert_ebb_layer(
                    sim_l_fname, 
                    surf_set,
                    patch_size=patch_size, 
                    n_temp_modes=n_temp_modes_ebb,
                    n_spatial_modes=n_spatial_modes, 
                    foi=None, 
                    hann_windowing=hann_ebb, 
                    viz=False,
                    return_mu_matrix=True, 
                    spm_instance=spm
                )

                # retreive ebb_layer time series, only for specified vertex
                ts_ebb_layer, _, _ = load_source_time_series(
                    sim_l_fname, 
                    mu_matrix=MU, 
                    vertices=all_layers_vertices
                )

                # get the ts within full time window (for comparison with free energy)
                sim_vx_res["ts_ebb_layer"][sim_idx, :, :] = ts_ebb_layer

                # ensures you get enough samples to optimize the ebb hyperparameters 
                if win_size <= 15:
                    n_temp_modes_ebb_slidwd = 1
                else:
                    n_temp_modes_ebb_slidwd = n_temp_modes_ebb

                # inversion with sliding window ebb_layer (METHOD 1-bis) # if epoch too big import from helper
                [_, _] = invert_sliding_window_ebb_layer(
                    sim_l_fname, 
                    surf_set,
                    patch_size=patch_size, 
                    n_temp_modes=n_temp_modes_ebb_slidwd,
                    n_spatial_modes=n_spatial_modes, 
                    win_size=win_size,
                    win_overlap=True,
                    foi=None, 
                    hann_windowing=True, #hardcoded: optimal for sliding window 
                    viz=False,
                    spm_instance=spm
                )
                

                # retreive ebb_layer time series, only for specified vertex
                ts_slidwd_ebb_layer, _, _ = load_source_time_series(
                    sim_l_fname, 
                    vertices=all_layers_vertices
                )

                # get the ts within full time window (for comparison with free energy)
                sim_vx_res["ts_slidwd_ebb_layer"][sim_idx, :, :] = ts_slidwd_ebb_layer

                # model comparison (METHOD 2 - free energy model comparison, msp inversion at specific vertex)
                [fs_slidwd, _] = sliding_window_model_comparison(
                    [sim_vertex],
                    coreg_fid_coords, 
                    sim_l_fname, 
                    surf_set,
                    viz=False, 
                    spm_instance=spm,
                    invert_kwargs={
                        'patch_size': patch_size,
                        'n_temp_modes': 4, #hardcoded: optimal for fe comparison
                        'win_size': win_size, 
                        'win_overlap': True, #hardcoded: optimal for fe comparison
                    }
                )

                # saves the free energy
                sim_vx_res["fs_slidwd"][sim_idx, :, :] = fs_slidwd

                # Cleanup layer sim files
                for ext in ['mat', 'dat']:
                    for f_prefix in [prefix, f'm{prefix}']:
                        fpath = os.path.join(
                            tmp_dir,
                            f'{f_prefix}ppspm_converted_{subject_id}-{session_id}-motor-epo.{ext}'
                        )
                        if os.path.exists(fpath):
                            os.remove(fpath)

        # Save full results to pickle
        with open(output_file, "wb") as fp:
            pickle.dump(sim_vx_res, fp)

        shutil.rmtree(tmp_dir)
    
    except Exception:
        print(traceback.format_exc())
        shutil.rmtree(tmp_dir)

# ------------------------------------------------------------------------------------

if __name__ == '__main__':

    try:
        sim_idx = int(sys.argv[1])
    except Exception:
        print("incorrect simulation index")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except Exception:
        json_file = "settings_cluster.json"
        print("USING:", json_file)

    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    out_folder = 'sim_2_consec_sources'

    # Fixed params
    subject_id = 'sub-001'
    session_id = 'ses-01'
    n_layers = 11
    sim_layers = [(1, 9), (3, 7)]
    #sim_layers = [l for l in range(n_layers)]
    dipole_moment = 10
    snr_level = 0
    err_level = 0
    patch_size = 5
    sim_patch_size = 5
    n_temp_modes_ebb = 4 
    hann_ebb = False
    vertices = parameters["vertices"]

    # Modulated params
    temp_dist = [10, 25, 50] # in ms, added to the second source in the pair
    win_size = [10, 25, 50]

    # Build all (vertex, snr) combinations
    all_verts = []
    all_temp_dist = []
    all_win_sizes = []
    for vert in vertices:
        for tmp_dist_i in temp_dist:
            for win_s in win_size:
                all_verts.append(vert)
                all_temp_dist.append(tmp_dist_i)
                all_win_sizes.append(tmp_dist_i)
    
    print(f'Total number of unique simulations: {len(all_temp_dist)}')

    vertex_sim_idx = all_verts[sim_idx]
    tmp_dist_sim_idx = all_temp_dist[sim_idx]
    win_size_sim_idx = all_win_sizes[sim_idx]
    output_sim_fname = f"vx{vertex_sim_idx}_2_consec_sources_tmp_dist{tmp_dist_sim_idx}win_size{win_size_sim_idx}"

    run(json_file,
        out_folder,
        output_sim_fname,
        subject_id,
        session_id,
        n_layers,
        sim_layers,
        vertex_sim_idx,
        err_level,
        snr_level,
        tmp_dist_sim_idx,
        dipole_moment,
        win_size_sim_idx,
        patch_size,
        sim_patch_size,
        n_temp_modes_ebb,
        hann_ebb)