"""
lameg_pipeline.py
=================
Reorganized LaMEG pipeline as a class.
 
Responsibilities
----------------
- LamegPipeline.__init__   : load config, build paths, initialise LayerSurfaceSet
- sample_verts_per_anat    : representative vertex sampling across anatomical quintiles
- initialize_base_fname    : set up per-simulation tmp directory (copy meshes + data)
- compute_sim              : build / load a laminar simulation
- load_or_compute_ebb      : invert real data (multiple methods) and return layer time-series
- load_or_compute_fe       : invert real data (sliding-window EBB) and return free energy
- filename_from_params     : deterministic filename from a parameter dict
"""
 
import io
import json
import os
import os.path as op
import pickle
import shutil
import sys
import tempfile
import time as time_sys
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
 
from lameg.invert import (
    coregister,
    get_lead_field_rms_diff,
    invert_ebb,
    invert_ebb_layer,
    invert_sliding_window_ebb,
    invert_sliding_window_ebb_layer,
    load_source_time_series,
)
from lameg.laminar import sliding_window_model_comparison
from lameg.simulate import run_current_density_simulation
from lameg.surf import LayerSurfaceSet
from lameg.util import spm_context, load_meg_sensor_data, get_fiducial_coords
from lameg.viz import color_map, show_surface

from .config_loader import load_config #ensure that the two script are well into utils

# ---------------------------------------------------------------------------
# General helper
# ---------------------------------------------------------------------------
def filename_from_params(params_run, prefix, extension, exclude=None):
    """
    Build a deterministic filename from a parameter dictionary.

    Parameters
    ----------
    params_run : dict
        Run parameters to encode in the filename.
    prefix : str
        Purpose tag, e.g. 'multilayer_ts', 'layer_fe', 'fig'.
    extension : str
        File extension including the dot, e.g. '.npy', '.png'.
    exclude : list, optional
        Keys to omit from the filename (e.g. ['subject_id', 'session_id']).

    Returns
    -------
    str  Filename (no directory component).
    """
    exclude = set(exclude or ["n_jobs"] if "n_jobs" in params_run else [])

    # Build abbreviated key names: first letter after each '_' segment
    # e.g. sim_patch_size -> sps, win_size -> ws
    # Collision resolution: if two keys share an abbreviation, extend
    # both one character at a time until they differ.
    def abbreviate(key):
        return "".join(part[0] for part in key.split("_") if part)

    def extend_abbrev(key, length):
        """Return abbreviation using `length` chars per segment (min 1)."""
        return "".join(part[:length] for part in key.split("_") if part)

    keys = sorted(k for k in params_run if k not in exclude)

    # Resolve collisions by progressively lengthening segments
    abbrevs = {}
    for k in keys:
        abbrevs[k] = abbreviate(k)

    changed = True
    while changed:
        changed = False
        # Group keys by their current abbreviation
        from collections import defaultdict
        groups = defaultdict(list)
        for k, a in abbrevs.items():
            groups[a].append(k)
        for a, colliding in groups.items():
            if len(colliding) > 1:
                # Increase length by 1 for all colliding keys
                cur_len = max(
                    len(part)
                    for k in colliding
                    for part in k.split("_") if part
                )
                new_len = cur_len + 1  # won't help if already at max, but keys would be identical then
                for k in colliding:
                    abbrevs[k] = extend_abbrev(k, new_len)
                changed = True
    
    # Check for colliding string values
    string_values = [str(params_run[k]) for k in keys 
                    if params_run[k] is not None and isinstance(params_run[k], str)]
    colliding_string_keys = {
        k for k in keys
        if params_run[k] is not None
        and isinstance(params_run[k], str)
        and string_values.count(str(params_run[k])) > 1
    }

    parts = [prefix]
    for k in keys:
        v = params_run[k]
        if v is None:
            continue
        if isinstance(v, str) and k not in colliding_string_keys:
            parts.append(v)
        else:
            parts.append(f"{abbrevs[k]}{v}")

    return "_".join(parts) + extension

# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------
 
class LamegPipeline:
    """
    Configuration, paths, and surface set for one
    subject / session / dataset combination.
 
    Parameters
    ----------
    dataset    : str   Dataset identifier (used by load_config).
    subject_id : str   FreeSurfer / subject id, e.g. 'sub-001'.
    session_id : str   session id, e.g. 'ses-01'.
    env        : str   Environment tag passed to load_config ('local', 'hpc', …).
    n_layers   : int   Number of cortical layers (default 11).
    """
 
    # Default EBB inversion parameters
    DEFAULT_PARAMS_RUN_EBB = {  # type: Dict
        "unique_id": None,  # for filename generation; if None, ignored in filename
        "prefix": "",  
        "cut_stage": 'p',
        "proc_stage": '-ave',
        "win_size": 25,
        "patch_size": 5,
        "win_overlap": True,
        "n_temp_modes": 4,
        "n_spatial_modes": 'auto', #full n_spatial_modes
        "wois": None,
        "hann_windowing": False,
        "method": "slidwd_ebb_layer",
        "n_jobs":-1,
    }
    DEFAULT_PARAMS_RUN_MSP = {  # type: Dict
        "unique_id": None,  # for filename generation; if None, ignored in filename
        "prefix": "",  
        "cut_stage": 'p',
        "proc_stage": '-ave',
        "vertex": None,
        "win_size": 25,
        "patch_size": 5,
        "win_overlap": True,
        "n_temp_modes": 4,
        "n_spatial_modes": 'all', #will always be all for model comparison
        "wois": None,
        "hann_windowing": True,
        "method": "slidwd_ebb_layer",
        "n_jobs":-1,
    }
 
    DEFAULT_PARAMS_SIM = {  # type: Dict
        "unique_id": None,  # for filename generation; if None, ignored in filename
        "prefix": "sim_",  
        "cut_stage": 'p',
        "proc_stage": '-ave',
        "sim_vertex": None,
        "n_temp_modes": 4,
        "n_spatial_modes": 'all',   # filled from data rank at runtime
        "patch_size": 5,
        "sim_patch_size": [5],
        "sim_layers": [5],
        "dipole_moment": [5],
        "sim_signal": "gaussian",  # 'gaussian' | 'dipole'
        "coreg_error": None,
        "snr": None,
        "n_jobs":-1,
    }
    
    def __init__(
        self,
        dataset,
        subject_id,
        session_id,
        env="server",
        root_tmp_dir=None,
        n_layers=11,
    ):
        self.dataset    = dataset
        self.subject_id = subject_id
        self.session_id = session_id
        self.env        = env
        self.n_layers   = n_layers
 
        # --- Config ---
        self.cfg = load_config(dataset, env)
        os.environ["SUBJECTS_DIR"] = self.cfg["fs_subjects_dir"]
        self.root_tmp_dir    = root_tmp_dir if root_tmp_dir else self.cfg["default_tmp_dir"]
 
        # --- FreeSurfer source paths (always point at the real data) ---
        self.dataset_path   = self.cfg["dataset_path"]
        self.fs_mri_dir     = op.join(self.cfg["dataset_path"], subject_id, "mri")
        self.fs_laminar_dir = op.join(self.cfg["dataset_path"], subject_id, "surf", "laminar")
 
        # --- Surface set (reads from SUBJECTS_DIR, set above) ---
        self.surf_set = LayerSurfaceSet(subject_id, n_layers)
        self.fid_coords = get_fiducial_coords(self.subject_id, os.path.join(self.dataset_path, 'raw', 'participants.tsv'))
 
        # --- SPM data directory: config-driven, dataset-agnostic ---
        # Config example: "derivatives/processed/{subj_id}/{ses_id}/spm/"
        # or:             "derivatives/{subj_id}/meg/{ses_id}/spm/"
        self.spm_subj_dir = op.join(
            self.dataset_path,
            self.cfg["spm_subj_dir"].format(
                subj_id=subject_id,
                ses_id=session_id,
            ),
        )
 
        # --- Resolve spm_filename template; validate required keys ---
        # data_fname_stem and base_fname are set lazily in initialize_base_fname
        self.data_fname_stem = None  
        self.base_fname = None  

    # ------------------------------------------------------------------
    # Public: representative vertex sampling
    # ------------------------------------------------------------------
 
    def sample_verts_per_anat(
        self,
        tmp_dir_basefname = None,
        output_dir = None,
        force_recompute = False,
        remove_tmp_dir=True,
    ):
        """
        Draw a representative sample of vertices across anatomical quintiles
        (thickness, lead-field variability, orientation, distance to scalp).
 
        Saves:
        - <output_dir>/verts_per_anat_q_<dataset><subject_id>.csv
        - <output_dir>/Distrib_anat_variables_<subject_id>_n_layers<n>.png
        - <output_dir>/verts_per_anat_q_<dataset><subject_id>.pdf  (brain plot)
        Updates the dataset JSON with the selected vertex list.
 
        Returns the vertices dict (and loads from disk if already computed).
        """

        if output_dir is None:
            output_dir = op.join(Path(self.spm_subj_dir).parent, 'descript_stats')

        tag = f"verts_stats_{self.dataset}{self.subject_id}"
        csv_path = op.join(output_dir, f"{tag}.csv")
 
        if op.exists(csv_path) and not force_recompute:
            print(f"[sample_verts_per_anat] Already exists – loading: {csv_path}")
            df = pd.read_csv(csv_path)
            all_verts = df["vertex"].tolist()
            return {"vertices": all_verts, "df": df}
 
        os.makedirs(output_dir, exist_ok=True)
 
        # Coregister & invert if the SPM gain matrix does not yet exist
        p = self.DEFAULT_PARAMS_RUN_EBB

        base_fname = self.initialize_base_fname(
                        params=p,
                        tmp_dir_basefname=tmp_dir_basefname,
                        force_recompute=force_recompute,)
        inversion_exists = self._inversion_file_exist(self.data_fname_stem, self.tmp_dir_basefname)
        
        if not inversion_exists:
            with spm_context(n_jobs=-1) as spm:
                coregister(
                    self.fid_coords,
                    base_fname,
                    self.surf_set,
                    spm_instance=spm
                )
                _, _ = invert_ebb(
                    base_fname,
                    self.surf_set,
                    patch_size=p["patch_size"],
                    n_temp_modes=p["n_temp_modes"],
                    spm_instance=spm
                )

        surf_set = self.surf_set
        nb_verts     = surf_set.get_vertices_per_layer()
        thickness    = surf_set.get_cortical_thickness()
        lf_rms_diff  = get_lead_field_rms_diff(base_fname, surf_set)
        orientations = surf_set.get_radiality_to_scalp()
        distances    = surf_set.get_distance_to_scalp()

        if remove_tmp_dir:
            shutil.rmtree(self.tmp_dir_basefname)
 
        pial_verts = np.arange(nb_verts)
        df = pd.DataFrame({
            "vertex":       pial_verts,
            "thickness":    thickness,
            "lf_rms_diff":  lf_rms_diff,
            "orientations": orientations,
            "dist_to_scalp": distances,
        })
 
        # --- Quintile boundaries ---
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        quintiles = df.quantile(bins, numeric_only=True, interpolation="nearest")
 
        # --- Distribution plot ---
        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        groups = ["thickness", "lf_rms_diff", "orientations", "dist_to_scalp"]
        for ax, group in zip(axes, groups):
            ax.hist(df[group], bins=20)
            for q in quintiles.index:
                ax.axvline(quintiles.loc[q, group], color="r", linestyle="--")
            ax.set_xlabel(group)
        fig.suptitle("Distributions of anatomical variables with quintile boundaries")
        dist_fig_path = op.join(output_dir,
            f"Distrib_anat_variables_{self.subject_id}_n_layers{self.n_layers}.png")
        fig.savefig(dist_fig_path, bbox_inches="tight")
        plt.close(fig)
 
        # --- Sample 10 vertices per bin per variable ---
        df_copy = df.copy()
        verts_per_bin = {g: [] for g in groups}  # type: Dict[str, List]
 
        for g in groups:
            for i in range(len(bins) - 1):
                lower = quintiles.loc[bins[i],   g]
                upper = quintiles.loc[bins[i+1], g]
                mask  = (df_copy[g] >= lower) & (df_copy[g] < upper)
                candidates = df_copy.loc[mask, "vertex"]
                n_sample = min(10, len(candidates))
                selected = np.random.choice(candidates.values, size=n_sample, replace=False)
                verts_per_bin[g].append(selected)
                df_copy = df_copy[~df_copy["vertex"].isin(selected)]
 
        # Quintile labels on the full df
        for g in groups:
            df[f"{g}_q"] = pd.qcut(df[g], q=5, labels=False) + 1
 
        df.to_csv(csv_path, index=False)
 
        all_verts = np.array([
            v
            for group_verts in verts_per_bin.values()
            for bin_verts   in group_verts
            for v           in bin_verts
        ])
 
        # --- Brain surface visualisation ---
        cam_view = [-232, -8, -16, 60, 37, 17, -0.1, -0.35, 0.93]
        c_range  = [np.percentile(lf_rms_diff, 1), np.percentile(lf_rms_diff, 99.9)]
        colors, _ = color_map(lf_rms_diff, "Spectral_r", c_range[0], c_range[1], norm="N")
 
        plot = show_surface(
            surf_set,
            vertex_colors=colors,
            marker_vertices=all_verts,
            marker_size=2,
            camera_view=cam_view,
        )
        plot.fetch_screenshot()
        image_data = b64decode(plot.screenshot)
        image_array = np.array(Image.open(io.BytesIO(image_data)))
 
        brain_fig_path = op.join(output_dir, f"{tag}.pdf")
        fig2, ax2 = plt.subplots(figsize=(24, 16))
        ax2.imshow(image_array)
        ax2.axis("off")
        fig2.savefig(brain_fig_path, bbox_inches="tight")
        plt.close(fig2)
 
        # --- Persist vertex list to dataset JSON ---
        json_file = _find_project_root() / "config" / f"{self.dataset}.json"
        self._update_dataset_json(json_file, "vertices", [int(v) for v in all_verts])
 
        return {"vertices": all_verts.tolist(), "df": df}
 
    # ------------------------------------------------------------------
    # Public: set up per-simulation tmp directory
    # ------------------------------------------------------------------
  
    def initialize_base_fname(
        self,
        params=None,
        tmp_dir_basefname=None,
        force_recompute=False,
        check_only=False,
    ):
        """
        Resolve ``self.base_fname`` (the .mat copy in tmp) for a specific run.

        Designed to be called lazily — at the top of ``load_or_compute_ebb`` /
        ``load_or_compute_fe`` / ``compute_sim`` — only after confirming the
        final output does not already exist.  If the .mat copy is already
        present in tmp it is reused without re-copying.

        Parameters
        ----------
        params            : dict  Full run params (e.g. DEFAULT_PARAMS_RUN_EBB merged with
                                per-run overrides). Keys used: cut_stage, proc_stage,
                                subject_id, session_id — plus any extras required by
                                the spm_filename template.
        tmp_dir_basefname : str   Root output directory. Uses self.root_tmp_dir if None.
        force_recompute   : bool  Re-copy even if .mat already present in tmp.
        check_only        : bool  If True, only resolve the path and return (exists, mat_dest)
                                without copying anything.

        Returns
        -------
        base_fname : str                         (check_only=False)
        (exists, mat_dest) : (bool, str)         (check_only=True)
        """
        import string

        p = params or {}
        required_keys = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.cfg["spm_filename"])
            if field_name is not None
        }
        injected_keys = {"subject_id", "session_id"}
        missing = required_keys - injected_keys - set(p.keys())
        if missing:
            raise ValueError(
                f"initialize_base_fname: missing params {missing} required by "
                f"spm_filename template '{self.cfg['spm_filename']}'"
            )

        template_keys = {k: p[k] for k in (required_keys - injected_keys)}
        self.data_fname_stem = self.cfg["spm_filename"].format(
            subject_id=self.subject_id,
            session_id=self.session_id,
            **template_keys,
        )
        
        # --- Check-only mode: just return existence without side effects ---
        if check_only:
            mat_src = op.join(self.spm_subj_dir, f"{self.data_fname_stem}.mat")
            return op.exists(mat_src), mat_src

        # --- Resolve (or create the tmp directory)
        tmp_dir = tmp_dir_basefname or tempfile.mkdtemp(dir=self.root_tmp_dir)
        mat_dest = op.join(tmp_dir, f"{self.data_fname_stem}.mat")

        if op.exists(mat_dest) and not force_recompute:
            print(f"[initialize_base_fname] Already in tmp, skipping copy: {mat_dest}")
        else:
            os.makedirs(tmp_dir, exist_ok=True)
            print(f"[initialize_base_fname] Using tmp directory: {tmp_dir}")
            # --- Copy .mat and .dat to tmp ---
            for ext in ("mat", "dat"):
                try:                                                         
                    shutil.copy(op.join(self.spm_subj_dir, f"{self.data_fname_stem}.{ext}"), 
                                op.join(tmp_dir, f"{self.data_fname_stem}.{ext}"))
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"SPM .{ext} file not found for this subject/session or parameters "
                        f"combination at expected location: {self.spm_subj_dir} with stem {self.data_fname_stem}.{ext}. "
                    ) from exc

        self.tmp_dir_basefname = tmp_dir
        self.base_fname = mat_dest
        return self.base_fname
 
    # ------------------------------------------------------------------
    # Private: copy FreeSurfer / laminar files to a tmp directory
    # Called when fs_tmp_dir=True (parallel simulations)
    # ------------------------------------------------------------------
 
    def _copy_fs_to_tmp(self, tmp_dir_basefname, patch_size, sim_patch_size=None):
        # type: (str, float, float) -> None
        """Copy orig.nii + all laminar meshes into tmp_dir/fs/<subject_id>/…"""
        mri_dir     = op.join(tmp_dir_basefname, "fs", self.subject_id, "mri")
        laminar_tmp = op.join(tmp_dir_basefname, "fs", self.subject_id, "surf", "laminar")
        os.makedirs(mri_dir,     exist_ok=True)
        os.makedirs(laminar_tmp, exist_ok=True)
 
        shutil.copy(
            op.join(self.fs_mri_dir, "orig.nii"),
            op.join(mri_dir, "orig.nii"),
        )
 
        o = "link_vector.fixed" #o for orientation
        src = self.fs_laminar_dir
 

        fnames = [
            "multilayer.{n}.ds.{o}.gii".format(n=self.n_layers, o=o),
            "FWHM{p:.2f}_multilayer.{n}.ds.{o}.mat".format(p=patch_size, n=self.n_layers, o=o),
            "pial.ds.gii",
            "white.ds.gii",
        ]
        if sim_patch_size is not None:
            fnames.append(
                "FWHM{p:.2f}_multilayer.{n}.ds.{o}.mat".format(p=sim_patch_size, n=self.n_layers, o=o)
            )

        for fname in fnames:
            shutil.copy(op.join(src, fname), op.join(laminar_tmp, fname))
 
        for layer in np.linspace(1, 0, self.n_layers):
            if layer == 1.0:
                name = "pial.ds.{o}".format(o=o)
            elif layer == 0.0:
                name = "white.ds.{o}".format(o=o)
            else:
                name = "{l:.3f}.ds.{o}".format(l=layer, o=o)
            for fname in [
                "{name}.gii".format(name=name),
                "FWHM{p:.2f}_{name}.mat".format(p=patch_size, name=name),
            ]:
                shutil.copy(op.join(src, fname), op.join(laminar_tmp, fname))

 
    # ------------------------------------------------------------------
    # Public: simulation
    # ------------------------------------------------------------------
 
    def compute_sim(self,
        params_sim=None,
        dipole_fname=None,
        base_fname=None,
        tmp_dir_basefname=None,
        copy_to_main_dir=False,
        remove_tmp_dir=False,
    ):
        """
        Generate (or load an existing) laminar simulation.

        Parameters
        ----------
        base_fname        : str   Path to the real-data .mat file (already in tmp) otherwise build on params_sim.
        params_sim        : dict  Override any field from DEFAULT_PARAMS_SIM.
                                sim_vertex   : int | list[int]   One or several pial vertices.
                                sim_layers   : list[int] | list[list[int]]  
                                                Per-vertex layer list. If sim_vertex is a single int,
                                                sim_layers is a flat list. If sim_vertex is a list,
                                                sim_layers must be a list of lists, one per vertex.
        dipole_fname      : str | None  Path to HNN dipole file; None → Gaussian.
        copy_to_main_dir  : bool  Whether to copy the generated .mat back to the main SPM directory.

        Returns
        -------
        sim_fname : str  Path to the generated simulation .mat file.
        """
        p = {**self.DEFAULT_PARAMS_SIM, **(params_sim or {})}

        # --- Normalise sim_vertex / sim_layers to parallel lists ---
        # Single vertex:  sim_vertex=3000,        sim_layers=[5, 10]
        # Multi vertex:   sim_vertex=[3000, 5000], sim_layers=[[5, 10], [6, 8]]
        sim_vertex_raw = p["sim_vertex"]
        sim_layers_raw = p["sim_layers"]

        if isinstance(sim_vertex_raw, (int, np.integer)):
            # single vertex — wrap both in a list for uniform handling below
            pial_vertices = [int(sim_vertex_raw)]
            layers_per_vertex = [sim_layers_raw]
        else:
            pial_vertices = [int(v) for v in sim_vertex_raw]
            layers_per_vertex = sim_layers_raw
            if len(pial_vertices) != len(layers_per_vertex):
                raise ValueError(
                    f"sim_vertex has {len(pial_vertices)} entries but "
                    f"sim_layers has {len(layers_per_vertex)} — must match."
                )

        patch_size    = float(p["patch_size"])
        n_temp_modes  = int(p["n_temp_modes"])
        snr           = p["snr"]
        dm            = p["dipole_moment"]       # list, one per active layer entry
        sim_patch_size = p["sim_patch_size"]     # list, one per active layer entry
        sim_prefix = p["prefix"]

        #to check if sim exists, need to look for the averaged one
        p["prefix"] = f'm{p["prefix"]}'
        exists, sim_fname = self.initialize_base_fname(params=p, check_only=True)
        if exists:
            print(f"[compute_sim] Simulation exists at: {sim_fname}")
            return sim_fname

        try:
            p["prefix"] = ""
            base_fname = self.initialize_base_fname(
                p,
                tmp_dir_basefname=tmp_dir_basefname,
                force_recompute=False,
            )

            _, time_b, _ = load_meg_sensor_data(base_fname)

            # --- Build per-layer signal / moment / dipfwhm arrays ---
            # All indexed 0..n_layers-1 across ALL vertices; contributions accumulate.
            sim_signal     = np.zeros((self.n_layers, len(time_b)))
            dipole_moments = [0.0] * self.n_layers
            sim_dipfwhm    = [0.0] * self.n_layers

            # flat list of all (vertex, layer) active pairs — order must match dm / sim_patch_size
            active_pairs = [
                (v, l)
                for v, layers in zip(pial_vertices, layers_per_vertex)
                for l in layers
            ]
            if len(active_pairs) != len(dm):
                raise ValueError(
                    f"{len(active_pairs)} active (vertex, layer) pairs but "
                    f"dipole_moment has {len(dm)} entries — must match."
                )

            # ---- Build signal ----
            if dipole_fname is not None:
                from .utils import dipole_format
                sfreq = self.cfg.get("downsample_dataset", 1000)
                dipole_signals, _ = dipole_format("hnn", dipole_fname, sfreq)
                if len(dipole_signals) != len(active_pairs):
                    raise ValueError(
                        f"dipole file has {len(dipole_signals)} signals but "
                        f"there are {len(active_pairs)} active (vertex, layer) pairs."
                    )
                for idx, (_, l) in enumerate(active_pairs):
                    if dipole_signals[idx].shape[0] != len(time_b):
                        raise ValueError(
                            f"dipole signal {idx} length {dipole_signals[idx].shape[0]} "
                            f"≠ time length {len(time_b)}"
                        )
                    sim_signal[l] += dipole_signals[idx]   # accumulate if same layer hit twice
            else:
                signal_width = 25
                zero_time    = time_b[int((len(time_b) - 1) / 2 + 1)]
                gauss        = np.exp(-((time_b - zero_time) ** 2) / (2 * signal_width ** 2))
                for _, l in active_pairs:
                    sim_signal[l] = gauss

            for idx, (_, l) in enumerate(active_pairs):
                dipole_moments[l] = dm[idx]
                sim_dipfwhm[l]    = sim_patch_size[idx]

            # ---- Build arrays for active pairs only ----
            # active_pairs = [(pial_vertex, layer), ...] in order matching dm / sim_patch_size
            sim_vertices_active = [
                self.surf_set.get_multilayer_vertex(l, v)
                for v, l in active_pairs
            ]
            sim_signal_active = np.array([sim_signal[l] for _, l in active_pairs])
            dipole_moments_active = [dm[idx] for idx in range(len(active_pairs))]
            sim_dipfwhm_active    = [sim_patch_size[idx] for idx in range(len(active_pairs))]

            n_jobs = p.get("n_jobs", -1)

            with spm_context(n_jobs=n_jobs) as spm:
                coregister(self.fid_coords, base_fname, self.surf_set, spm_instance=spm)
                invert_ebb(
                    base_fname, self.surf_set,
                    patch_size=patch_size,
                    n_temp_modes=n_temp_modes,
                    n_spatial_modes='all',
                    spm_instance=spm,
                )
                sim_fname = run_current_density_simulation(
                    base_fname,
                    sim_prefix,
                    sim_vertices_active,
                    sim_signal_active,
                    dipole_moments_active,
                    sim_dipfwhm_active,
                    snr,
                    average_trials=True,
                    spm_instance=spm,
                )

            if copy_to_main_dir:
                for ext in ("mat", "dat"):
                    shutil.copy(op.join(self.tmp_dir_basefname, f"m{sim_prefix}{self.data_fname_stem}.{ext}"), 
                                op.join(self.spm_subj_dir, f"m{sim_prefix}{self.data_fname_stem}.{ext}"))
                print(f"Copied simulation from {self.tmp_dir_basefname} to {self.spm_subj_dir}")

        except Exception as exc:
            raise
        finally:
            if remove_tmp_dir:
                shutil.rmtree(self.tmp_dir_basefname)

        return sim_fname
 
    # ------------------------------------------------------------------
    # Public: invert real data → layer time-series
    # ------------------------------------------------------------------
 
    def load_or_compute_ebb(
        self,
        params_run,
        out_dir,
        tmp_dir_basefname = None,
        output_filename = None,
        fs_to_tmp_dir = False,
        force_recompute = False,
        remove_tmp_dir=False,
    ):
        """
        Invert real MEG data and return multi-layer time-series.
 
        Parameters
        ----------
        params_run       : dict  Run parameters (merged with DEFAULT_PARAMS_RUN).
        out_dir         : str   Directory to write output .npy files.
        tmp_dir_basefname   : str   Directory to write basefname & working directory - 
                                    if none, uses self.root_tmp_dir with an auto-generated subdir, 
                                    will be removed upon end
        output_filename  : str   Override output filename; None → auto from params.
        fs_to_tmp_dir    : bool  Whether to copy FreeSurfer files to tmp (needed for parallel simulations).
        force_recompute  : bool  Skip disk cache if True.
 
        Returns
        -------
        (multilayer_ts, time, enriched_params_run)
        """
        p = {**self.DEFAULT_PARAMS_RUN_EBB, **params_run}
 
        if output_filename is None:
            output_filename = filename_from_params(p, "multilayer_ts", ".npy")
 
        output_file      = op.join(out_dir, output_filename)
        output_time_file = op.join(out_dir, f"time_{p['cut_stage']}.npy")
 
        # ---- Cache hit ----
        if op.exists(output_file) and not force_recompute:
            print(f"[load_or_compute_ebb] Loading cached: {output_file}")
            multilayer_ts = np.load(output_file)
            time          = np.load(output_time_file) if op.exists(output_time_file) else None
            return multilayer_ts, time, p
 
        os.makedirs(out_dir, exist_ok=True)

        # ---- Prepare tmp directory & base_fname ----
        try:
            base_fname = self.initialize_base_fname(
                p,
                tmp_dir_basefname=tmp_dir_basefname,
                force_recompute=force_recompute,
            )

            # --- Optionally copy FS files to tmp (needed for parallel simulations) ---
            if fs_to_tmp_dir:
                self._copy_fs_to_tmp(self.tmp_dir_basefname, patch_size=p["patch_sizes"])
                # Surface set and SUBJECTS_DIR must now point at the tmp copy
                os.environ["SUBJECTS_DIR"] = self.tmp_dir_basefname
                self.surf_set = LayerSurfaceSet(self.subject_id, self.n_layers)

            n_spatial_modes = p["n_spatial_modes"]
            n_temp_modes  = p["n_temp_modes"]
            patch_size    = float(p["patch_size"])
            win_size      = p["win_size"]
            win_overlap   = p["win_overlap"]
            hann_windowing = p["hann_windowing"]
            method        = p["method"]

            n_jobs = p["n_jobs"] if "n_jobs" in p else -1
 
            with spm_context(n_jobs=n_jobs) as spm:

                coregister(
                    self.fid_coords, 
                    base_fname, 
                    self.surf_set, 
                    spm_instance=spm)
 
                if method == "ebb":
                    invert_ebb(
                        base_fname, self.surf_set,
                        patch_size=patch_size,
                        n_temp_modes=n_temp_modes,
                        n_spatial_modes=n_spatial_modes,
                        hann_windowing=hann_windowing,
                        spm_instance=spm,
                    )
 
                elif method == "ebb_layer":
                    invert_ebb_layer(
                        base_fname, self.surf_set,
                        patch_size=patch_size,
                        n_temp_modes=n_temp_modes,
                        n_spatial_modes=n_spatial_modes,
                        hann_windowing=hann_windowing,
                        spm_instance=spm,
                    )
                
                # call function _invert_sliding that allows to compute on larger epochs 
                # by blocks (otherwise too heavy for RAM)
                elif method in ("ebb_slidwd", "ebb_layer_slidwd"):
                    multilayer_ts, time_b = self._invert_sliding(
                        base_fname, time_b, p, spm,
                        patch_size, 
                        n_temp_modes, 
                        n_spatial_modes,
                        win_size, 
                        win_overlap, 
                        hann_windowing, method,
                    )
                    # save and return early from the sliding branch
                    np.save(output_file, multilayer_ts)
                    np.save(output_time_file, time_b)
                    return multilayer_ts, time_b, p
 
                else:
                    raise ValueError(f"Unknown method: {method!r}")
 
                # Non-sliding: extract full time-series from MU matrix directly
                multilayer_ts, time_b, _ = load_source_time_series(base_fname)
 
            np.save(output_file, multilayer_ts)
            np.save(output_time_file, time_b)
 
        except Exception as exc:
            raise
        finally:
            if remove_tmp_dir:
                shutil.rmtree(self.tmp_dir_basefname)
 
        return multilayer_ts, time_b, p
 
    # ------------------------------------------------------------------
    # Public: invert real data → free energy
    # ------------------------------------------------------------------

    def load_or_compute_fe(
        self,
        params_run,
        out_dir,
        tmp_dir_basefname,
        output_filename = None,
        fs_to_tmp_dir = False,
        force_recompute = False,
        remove_tmp_dir=False,
    ):
        """
        Invert real MEG data with sliding-window MSP and return the free energy.
 
        Returns
        -------
        (mean_layer_ts, time, params_run)
        """
        p = {**self.DEFAULT_PARAMS_RUN_MSP, **params_run}
 
        if output_filename is None:
            output_filename = filename_from_params(p, "layer_fe", ".pickle")
 
        output_file      = op.join(out_dir, output_filename)
 
        # ---- Cache hit ----
        if op.exists(output_file) and not force_recompute:
            print(f"[load_or_compute_fe] Loading cached: {output_file}")
            with open(output_file, "rb") as fp:
                layer_fe = pickle.load(fp)
            return layer_fe, p
 
        os.makedirs(out_dir, exist_ok=True)

        # ---- Prepare tmp directory & base_fname ----
        try:
 
            base_fname = self.initialize_base_fname(
                p,
                tmp_dir_basefname=tmp_dir_basefname,
                force_recompute=force_recompute,
            )

            # --- Optionally copy FS files to tmp (needed for parallel simulations) ---
            if fs_to_tmp_dir:
                self._copy_fs_to_tmp(self.tmp_dir_basefname, patch_size=p["patch_size"])
                # Surface set and SUBJECTS_DIR must now point at the tmp copy
                os.environ["SUBJECTS_DIR"] = self.tmp_dir_basefname
                self.surf_set = LayerSurfaceSet(self.subject_id, self.n_layers)

            n_jobs = p["n_jobs"] if "n_jobs" in p else -1

            with spm_context(n_jobs=n_jobs) as spm:
        
                coregister(
                    self.fid_coords, 
                    base_fname, 
                    self.surf_set, 
                    spm_instance=spm)

                [Fs, wois] = sliding_window_model_comparison(
                    p["vertex"],
                    self.fid_coords,
                    base_fname,
                    self.surf_set,
                    viz=False,
                    spm_instance=spm,
                    invert_kwargs={
                        'patch_size': p["patch_size"],
                        'n_temp_modes': p["n_temp_modes"],
                        'win_size': p["win_size"],
                        'win_overlap': p["win_overlap"],
                    }
                )

            with open(output_file, "wb") as fp:
                pickle.dump([Fs, wois], fp)
        
        except Exception as exc:
            raise
        finally:
            if remove_tmp_dir:
                shutil.rmtree(self.tmp_dir_basefname)
 
        return [Fs, wois], p
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def find_project_root(marker = "utils"):   
        for parent in Path(__file__).resolve().parents:
            if (parent / marker).exists():
                return parent
        raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

    def _inversion_file_exist(self, data_fname_stem, tmp_dir_basefname):
        """Run coreg & inversion only when the gain matrix is absent."""
        inversion_idx = 0
        gain_mat = op.join(tmp_dir_basefname, f"SPMgainmatrix_{data_fname_stem}_{inversion_idx+1}.mat")
        if op.exists(gain_mat):
            print("Inversion exists in directory")
            return True  
        return False 

    def _invert_sliding(
        self,
        base_fname,
        time_b,
        p,
        spm,
        patch_size,
        n_temp_modes,
        n_spatial_modes,
        win_size,
        win_overlap,
        hann_windowing,
        method,
    ):
        """
        Sliding-window inversion split into 200 ms blocks.
        Returns (layer_ts, time).
        need from utils import compute_wois
        """
        from .utils import compute_wois
 
        block_length  = 200  # ms
        epoch_duration = float(time_b[-1] - time_b[0])
        nb_blocks     = max(1, int(epoch_duration // block_length))
 
        layer_ts_blocks = []
        time_blocks     = []
 
        invert_fn = (
            invert_sliding_window_ebb_layer
            if method == "ebb_layer_slidwd"
            else invert_sliding_window_ebb
        )
 
        for i in range(nb_blocks):
            start = time_b[0] + i * block_length
            end   = time_b[-1] if i == nb_blocks - 1 else time_b[0] + (i + 1) * block_length
 
            wois_i = compute_wois(
                base_fname,
                start=start,
                end=end,
                win_size=win_size,
                win_overlap=win_overlap,
                n_temp_modes=n_temp_modes,
                overlap_fraction=1,
            )
 
            invert_fn(
                base_fname, self.surf_set,
                patch_size=patch_size,
                n_temp_modes=n_temp_modes,
                n_spatial_modes=n_spatial_modes,
                win_size=win_size, #will be overwritten by wois
                win_overlap=win_overlap, #will be overwritten by wois
                wois=wois_i,
                hann_windowing=hann_windowing,
                spm_instance=spm,
            )
 
            layer_ts_i, time_i, _ = load_source_time_series(base_fname)
            mask = (time_i >= start) & (time_i < end)
            layer_ts_blocks.append(layer_ts_i[:, mask])
            time_blocks.append(time_i[mask])
 
        layer_ts = np.concatenate(layer_ts_blocks, axis=-1)
        time     = np.concatenate(time_blocks)
        return layer_ts, time
 
    @staticmethod
    def _update_dataset_json(json_file: Path, key: str, value) -> None:
        """
        Write `key: value` into the dataset JSON.
        Warns and skips if the key already exists.
        """
        if not json_file.exists():
            data = {}
        else:
            with open(json_file) as f:
                data = json.load(f)
 
        if key in data:
            print(f"[_update_dataset_json] Warning: '{key}' already exists – not overwriting.")
            return
 
        data[key] = value
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

# ---------------------------------------------------------------------------
# run_logged: thin wrapper — call any LamegPipeline method and log it
# ---------------------------------------------------------------------------
# TODO do not log when the it's only loading file already existing

def run_logged(
    pipeline,           # type: LamegPipeline
    method_name,        # type: str
    method_kwargs,      # type: Dict
    calling_script,     # type: str
    extra_params=None,  # type: Optional[Dict]
):
    """
    Call ``pipeline.<method_name>(**method_kwargs)`` and log the run
    with script_logger regardless of success or failure.
 
    Parameters
    ----------
    pipeline        : LamegPipeline instance.
    method_name     : Name of the method to call, e.g. 'load_or_compute_ebb'.
    method_kwargs   : Keyword arguments forwarded to the method.
    calling_script  : Pass ``__file__`` from your script so the log records
                      the actual entry-point, not lameg_pipeline.py.
    extra_params    : Any additional key/value pairs to store in the log
                      (e.g. the full params_run dict).
 
    Returns
    -------
    Whatever the called method returns.
 
    Example
    -------
    >>> ts, time, p = run_logged(
    ...     pipeline       = pipe,
    ...     method_name    = "load_or_compute_ebb",
    ...     method_kwargs  = dict(params_run=PARAMS, out_dir=OUT),
    ...     calling_script = __file__,
    ...     extra_params   = PARAMS,
    ... )
    """
    from .script_logger import log_run
 
    t_start = time_sys.time()
    result  = None
    status  = "failed"
    error   = None
    outputs = []  # type: List[str]
 
    try:
        method = getattr(pipeline, method_name)
        result = method(**method_kwargs)
 
        # Collect output paths: if the result is a tuple whose last element
        # is a dict with an 'output_file' key, record it automatically.
        # Otherwise the caller can pass output paths via extra_params["outputs"].
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, str) and op.isfile(item):
                    outputs.append(item)
 
        status = "completed"
 
    except Exception as exc:
        error = str(exc)
        raise
 
    finally:
        log_run(
            script     = calling_script,
            config     = {"env": pipeline.cfg["env"],
                          "dataset": pipeline.cfg["dataset_name"],
                          "subject": pipeline.subject_id,
                          "session": pipeline.session_id,
                          "n_layers": pipeline.n_layers,
                          },
            params     = {"method_name": method_name, **(extra_params or method_kwargs)},
            status     = status,
            outputs    = outputs,
            error      = error,
            duration_s = time_sys.time() - t_start,
        )
 
    return result
