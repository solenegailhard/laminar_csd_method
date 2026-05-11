## Check the format of the dipole from model - resample to data time format

# HNN write dipole as txt files and has a very high sampling rate

import json
import os
import os.path as op
import numpy as np
import pandas as pd
from scipy.signal import resample

def dipole_format(model_type, filename, sfreq):
    if model_type == 'hnn':
        dipole_erf = np.loadtxt(filename)
        dipole_erf = dipole_erf.T
        tot_signal = dipole_erf[1, :]
        L2_signal  = dipole_erf[2, :]
        L5_signal  = dipole_erf[3, :]
        time       = dipole_erf[0, :]

        sfreq_ori    = len(time) / (time[-1]/1000 - time[0]/1000)
        n_samples_new = int(len(time) * sfreq / sfreq_ori) + 1

        tot_signal_d = resample(tot_signal, n_samples_new)
        L2_signal_d  = resample(L2_signal,  n_samples_new)
        L5_signal_d  = resample(L5_signal,  n_samples_new)

        return [L2_signal_d, L5_signal_d], tot_signal_d

    if model_type == 'lfpy':
        raise NotImplementedError("lfpy format not yet supported")

import os
import h5py
import numpy as np

from lameg.util import load_meg_sensor_data

def compute_wois(data_fname, start, end, win_size, win_overlap, n_temp_modes, overlap_fraction=0.5):
    """
    Compute windows of interest (wois) for sliding window
    
    Parameters
    ----------
    data_fname : str
        Path to MEG data file
    start, end : float
        Time range to compute windows within (ms)
    win_size : float
        Window size (ms)
    win_overlap : bool
        Whether windows overlap
    n_temp_modes : int
        Number of temporal modes (in the sliding windows)
    overlap_fraction : float, optional
        Fraction of overlap between consecutive windows (default: 0.5 = 50% overlap)
        Only used if win_overlap=True. Must be in (0, 1).
    
    Returns
    -------
    wois : np.ndarray
        Array of shape (n_windows, 2) with [start_time, end_time] for each window
    """
    _, time, _ = load_meg_sensor_data(data_fname)
    
    # Time resolution of the data
    time_step = time[1] - time[0]          # ms
    sampling_rate = 1000.0 / time_step     # Hz
    win_steps = int(round(win_size / time_step))
    
    # Check temporal modes
    if win_steps / n_temp_modes < 2:
        raise ValueError(
            f"win_size={win_size} ms yields only {win_steps} samples "
            f"({sampling_rate:.2f} Hz sampling). With n_temp_modes={n_temp_modes}, "
            f"win_samples / n_temp_modes = {win_steps / n_temp_modes:.2f} < 2. "
            "Increase win_size or reduce n_temp_modes."
        )
    
    t0_idx = np.where(time >= start)[0][0]
    t1_idx = np.where(time <= end)[0][-1]
    
    wois = []
    
    if win_overlap:
        step_size = win_size * (1 - overlap_fraction)
        step_samples = max(1, int(round(step_size / time_step)))
        
        current_idx = t0_idx
        while current_idx + win_steps - 1 <= t1_idx:
            win_l = current_idx
            win_r = current_idx + win_steps - 1
            wois.append([time[win_l], time[win_r]])
            current_idx += step_samples
        
        # add a last window if it's large enough
        if current_idx < t1_idx and (t1_idx - current_idx) >= win_steps // 2:
            wois.append([time[current_idx], time[t1_idx]])
    
    else:
        current_idx = t0_idx
        while current_idx + win_steps <= t1_idx:
            win_l = current_idx
            win_r = min(current_idx + win_steps - 1, t1_idx)
            wois.append([time[win_l], time[win_r]])
            current_idx += win_steps
        
        if current_idx < t1_idx and (t1_idx - current_idx) >= win_steps // 2:
            wois.append([time[current_idx], time[t1_idx]])
    
    wois = np.array(wois, dtype=float)
    return wois

''' MATPLOTLIB SURFACE PLOTTER - from laminar_erf repo github'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def normal_vectors(vertices, faces):
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalize_v3(n)
    return n


def vertex_normals(vertices, faces):
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    return normalize_v3(norm)


def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5 * np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0],
                     [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0],
                     [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)


def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def shading_intensity(vertices, faces, light=np.array([0, 0, 1]), shading=0.7):
    """shade calculation based on light source
       default is vertical light.
       shading controls amount of shading.
       Also saturates so top 20 % of vertices all have max intensity."""
    face_normals = normal_vectors(vertices, faces)
    intensity = np.dot(face_normals, light)
    intensity[np.isnan(intensity)] = 1
    shading = 0.7
    # top 20% all become fully coloured
    intensity = (1 - shading) + shading * (intensity - np.min(intensity)) / (
    (np.percentile(intensity, 80) - np.min(intensity)))
    # saturate
    intensity[intensity > 1] = 1

    return intensity


def f7(seq):
    # returns uniques but in order to retain neighbour triangle relationship
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))];


def get_ring_of_neighbours(island, neighbours, vertex_indices=None, ordered=False):
    """Calculate ring of neighbouring vertices for an island of cortex
    If ordered, then vertices will be returned in connected order"""
    if not vertex_indices:
        vertex_indices = np.arange(len(island))
    if not ordered:

        neighbours_island = neighbours[island]
        unfiltered_neighbours = []
        for n in neighbours_island:
            unfiltered_neighbours.extend(n)
        unique_neighbours = np.setdiff1d(np.unique(unfiltered_neighbours), vertex_indices[island])
        return unique_neighbours


def get_neighbours_from_tris(tris, label=None):
    """Get surface neighbours from tris
        Input: tris
         Returns Nested list. Each list corresponds
        to the ordered neighbours for the given vertex"""
    n_vert = np.max(tris + 1)
    neighbours = [[] for i in range(n_vert)]
    for tri in tris:
        neighbours[tri[0]].extend([tri[1], tri[2]])
        neighbours[tri[2]].extend([tri[0], tri[1]])
        neighbours[tri[1]].extend([tri[2], tri[0]])
    # Get unique neighbours
    for k in range(len(neighbours)):
        if label is not None:
            neighbours[k] = set(neighbours[k]).intersection(label)
        else:
            neighbours[k] = f7(neighbours[k])
    return np.array(neighbours, dtype=object)


def mask_colours(colours, triangles, mask, mask_colour=None):
    """grey out mask"""
    if mask is not None:
        if mask_colour is None:
            mask_colour = np.array([0.86, 0.86, 0.86, 1])
        verts_masked = mask[triangles].any(axis=1)
        colours[verts_masked, :] = mask_colour
    return colours


def adjust_colours_pvals(colours, pvals, triangles, mask=None, mask_colour=None):
    """red ring around clusters and greying out non-significant vertices"""
    colours = mask_colours(colours, triangles, mask, mask_colour)
    neighbours = get_neighbours_from_tris(triangles)
    ring = get_ring_of_neighbours(pvals < 0.05, neighbours)
    if len(ring) > 0:
        ring_label = np.zeros(len(neighbours)).astype(bool)
        ring_label[ring] = 1
        ring = get_ring_of_neighbours(ring_label, neighbours)
        ring_label[ring] = 1
        colours[ring_label[triangles].any(axis=1), :] = np.array([1.0, 0, 0, 1])
    grey_out = pvals < 0.05
    verts_grey_out = grey_out[triangles].any(axis=1)
    colours[verts_grey_out, :] = (1.5 * colours[verts_grey_out] + np.array([0.86, 0.86, 0.86, 1])) / 2.5
    return colours


def add_parcellation_colours(colours, parcel, triangles, labels=None, mask=None, filled=False, mask_colour=None):
    """delineate regions"""
    colours = mask_colours(colours, triangles, mask, mask_colour=mask_colour)
    # normalise rois and colors
    rois = list(set(parcel))
    if 0 in rois:
        rois.remove(0)
    if labels is None:
        labels = dict(zip(rois, np.random.rand(len(rois), 4)))
    # remove transparent rois
    # find vertices that delineate rois
    if filled:
        colours = np.zeros_like(colours)
        for l, label in enumerate(rois):
            colours[np.median(parcel[triangles], axis=1) == label] = labels[label]
        return colours
    neighbours = get_neighbours_from_tris(triangles)
    matrix_colored = np.zeros([len(triangles), len(rois)])
    for l, label in enumerate(rois):
        ring = get_ring_of_neighbours(parcel != label, neighbours)
        if len(ring) > 0:
            ring_label = np.zeros(len(neighbours)).astype(bool)
            ring_label[ring] = 1
            #          ring=get_ring_of_neighbours(ring_label,neighbours)
            #           ring_label[ring]=1
            #            matrix_colored[:,l] = ring_label[triangles].sum(axis=1)
            matrix_colored[:, l] = np.median(ring_label[triangles], axis=1)  # ring_label[triangles].sum(axis=1)
    # update colours with delineation
    maxis = [max(matrix_colored[i, :]) for i in range(0, len(colours))]
    colours = np.array([labels[rois[np.random.choice(np.where(matrix_colored[i, :] == maxi)[0])]]
                        if maxi != 0 else colours[i] for i, maxi in enumerate(maxis)])
    return colours


def adjust_colours_alpha(colours, alpha):
    """grey out vertices according to scalar"""
    # rescale alpha to 0.2-1.0
    alpha_rescaled = 0.1 + 0.9 * (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    colours = (alpha_rescaled * colours.T).T + ((1 - alpha_rescaled) * np.array([0.86, 0.86, 0.86, 1]).reshape(-1, 1)).T
    colours = np.clip(colours, 0, 1)
    return colours


def frontback(T):
    """
    Sort front and back facing triangles
    Parameters:
    -----------
    T : (n,3) array
       Triangles to sort
    Returns:
    --------
    front and back facing triangles as (n1,3) and (n2,3) arrays (n1+n2=n)
    """
    Z = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 1, 1] + T[:, 0, 1]) + \
        (T[:, 2, 0] - T[:, 1, 0]) * (T[:, 2, 1] + T[:, 1, 1]) + \
        (T[:, 0, 0] - T[:, 2, 0]) * (T[:, 0, 1] + T[:, 2, 1])
    return Z < 0, Z >= 0


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def plot_surf(vertices, faces, overlay, rotate=[90, 270], cmap='viridis', filename='plot.png', label=False,
              vmax=None, vmin=None, x_rotate=270, pvals=None, colorbar=True, cmap_label='value',
              title=None, mask=None, base_size=6, arrows=None, arrow_subset=None, arrow_size=0.5,
              arrow_colours=None, arrow_head=0.05, arrow_width=0.001, coords=None, coord_colours=None, coord_size=0.02,
              mask_colour=None, transparency=1,
              show_back=False, alpha_colour=None, flat_map=False, z_rotate=0, parcel=None, parcel_cmap=None,
              filled_parcels=False, return_ax=False, ax=None):
    """ This function plot mesh surface with a given overlay.
        Features available : display in flat surface, display parcellation on top, display gradients arrows on top


    Parameters:
    ----------
        vertices     : numpy array
                       vertex locations
        faces        : numpy array
                       triangles of vertex indices definings faces
        overlay      : numpy array
                       array to be plotted
        rotate       : tuple, optional
                       rotation angle for lateral on lh,  and medial
        cmap         : string, optional
                       matplotlib colormap
        filename     : string, optional
                       name of the figure to save
        label        : bool, optional
                       colours smoothed (mean) or median if label
        vmin, vmax   : float, optional
                       min and max value for display intensity
        x_rotate     : int, optional

        pvals        : bool, optional

        colorbar     : bool, optional
                       display or not colorbar
        cmap_label   : string, optional
                       label of the colorbar
        title        : string, optional
                       title of the figure
        mask         : numpy array, optional
                       vector to mask part of the surface
        base_size    : int, optional

        arrows       : numpy array, optional
                       dipsplay arrows in the directions of gradients on top of the surface
        arrow_subset : numpy array, optional
                       vector containing at which vertices display an arrow
        arrow_size   : float, optional
                       size of the arrow
        arrow_colours:

        coords       : numpy array, optional
                       displays spheres on top of the surface
        coord_colours:

        alpha_colour : float, optional
                       value to play with transparency of the overlay
        flat_map     : bool, optional
                       display on flat map
        z_rotate     : int, optional
        transparency : float, optional
                       value between 0-1 to play with mesh transparency
        show_back    : bool, optional
                       display or hide the faces in the back of the mesh (z<0)
        parcel       : numpy array, optional
                       delineate rois on top of the surface
        parcel_cmap  : dictionary, optional
                       dic containing labels and colors associated for the parcellation
        filled_parcels: fill the parcel colours

    """
    vertices = vertices.astype(np.float)
    F = faces.astype(int)
    if coords is not None:
        coords = (coords - (vertices.max(0) + vertices.min(0)) / 2) / max(vertices.max(0) - vertices.min(0))
    vertices = (vertices - (vertices.max(0) + vertices.min(0)) / 2) / max(vertices.max(0) - vertices.min(0))
    if not isinstance(rotate, list):
        rotate = [rotate]
    if not isinstance(overlay, list):
        overlays = [overlay]
    else:
        overlays = overlay
    if parcel is not None:
        if parcel.sum() == 0:
            parcel = None
    if flat_map:
        z_rotate = 90
        rotate = [90]
        intensity = np.ones(len(F))
    else:
        # change light source if z is rotate
        light = np.array([0, 0, 1, 1]) @ yrotate(z_rotate)
        intensity = shading_intensity(vertices, F, light=light[:3], shading=0.7)
    # make figure dependent on rotations

    if ax is None:
        fig = plt.figure(figsize=(base_size * len(rotate) + colorbar * (base_size - 2),
                                  (base_size - 1) * len(overlays)))
    else:
        fig=ax.get_figure()

    if title is not None:
        plt.title(title, fontsize=25)
    #plt.axis('off')
    for k, overlay in enumerate(overlays):
        # colours smoothed (mean) or median if label
        if label:
            colours = np.median(overlay[F], axis=1)
        else:
            colours = np.mean(overlay[F], axis=1)
        if vmax is not None:
            colours = (colours - vmin) / (vmax - vmin)
            colours = np.clip(colours, 0, 1)
        else:
            vmax = colours.max()
            vmin = colours.min()
            colours = (colours - colours.min()) / (colours.max() - colours.min())
        C = np.squeeze(plt.get_cmap(cmap)(colours))
        if alpha_colour is not None:
            C = adjust_colours_alpha(C, np.mean(alpha_colour[F], axis=1).T)
        if pvals is not None:
            C = adjust_colours_pvals(C, pvals, F, mask, mask_colour=mask_colour)
        elif mask is not None:
            C = mask_colours(C, F, mask, mask_colour=mask_colour)
        if parcel is not None:
            C = add_parcellation_colours(C, parcel, F, parcel_cmap, mask, mask_colour=mask_colour,
                                         filled=filled_parcels)

        # adjust intensity based on light source here
        C[:, 0] *= intensity
        C[:, 1] *= intensity
        C[:, 2] *= intensity

        collection = PolyCollection([], closed=True, linewidth=0, antialiased=False, facecolor=C, cmap=cmap)
        for i, view in enumerate(rotate):
            MVP = perspective(25, 1, 1, 100) @ translate(0, 0, -3) @ yrotate(view) @ zrotate(z_rotate) @ xrotate(
                x_rotate) @ zrotate(270 * flat_map)
            # translate coordinates based on viewing position
            V = np.c_[vertices, np.ones(len(vertices))] @ MVP.T

            V /= V[:, 3].reshape(-1, 1)
            center = np.array([0, 0, 0, 1]) @ MVP.T;
            center /= center[3];
            # add vertex positions to A_dir before transforming them
            if arrows is not None:
                # calculate arrow position + small shift in surface normal direction
                vertex_normal_orig = vertex_normals(vertices, faces)
                A_base = np.c_[vertices + vertex_normal_orig * 0.01, np.ones(len(vertices))] @ MVP.T
                A_base /= A_base[:, 3].reshape(-1, 1)

                # calculate arrow direction
                A_dir = np.copy(arrows)
                # normalise arrow size
                max_arrow = np.max(np.linalg.norm(arrows, axis=1))
                A_dir = arrow_size * A_dir / max_arrow
                A_dir = np.c_[A_dir, np.ones(len(A_dir))] @ MVP.T
                A_dir /= A_dir[:, 3].reshape(-1, 1)
            # A_dir *= 0.1;

            if coords is not None:
                C_base = np.c_[coords, np.ones(coords.shape[0])] @ MVP.T
                C_base /= C_base[:, 3].reshape(-1, 1)

            V = V[F]

            # triangle coordinates
            T = V[:, :, :2]
            # get Z values for ordering triangle plotting
            Z = -V[:, :, 2].mean(axis=1)
            # sort the triangles based on their z coordinate. If front/back views then need to sort a different axis
            front, back = frontback(T)
            if show_back == False:
                T = T[front]
                s_C = C[front]
                Z = Z[front]
            else:
                s_C = C
            I = np.argsort(Z)
            T, s_C = T[I, :], s_C[I, :]
            if ax is None:
                ax = fig.add_subplot(len(overlays), len(rotate) + 1, 2 * k + i + 1, xlim=[-.98, +.98], ylim=[-.98, +.98],
                                     aspect=1, frameon=False,
                                     xticks=[], yticks=[])
            collection = PolyCollection(T, closed=True, linewidth=0, antialiased=False, facecolor=s_C, cmap=cmap, edgecolors=None)
            collection.set_alpha(transparency)
            ax.add_collection(collection)
            # add arrows to image
            if arrows is not None:
                front_arrows = F[front].ravel()
                for arrow_index, i in enumerate(arrow_subset):
                    if i in front_arrows and A_base[i, 2] < center[2] + 0.01:
                        arrow_colour = 'k'
                        if arrow_colours is not None:
                            arrow_colour = arrow_colours[arrow_index]
                        # if length of arrows corresponds perfectly with coordinates
                        # assume 1:1 matching
                        if len(A_dir) == len(A_base):
                            direction = A_dir[i]
                        # otherwise, assume it is a custom list matching the
                        elif len(A_dir) == len(arrow_subset):
                            direction = A_dir[arrow_index]
                        half = direction * 0.5

                        ax.arrow(A_base[i, 0] - half[0],
                                 A_base[i, 1] - half[1],
                                 direction[0], direction[1],
                                 head_width=arrow_head, width=arrow_width,
                                 color=arrow_colour)
                    # ax.arrow(A_base[idx,0], A_base[idx,1], A_dir[i,0], A_dir[i,1], head_width=0.01)
            if coords is not None:
                for i in range(coords.shape[0]):
                    coord_colour = 'r'
                    if coord_colours is not None:
                        coord_colour = coord_colours[i]
                    circle = plt.Circle(C_base[i,:], coord_size, color=coord_colour)
                    ax.add_patch(circle)

            #plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if colorbar:
        l = 0.7
        if len(rotate) == 1:
            l = 0.5
        cbar_size = [l, 0.3, 0.03, 0.38]
        cbar = fig.colorbar(collection,
                            ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels([np.round(vmin, decimals=2), np.round(np.mean([vmin, vmax]), decimals=2),
                                 np.round(vmax, decimals=2)])
        #cbar.ax.tick_params(labelsize=25)
        cbar.ax.set_title(cmap_label)

    if return_ax:
        return fig, ax, MVP
    return


import os
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def _mesh_adjacency(faces):
    """
    Compute a vertex adjacency matrix from a triangular surface mesh.

    This function derives the vertex-vertex connectivity structure of a mesh by
    identifying shared edges among faces. The resulting sparse matrix can be used
    for neighborhood-based computations such as smoothing, interpolation, or graph-based
    traversal.

    Parameters
    ----------
    faces : np.ndarray, shape (F, 3)
        Array of triangular faces, where each row contains vertex indices.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (V, V)
        Binary sparse adjacency matrix, where entry (i, j) = 1 indicates that
        vertices *i* and *j* share an edge. The matrix is symmetric.

    Notes
    -----
    - The number of vertices *V* is inferred from the maximum index in `faces` + 1.
    - Duplicate edges are merged automatically by sparse matrix construction.
    - The diagonal of the matrix is zero (no self-connections).

    Examples
    --------
    >>> adj = _mesh_adjacency(faces)
    >>> adj.shape
    (10242, 10242)
    >>> adj.nnz  # number of edges * 2 (since symmetric)
    61440
    """

    faces = np.asarray(faces, dtype=int)
    n_vertices = np.max(faces) + 1  # Assuming max vertex index represents the number of vertices

    # Flatten the indices to create row and column indices for the adjacency matrix
    row_indices = np.hstack([faces[:, 0], faces[:, 0], faces[:, 1],
                             faces[:, 1], faces[:, 2], faces[:, 2]])
    col_indices = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 2], faces[:, 0], faces[:, 1]])

    # Create a sparse matrix from row and column indices
    adjacency = csr_matrix(
        (np.ones_like(row_indices), (row_indices, col_indices)),
        shape=(n_vertices, n_vertices)
    )

    # Ensure the adjacency matrix is binary
    adjacency = (adjacency > 0).astype(int)

    return adjacency


def find_clusters(faces, threshold_indices, n_hops=1):
    
    adjacency_matrix = _mesh_adjacency(faces).tocsr()

    # Build N-hop adjacency matrix
    expanded_adj = adjacency_matrix.copy()
    power = adjacency_matrix.copy()

    for _ in range(1, n_hops):
        power = power @ adjacency_matrix  # Matrix multiplication for hop expansion
        expanded_adj = expanded_adj + power

    # Binarize: set all non-zero entries to 1
    expanded_adj.data[:] = 1.0
    expanded_adj.eliminate_zeros()

    # Extract subgraph of thresholded vertices
    subgraph = expanded_adj[threshold_indices, :][:, threshold_indices]

    # Find connected components
    n_components, labels = connected_components(csgraph=subgraph, directed=False, return_labels=True)

    # Group into clusters
    clusters = []
    for i in range(n_components):
        cluster_indices = np.where(labels == i)[0]
        clusters.append([threshold_indices[idx] for idx in cluster_indices])

    return clusters

def sort_clusters_by_metrics(clusters, vertex_values, sort_by='combined', 
                              weight_size=0.5, weight_mean=0.5):
    """
    Trier les clusters selon leur taille et/ou leurs valeurs moyennes.
    
    Parameters
    ----------
    clusters : list or np.ndarray
        Liste des clusters retournés par find_clusters().
        Chaque cluster est un array d'indices de vertices.
    vertex_values : np.ndarray
        Valeurs pour chaque vertex (ex: layer_ts_mean).
        Shape: (n_vertices,)
    sort_by : str, optional
        Critère de tri:
        - 'size': trier par taille (nombre de vertices)
        - 'mean': trier par valeur moyenne
        - 'combined': combinaison pondérée des deux (default)
    weight_size : float, optional
        Poids pour la taille dans le score combiné (default: 0.5)
    weight_mean : float, optional
        Poids pour la valeur moyenne dans le score combiné (default: 0.5)
    
    Returns
    -------
    sorted_clusters : np.ndarray
        Clusters triés par ordre décroissant
    sorted_indices : np.ndarray
        Indices de tri (pour accéder aux métriques originales)
    metrics : dict
        Dictionnaire contenant:
        - 'sizes': tailles de chaque cluster (ordre original)
        - 'means': valeurs moyennes de chaque cluster (ordre original)
        - 'sorted_sizes': tailles triées
        - 'sorted_means': valeurs moyennes triées
    """
    if len(clusters) == 0:
        raise ValueError("No clusters provided")
    
    # Calculer les métriques pour chaque cluster
    sizes = np.array([len(clust) for clust in clusters])
    means = np.array([np.mean(vertex_values[clust]) for clust in clusters])
    
    # Trier selon le critère choisi
    if sort_by == 'size':
        sorted_indices = np.argsort(sizes)[::-1]
        
    elif sort_by == 'mean':
        sorted_indices = np.argsort(means)[::-1]
        
    elif sort_by == 'combined':
        # Normaliser les métriques entre 0 et 1
        size_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10)
        mean_norm = (means - means.min()) / (means.max() - means.min() + 1e-10)
        
        # Score combiné
        combined_score = weight_size * size_norm + weight_mean * mean_norm
        sorted_indices = np.argsort(combined_score)[::-1]
        
    else:
        raise ValueError(f"sort_by must be 'size', 'mean', or 'combined', got '{sort_by}'")
    
    # Créer l'array de clusters triés
    sorted_clusters = np.array(clusters, dtype=object)[sorted_indices]
    
    # Préparer les métriques
    metrics = {
        'sizes': sizes,
        'means': means,
        'sorted_sizes': sizes[sorted_indices],
        'sorted_means': means[sorted_indices],
    }
    
    return sorted_clusters, sorted_indices, metrics