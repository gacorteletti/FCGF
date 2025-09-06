import open3d as o3d
import numpy as np
import logging


def add_noise(pcd: o3d.geometry.PointCloud, seed: int = 42, fixed: bool = True,
                       sigma: float = 0.0, sigma_max: float = 0.05,
                       spike_ratio: float = 0.0, spike_min: float = 0.2, spike_max: float = 1.0, spike_skew: float = 2.0,
                       pepper_ratio: float = 0.0) -> o3d.geometry.PointCloud:
    """
    Returns a copy of `pcd` with:
      1) Gaussian noise (fixed or varying sigma)
      2) Optional spike noise on a given percentage of points
      3) Optional "pepper" noise: randomly remove a percentage of points

    Modes:
      - fixed=True:  all points get noise ~N(0, sigma²)
      - fixed=False: each point i draws sigma_i ~ Uniform[`sigma`, `sigma_max`], then noise ~N(0, sigma_ᵢ²).
      - sigma=0.0 AND fixed=True: no Gaussian noise
      - sigma=0.0 AND sigma_max=0.0 AND fixed=False: no Gaussian noise
      - spike_ratio=0.0: no spike readings
      - spike_ratio>0.0: then given (ideally small) ratio of points is affected by a huge error
      - pepper_ratio=0.0: no points removed
      - pepper_ratio>0.0: fraction of points is randomly dropped

    Args:
        pcd:           input Open3D PointCloud
        seed:          RNG seed for reproducibility
        fixed:         whether to use a single global sigma (True) or per-point sigma (False)
        sigma:         lower‐bound (or sole) standard deviation
        sigma_max:     upper‐bound standard deviation when fixed=False
        spike_ratio:   fraction of points to turn into spikes (0.0 – 1.0)
        spike_min      minimum spike magnitude
        spike_max      maximum spike magnitude
        spike_skew     exponent to skew magnitude distribution (>1 to produce more small spikes)
        pepper_ratio:  fraction of points to randomly remove (0.0–1.0)

    Returns:
        A new Open3D PointCloud with noisy points.
    """

    # 1) Convert points to an (N,3) NumPy array
    pts = np.asarray(pcd.points)
    N = pts.shape[0]                # number of points

    # 2) Fix random seed
    rng = np.random.default_rng(seed)

    # 3) Generate Gaussian noise
    if fixed:
        if sigma > 0.0:
            # Sample noise ~ N(0, sigma^2) for each coordinate of each point
            noise = rng.normal(loc=0.0, scale=sigma, size=(N, 3))
        else:
            noise = np.zeros((N,3))
    else:
        if sigma_max > 0.0:
            # Sample a sigma for each point
            sigmas_arr = rng.uniform(sigma, sigma_max, size=(N,1))  # shape (N,1)
            # Use the varying sigma to generate the noise
            noise = rng.normal(loc=0.0, scale=sigmas_arr, size=(N,3))
        else:
            noise = np.zeros((N,3))
    pts_noisy = pts + noise

    # 4) Generate Spike Noise
    if spike_ratio > 0.0:
        n_spikes = int(np.floor(spike_ratio * N))                   # number of spikes to be produced
        spike_idxs = rng.choice(N, size=n_spikes, replace=False)    # randomly select indices of points to be affected

        # Gerate variable magnitude
        u = rng.random(n_spikes)                                    # draw from uniform distribution from 0.0 to 1.0
        mags = spike_min + (spike_max - spike_min)*(u**spike_skew)  # magnitude between the given boundaries with a positive skew
        mags = mags.reshape(-1,1)                                   # shape=(n_spikes,1)

        # Create unit‐length random directions for each spike
        directions = rng.normal(size=(n_spikes, 3))                 # generate a n_spikes x 3 matrix where each row is a random 3D vector
        norms = np.linalg.norm(directions, axis=1, keepdims=True)   # compute the euclidian length (L2 norm) of each row (axis=1) keeping the n_spikes x 3 dimension
        directions = directions/norms                               # convert the random vectors to unit vector

        # Scale to the magnitude
        spike_offsets = directions*mags

        # Inject spikes
        pts_noisy[spike_idxs] += spike_offsets

    # Salt-and-pepper noise
    if pepper_ratio > 0.0:
        min_points_kept = 1000
        target_drop   = int(np.floor(pepper_ratio * N))             # compute number of points we want to "turned off"
        max_drop = max(0, N - min_points_kept)                      # but never drop so many that <min_points_kept remain:
        n_drop   = min(target_drop, max_drop)                       # cap the drop at a certain maximum
        if n_drop < target_drop:
            logging.warning(
                f"pepper_ratio={pepper_ratio:.3f} would drop {target_drop} pts,"
                f"clamping to {n_drop} to keep ≥{min_points_kept} points."
            )
        drop_idx = rng.choice(N, size=n_drop, replace=False)        # randomly select which points will be 'turned off'
        mask = np.ones(N, dtype=bool)                               # initialize an all-True mask with the length equal to the number of points
        mask[drop_idx] = False                                      # set the mask to False at the selected points

        pts_noisy = pts_noisy[mask]                                 # apply the mask to the points

        # Drop corresponding attributes (if any)
        colors = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None
        normals = np.asarray(pcd.normals)[mask] if pcd.has_normals() else None

    # If no salt-and-pepper noise, preserve all attributes (if any)
    else:
        colors  = np.asarray(pcd.colors)  if pcd.has_colors()  else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    # 6) Create a new point cloud (so the original remains unchanged)
    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(pts_noisy)
    if colors is not None:
        noisy_pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        noisy_pcd.normals = o3d.utility.Vector3dVector(normals)

    return noisy_pcd



def read_pcd(pcd_path: str, **noise_kwargs) -> o3d.geometry.PointCloud:
    """
    Loads a point cloud from a given file path and (optionally) apply noise.
    Check `add_noise` documentation for details on noise settings
    
    Args:
        pcd_path:         path to the file containing the point cloud data
        **noise_kwargs:   passed directly to `add_noise`, which supports:
                            - fixed, sigma, sigma_max,
                            - spike_ratio, spike_min, spike_max, spike_skew,
                            - pepper_ratio.

    Returns:
        The loaded (and possibly noised) point cloud.
    """

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = add_noise(pcd, **noise_kwargs)    
    return pcd