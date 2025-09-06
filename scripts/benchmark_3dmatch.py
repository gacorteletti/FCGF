"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d

# Add the parent directory (../FCGF/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.timer import Timer, AverageMeter

from util.misc import extract_features

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results

import torch

import MinkowskiEngine as ME

# import the timer decorator defined in the other file
from scripts.timer_decorator import timer, total_stage_times

# import the noise generator defined in the other file
from scripts.noise_generator import read_pcd


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

@timer("preprocessing")
def extract_features_batch(model, config, source_path, target_path, voxel_size, device, **noise_kwargs):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  logging.info(folders)
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  logging.info("==================================== FEATURE EXTRACTION ====================================")

  for fo in folders:
    if 'evaluation' in fo:
      continue
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file
      pcd = read_pcd(fi, **noise_kwargs)
      save_fn = "%s_%03d" % (fo_base, i)
      if i % 100 == 0:
        logging.info(f"{i} / {len(files)}: {save_fn}")

      timer.tic()
      xyz_down, feature = extract_features(
          model,
          xyz=np.array(pcd.points),
          rgb=None,
          normal=None,
          voxel_size=voxel_size,
          device=device,
          skip_check=True)
      t = timer.toc()
      if i > 0:
        tmeter.update(t)
        num_feat += len(xyz_down)

      np.savez_compressed(
          os.path.join(target_path, save_fn),
          points=np.array(pcd.points),
          xyz=xyz_down,
          feature=feature.detach().cpu().numpy())
      if i % 20 == 0 and i > 0:
        logging.info(
            f'\t\tAverage Time: {tmeter.avg}  |  FPS: {num_feat / tmeter.sum}  |  Time / #Features: {tmeter.sum / num_feat}, '
        )

  f.close()


def registration(feature_path, voxel_size, **noise_kwargs):
  """
  Gather .log files produced in --target folder and run this Matlab script
  https://github.com/andyzeng/3dmatch-toolbox#geometric-registration-benchmark
  (see Geometric Registration Benchmark section in
  http://3dmatch.cs.princeton.edu/)
  """
  
  logging.info("======================================= REGISTRATION =======================================")
  if args.subset:
    logging.info(f'======================================== Subset: {args.subset} =========================================')
  else:
    logging.info(f'==================================== Whole Test Split ======================================')
  
  output_folder = '/'.join(args.target.split('/')[:-1])
  registration_path = f"{output_folder}/registration"
  log_path = f"{registration_path}/initial_guesses_logs"
  ensure_dir(log_path)

  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  with open(f'{registration_path}/matching_pairs.txt', 'w') as out:
    out.write("")
  
  # Set Open3D's verbosity level to Debug to capture detailed iteration information
  o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
  
  for s in sets:
    set_name = s[0]
    pts_num = int(s[1]) 
    matching_pairs = gen_matching_pair(pts_num, args.source, set_name, args.subset, registration_path)  # for now, limit the test split to a subset of 5 clouds per scene (to save time)
    results = []                                                                                        # to run all split, remove additiona subset parameter or set it to False (default)

    logging.info("Set: %s" % (set_name))

    for m in matching_pairs:
      results.append(do_single_pair_matching(feature_path, set_name, m, voxel_size, args.inlier_th, **noise_kwargs))
    traj = gather_results(results)
    logging.info(f"Writing the trajectory to {log_path}/{set_name}_FCGF.log")
    write_trajectory(traj, "%s_FCGF.log" % (os.path.join(log_path, set_name)))


  # Returns Open3D's verbosity level to default mode
  o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def do_single_pair_evaluation(feature_path,
                              set_name,
                              traj,
                              voxel_size,
                              tau_1=0.1,
                              tau_2=0.05,
                              num_rand_keypoints=-1):
  trans_gth = np.linalg.inv(traj.pose)
  i = traj.metadata[0]
  j = traj.metadata[1]
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)

  # coord and feat form a sparse tensor.
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

  # use the keypoints in 3DMatch
  if num_rand_keypoints > 0:
    # Randomly subsample N points
    Ni, Nj = len(points_i), len(points_j)
    inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
    inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

    sample_i, sample_j = points_i[inds_i], points_j[inds_j]

    key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))
    key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

    key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))
    key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

    inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]
    inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

    coord_i, feat_i = coord_i[inds_i], feat_i[inds_i]
    coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

  coord_i = make_open3d_point_cloud(coord_i)
  coord_j = make_open3d_point_cloud(coord_j)

  hit_ratio = evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth,
                                       tau_1)

  # logging.info(f"Hit ratio of {name_i}, {name_j}: {hit_ratio}, {hit_ratio >= tau_2}")
  if hit_ratio >= tau_2:
    return True
  else:
    return False


def feature_evaluation(source_path, feature_path, voxel_size, num_rand_keypoints=-1):
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]

  assert len(
      sets
  ) > 0, "Empty list file. Makesure to run the feature extraction first with --do_extract_feature."

  tau_1 = 0.1  # 10cm
  tau_2 = 0.05  # 5% inlier
  logging.info("==================================== FEATURE EVALUATION ====================================")
  logging.info("==== Inlier Distance Threshold (tau_1): %.3f  |  Inlier Recall Threshold (tau_2): %.3f ===" % (tau_1, tau_2))
  recall = []
  for s in sets:
    set_name = s[0]
    traj = read_trajectory(os.path.join(source_path, set_name + "-evaluation/gt.log")) #<<<<<<<<<<<<<<<<<<<<<<
    assert len(traj) > 0, "Empty trajectory file"
    results = []
    for i in range(len(traj)):
      results.append(
          do_single_pair_evaluation(feature_path, set_name, traj[i], voxel_size, tau_1,
                                    tau_2, num_rand_keypoints))

    mean_recall = np.array(results).mean()
    std_recall = np.array(results).std()
    recall.append([set_name, mean_recall, std_recall])
    logging.info(f'\t{set_name}: {mean_recall} +- {std_recall}')
  #for r in recall:
  #  logging.info("%s : %.4f" % (r[0], r[1]))
  scene_r = np.array([r[1] for r in recall])
  logging.info("Average FMR: %.4f +- %.4f" % (scene_r.mean(), scene_r.std()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source', default=None, type=str, help='path to 3dmatch test dataset')
  parser.add_argument(
      '--source_high_res',
      default=None,
      type=str,
      help='path to high_resolution point cloud')
  parser.add_argument(
      '--target', default=None, type=str, help='path to produce generated data')
  parser.add_argument(
      '-m',
      '--model',
      default=None,
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.05,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument('--extract_features', action='store_true')
  parser.add_argument('--evaluate_feature_match_recall', action='store_true')
  parser.add_argument(
      '--evaluate_registration',
      action='store_true',
      help='The target directory must contain extracted features')
  parser.add_argument('--with_cuda', action='store_true')
  parser.add_argument(
      '--num_rand_keypoints',
      type=int,
      default=5000,
      help='Number of random keypoints for each scene')
  parser.add_argument(
      '--subset',
      default=None,
      type=int,
      help='Number of point clouds per scene to be considered. If you want to use the complete test split, remove this flag')
  parser.add_argument(
      '--inlier_th',
      default=None,
      type=float,
      help='Value for the threhsold of the maximum distance two corresponding points can have after the alignment for it to be considered an inlier')
  parser.add_argument(
      '--seed',
      default=None,
      type=int,
      help='Noise RNG seed for reproducibility'
  )
  parser.add_argument(
    '--fixed',
    action='store_true',
    help='If set, use a single global sigma (default)'
  )
  parser.add_argument(
    '--no-fixed',
    dest='fixed',
    action='store_false',
    help='Use per-point sigma (variable noise)'
  )
  parser.set_defaults(fixed=True)
  parser.add_argument(
      '--sigma',
      default=None,
      type=float,
      help='lower‐bound (or sole) standard deviation'
  )
  parser.add_argument(
      '--sigma_max',
      default=None,
      type=float,
      help='upper-bound standard deviation when fixed=False'
  )
  parser.add_argument(
      '--spike_ratio',
      default=None,
      type=float,
      help='fraction of points to turn into spikes (0.0 – 1.0)'
  )
  parser.add_argument(
      '--spike_min',
      default=None,
      type=float,
      help='minimum spike magnitude'
  )
  parser.add_argument(
      '--spike_max',
      default=None,
      type=float,
      help='maximum spike magnitude'
  )
  parser.add_argument(
      '--spike_skew',
      default=None,
      type=float,
      help='exponent to skew magnitude distribution (>1 to produce more small spikes)'
  )
  parser.add_argument(
      '--pepper_ratio',
      default=None,
      type=float,
      help='fraction of points to randomly remove (0.0–1.0)'
  )

  args = parser.parse_args()

  # Auxiliary dictionary with all possible settings the user might specify
  aux_full_noise_dict = {'seed': args.seed,
                         'fixed': args.fixed,
                         'sigma': args.sigma,
                         'sigma_max': args.sigma_max,
                         'spike_ratio': args.spike_ratio,
                         'spike_min': args.spike_min,
                         'spike_max': args.spike_max,
                         'spike_skew': args.spike_skew,
                         'pepper_ratio': args.pepper_ratio
                        }
  # Then we filter out arguments that were not defined by the user (i.e. were set to None as default)
  # This prevents overwriting the default values of the add_noise or read_pcd function with None
  noise_kwargs = {k:v for k,v in aux_full_noise_dict.items() if v is not None}

  device = torch.device('cuda' if args.with_cuda else 'cpu')

  if args.extract_features:
    assert args.model is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    with torch.no_grad():
      extract_features_batch(model, config, args.source, args.target, config.voxel_size,
                             device, **noise_kwargs)

  if args.evaluate_feature_match_recall:
    assert (args.target is not None)
    with torch.no_grad():
      feature_evaluation(args.source, args.target, args.voxel_size,
                         args.num_rand_keypoints)

  if args.evaluate_registration:
    assert (args.target is not None)
    with torch.no_grad():
      registration(args.target, args.voxel_size, **noise_kwargs)


  # Print the time results (only preprocessing and ransac, as icp will be done later by the main script)
  print("\n=== Total stage timings (benchmark_3dmatch.py) ===")
  for stage, t in total_stage_times.items():
    print(f"{stage} total time = {t:.5f}s")
