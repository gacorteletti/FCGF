import open3d as o3d
import os
import logging
import numpy as np
import random

from util.trajectory import CameraPose
from util.pointcloud import compute_overlap_ratio, \
    make_open3d_point_cloud, make_open3d_feature_from_numpy


def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size):
  distance_threshold = voxel_size * 1.5
  result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      xyz0, xyz1, feat0, feat1, True, distance_threshold,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
      ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1.0))
  return result_ransac.transformation


def gather_results(results):
  traj = []
  for r in results:
    success = r[0]
    if success:
      traj.append(CameraPose([r[1], r[2], r[3]], r[4]))
  return traj


def gen_matching_pair(pts_num, source_path=None, scene=None, subset=None):
  matching_pairs = []
  if not subset:
    for i in range(pts_num):
      for j in range(i + 2, pts_num): #<<<<<<<<<<<<< change +1 -> +2 to consider only non-consecutive pairs
        matching_pairs.append([i, j, pts_num])
  else:
    gt_path = os.path.join(source_path, '%s-evaluation' %scene, 'gt.log')
    with open(gt_path, 'r') as f:
      for idx, line in enumerate(f):
        line = line.replace('\n', '').replace('\t', '').split()
        if (idx%5==0) and (int(line[1])-int(line[0])>=2):
          matching_pairs.append([int(line[0]), int(line[1]), subset])
      n_comb = (subset-1)*(subset-2)/2
      matching_pairs = random.sample(matching_pairs, k=n_comb)
  with open('matching_pairs.txt', 'a') as out:
    out.write(f"Set: {scene}\n")
    for pair in matching_pairs:
      out.write(f"{pair[0]} {pair[1]}\n")
  return matching_pairs


def read_data(feature_path, name):
  data = np.load(os.path.join(feature_path, name + ".npz"))
  xyz = make_open3d_point_cloud(data['xyz'])
  feat = make_open3d_feature_from_numpy(data['feature'])
  return data['points'], xyz, feat


def do_single_pair_matching(feature_path, set_name, m, voxel_size):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  logging.info("\t\tMatching %03d %03d" % (i, j))
  points_i, xyz_i, feat_i = read_data(feature_path, name_i)
  points_j, xyz_j, feat_j = read_data(feature_path, name_j)
  if len(xyz_i.points) < len(xyz_j.points):
    trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
  else:
    trans = run_ransac(xyz_j, xyz_i, feat_j, feat_i, voxel_size)
    trans = np.linalg.inv(trans)
  ratio = compute_overlap_ratio(xyz_i, xyz_j, trans, voxel_size)
  logging.info("\t\t\tOverlap Ratio: %.4f" % (ratio))
  if ratio > 0.3:
    return [True, i, j, s, np.linalg.inv(trans)]
  else:
    return [False, i, j, s, np.identity(4)]
