// The local map: a sparse voxel hash holding ONE inverse-variance-weighted
// mean point per voxel.
//
// Why a weighted mean rather than KISS-ICP's "first N raw points per voxel":
// a stereo camera's depth noise grows as z^2 (at 424x240 about 2 cm at 2 m
// and 18 cm at 5 m), so raw far points are mostly noise. Averaging with
// weights 1/sigma^2 lets a near observation dominate a voxel it shares with
// far ones, shrinks the noise by sqrt(n) as a surface is re-observed, and
// leaves the mean ON the surface (a convex combination of points on a plane
// lies on that plane), which is what point-to-plane needs. A voxel's total
// weight is also an honest "how well do I know this", so a voxel seen once
// from 6 m can be kept but refused as a correspondence until it is
// confirmed (max_voxel_sigma_m).
//
// No ROS. Points are in the map frame.

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace dsim_slam
{

struct VoxelMapConfig
{
  double voxel_size_m {0.25};
  /// Voxels whose mean is further than this from the vehicle are dropped
  /// every frame. It bounds memory and makes this a LOCAL map: no loop
  /// closure, so old geometry only carries old drift.
  double max_distance_m {12.0};
  /// A voxel is used as a correspondence only once its mean is known to
  /// better than this (1/sqrt(total weight)). Below that it is kept but
  /// treated as unconfirmed.
  double max_voxel_sigma_m {0.10};
  /// Weight cap, as a sigma floor: once a voxel's mean is this certain it
  /// stops getting more certain, so it can still follow slow map drift
  /// instead of freezing for ever.
  double min_voxel_sigma_m {0.004};
  /// Plane fit (see query()): fitted to the plane_points confirmed means
  /// nearest the query, of which there must be at least min_neighbours, all
  /// within max_plane_distance_m (+ 2 sigma of each mean) of the plane.
  int min_neighbours {4};
  int plane_points {6};
  double max_plane_distance_m {0.03};
  double plane_sigma_factor {3.0};
  /// sqrt(lambda_min / lambda_mid) of the patch: ~ the normal's angular
  /// uncertainty in radians.
  double max_thickness_ratio {0.25};
  /// The stage-2 window scales with the query point's noise sigma: radius
  /// max(voxel, window_radius_sigmas sigma) along the plane, depth band
  /// max_plane_distance + window_band_sigmas sigma across it; and each mean
  /// may sit plane_sigma_factor * max(its own sigma, the point's sigma) off
  /// the fitted plane. A far surface is a slab ~2 sigma thick (noise spreads
  /// it over neighbouring voxels, whatever their weights say), and only a
  /// patch several sigma WIDE has a usable normal: with one-voxel patches a
  /// single wall at 3.4 m looked observable along itself. The price: wide
  /// patches straddle junctions, which biases motion towards surfaces (see
  /// the frame test and README).
  double window_radius_sigmas {4.0};
  double window_band_sigmas {3.0};
  /// judgeNormal(): the in-plane radius of the wide fit, metres.
  double judge_radius_m {0.5};
};

/// What a query found near a point: a local plane through the nearest
/// confirmed voxel means.
struct PlaneMatch
{
  bool found {false};
  Eigen::Vector3d point {Eigen::Vector3d::Zero()};    // the plane's anchor (weighted centroid)
  Eigen::Vector3d normal {Eigen::Vector3d::Zero()};   // unit, sign arbitrary
  double point_variance {0.0};                        // of the anchor along the normal, m^2
};

class VoxelMap
{
public:
  explicit VoxelMap(const VoxelMapConfig & cfg);

  /// Add points (map frame) each with its own measurement variance (m^2).
  void insert(const std::vector<Eigen::Vector3d> & points, const std::vector<double> & variances);

  /// Drop every voxel whose mean is further than max_distance_m from `centre`.
  void removeFar(const Eigen::Vector3d & centre);

  /// A local plane at p: fitted (weighted PCA) to the plane_points confirmed
  /// voxel means nearest p among the 27 voxels around it. Not found if there
  /// are fewer than min_neighbours, if they are collinear (no defined
  /// normal), or if any of them is off the fitted plane by more than
  /// max_plane_distance_m plus twice its own sigma (an edge or a corner).
  PlaneMatch query(const Eigen::Vector3d & p, double point_variance = 0.0) const;

  /// A normal for JUDGING observability (registration.hpp), refitted over
  /// every confirmed mean within judge_radius_m of p along the plane that
  /// query() found and within its noise band across it. A plane fitted to n
  /// means spread lambda in-plane, each a sigma off the surface, has normal
  /// variance ~ sigma^2 / (n lambda): with query()'s ~6 means over 0.2 m and
  /// the D435's sigma at 2.2 m (3.5 cm) that is ~10 deg, and ten degrees of
  /// normal scatter on a single wall looks like 4-7 % of its information
  /// lying ALONG the wall -- a wall reported observable while the pose slid
  /// along it. Some 30 means over 1 m bring it to ~1 deg. Too blunt for the
  /// residual (it would straddle junctions); right for counting directions.
  /// Returns false where no such plane exists.
  bool judgeNormal(
    const Eigen::Vector3d & p, double point_variance, const PlaneMatch & m,
    Eigen::Vector3d * normal) const;

  void clear() {voxels_.clear();}
  std::size_t size() const {return voxels_.size();}
  std::size_t confirmedSize() const;
  bool empty() const {return voxels_.empty();}
  const VoxelMapConfig & config() const {return cfg_;}

  /// Every voxel mean (confirmed or not). For visualisation and tests.
  std::vector<Eigen::Vector3d> points() const;

private:
  struct Key
  {
    int32_t x, y, z;
    bool operator==(const Key & o) const {return x == o.x && y == o.y && z == o.z;}
  };
  struct KeyHash
  {
    std::size_t operator()(const Key & k) const
    {
      // The classic spatial hash (Teschner et al. 2003), in unsigned
      // arithmetic so negative coordinates cannot overflow into UB.
      const uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(k.x)) * 73856093ULL) ^
        (static_cast<uint64_t>(static_cast<uint32_t>(k.y)) * 19349669ULL) ^
        (static_cast<uint64_t>(static_cast<uint32_t>(k.z)) * 83492791ULL);
      return static_cast<std::size_t>(h);
    }
  };
  struct Voxel
  {
    Eigen::Vector3d weighted_sum {Eigen::Vector3d::Zero()};
    double weight {0.0};
    Eigen::Vector3d mean() const {return weighted_sum / weight;}
  };

  Key keyOf(const Eigen::Vector3d & p) const;

  VoxelMapConfig cfg_;
  double min_weight_;   // 1 / max_voxel_sigma^2
  double max_weight_;   // 1 / min_voxel_sigma^2
  std::unordered_map<Key, Voxel, KeyHash> voxels_;
};

}  // namespace dsim_slam
