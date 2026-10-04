// Scan-to-map registration: robust, noise-weighted point-to-plane
// Gauss-Newton with an optional motion prior, and the degeneracy analysis of
// its normal equations.
//
// Why point-to-plane and not KISS-ICP's point-to-point: with point-to-point
// the translation block of the Hessian is sum(w) * I whatever the scene
// looks like, so a camera staring at one flat wall gets a perfectly
// well-conditioned answer that is wrong along the wall. Point-to-plane's
// translation block is sum(w n n^T): one wall leaves it rank 1, the wall and
// the floor rank 2, and the missing direction is exactly the motion the
// camera cannot see. That is what degeneracy() reports.
//
// Why a prior: in a direction the scan cannot observe, plain ICP returns
// whatever its initial guess plus noise happened to be, with nothing to say
// so. Adding the motion prediction as a Gaussian prior (the IMU's rotation,
// constant velocity's translation) makes the solve well-posed, makes the
// unobservable directions follow the prediction instead of the noise, and
// makes the posterior covariance (H + Lambda)^-1 honest in both cases: the
// scan's information where it has some, the prior's uncertainty where it
// does not. The prior's information is small next to a good scan's (a few
// 1e4 against ~1e7), so on a feature-rich frame it does not bias the answer.
//
// Each residual is a point-to-plane distance weighted by 1/sigma^2, sigma^2
// = the source point's depth-noise variance plus the map voxel's variance,
// times a Geman-McClure factor on the normalised residual. Correspondences
// further than max_correspondence_m (+ 3 sigma) off the plane are dropped.
//
// No ROS. Parameterisation in lie.hpp.

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "dsim_slam/lie.hpp"
#include "dsim_slam/voxel_map.hpp"

namespace dsim_slam
{

/// A point to register, in the BODY frame, with its measurement variance.
struct SourcePoint
{
  Eigen::Vector3d p;
  double variance;   // m^2
};

struct RegistrationConfig
{
  int max_iterations {20};
  /// Stop when the update is below this, in metres and radians. 0.1 mm is
  /// two orders below the depth noise; at 1e-6 noisy frames ran all their
  /// iterations chasing nothing and the mean frame cost doubled.
  double convergence_eps {1e-4};
  /// Geman-McClure scale on the normalised residual r / sigma.
  double kernel_sigmas {3.0};
  /// Degeneracy-aware solve (solution remapping, Zhang et al. 2016): in any
  /// direction whose weight-normalised information is below these (see
  /// Degeneracy), the SCAN's information is projected out and the prior
  /// alone decides. Without it, facing a wall, near-random normals on the
  /// noisy wall gave a weak but real pull back towards where the map was
  /// built, stronger than the constant-velocity prior: the estimate stopped
  /// moving along the wall and fell 5 m behind in 10 s. Zero disables.
  double degenerate_translation_eig {0.0};
  double degenerate_rotation_eig {0.0};
  /// Judge observability on wide-support normals (VoxelMap::judgeNormal)
  /// instead of the solve's own patch normals. OFF by default, measured both
  /// ways (README): ON, a single wall 2.2 m away is correctly reported
  /// degenerate (narrow normals scatter ~10 deg there and fake 4-7 % of
  /// information along the wall); but small objects rarely support a 1 m
  /// plane fit, so the feature-rich survey loses real directions to the
  /// prior and drifts 4.3 %/m instead of ~1.2 %/m.
  bool judge_wide_normals {false};
  /// Hysteresis: translation directions (map frame, unit) that were
  /// degenerate on the previous frame stay projected out until the scan's
  /// normalised information along them exceeds release_factor times the
  /// threshold. Without it, one frame where noisy normals pushed the
  /// along-wall information just over the line let that noise in, with the
  /// filter's (by then very uncertain) velocity taking the whole kick.
  std::vector<Eigen::Vector3d> sticky_translation_dirs {};
  double release_factor {3.0};
};

/// A Gaussian prior on the pose: information matrix in the [rho; phi]
/// convention of lie.hpp. Zero information means no prior.
struct PosePrior
{
  Eigen::Isometry3d pose {Eigen::Isometry3d::Identity()};
  Matrix6d information {Matrix6d::Zero()};
};

struct RegistrationResult
{
  Eigen::Isometry3d pose {Eigen::Isometry3d::Identity()};
  /// The SCAN's information (data term only), at the final pose. Units of
  /// 1/m^2 and 1/rad^2 in the [rho; phi] convention.
  Matrix6d hessian {Matrix6d::Zero()};
  /// Posterior covariance (H * chi2-deflation + prior)^-1. Directions with no
  /// information from either side are capped at 1e6, not infinity.
  Matrix6d covariance {Matrix6d::Identity() * 1e6};
  int correspondences {0};       // matched in the final iteration
  int iterations {0};
  bool converged {false};
  double rms_residual_m {0.0};   // over final correspondences, unweighted
  /// Weighted chi^2 per degree of freedom of the final residuals. Near 1 if
  /// the noise model is right; when larger the covariance is inflated by it.
  double chi2_per_dof {0.0};
  double weight_sum {0.0};
  /// The information observability -- the verdict, the covariance, which
  /// directions the prior decides -- is judged on, at the final pose: the
  /// solve's own information, or the same matches and weights with
  /// wide-support normals (judge_wide_normals). Normalise by
  /// judge_weight_sum.
  Matrix6d judge_hessian {Matrix6d::Zero()};
  double judge_weight_sum {0.0};
  /// How many directions were judged unobservable at the final pose (0..6).
  int projected_directions {0};
  /// The translation directions removed, map frame (feed back as sticky).
  std::vector<Eigen::Vector3d> projected_translation_dirs {};
};

/// Register `source` (body frame) against `map`, starting from `initial`
/// (body -> map). `prior` may be nullptr.
RegistrationResult registerScan(
  const std::vector<SourcePoint> & source, const VoxelMap & map,
  const Eigen::Isometry3d & initial, double max_correspondence_m,
  const PosePrior * prior, const RegistrationConfig & cfg);

/// How well a scan constrains each direction of motion.
struct Degeneracy
{
  /// Smallest eigenvalue of the weight-normalised translation block
  /// sum(w n n^T) / sum(w). Dimensionless, in [0, 1/3]: 1/3 is surfaces
  /// facing every way equally, 0 is a direction no surface faces (one wall,
  /// or a wall and the floor).
  double translation_min_eig {0.0};
  /// The weakest translation direction, map frame (unit; sign arbitrary).
  Eigen::Vector3d translation_weak_dir {Eigen::Vector3d::Zero()};
  /// Smallest eigenvalue of the weight-normalised rotation block after the
  /// translation is marginalised out (Schur complement), in m^2: roughly the
  /// squared lever arm a rotation error acts through. 0 when a rotation can
  /// hide in the scene (yaw over a floor seen alone, any turn inside a
  /// cylinder about its axis). Only meaningful if translation is observable;
  /// computed with a tiny ridge otherwise.
  double rotation_min_eig {0.0};
  Eigen::Vector3d rotation_weak_dir {Eigen::Vector3d::Zero()};
};

Degeneracy analyseDegeneracy(const Matrix6d & hessian, double weight_sum);

}  // namespace dsim_slam
