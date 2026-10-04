#include "dsim_slam/registration.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

namespace dsim_slam
{

namespace
{

/// Solve A x = b for symmetric PSD A, in the directions A actually
/// constrains. A plain LDLT on a rank-deficient A (no prior, one wall)
/// returns huge values along the null space; this returns zero there.
Vector6d solvePsd(const Matrix6d & A, const Vector6d & b)
{
  Eigen::SelfAdjointEigenSolver<Matrix6d> es(A);
  const auto & lam = es.eigenvalues();
  const double floor = std::max(1e-12, lam.maxCoeff() * 1e-12);
  Vector6d x = Vector6d::Zero();
  for (int i = 0; i < 6; ++i) {
    if (lam(i) > floor) {
      const auto v = es.eigenvectors().col(i);
      x += v * (v.dot(b) / lam(i));
    }
  }
  return x;
}

/// Inverse of a symmetric PSD matrix with unconstrained directions capped
/// at `cap` variance rather than infinity.
Matrix6d cappedInverse(const Matrix6d & A, double cap)
{
  Eigen::SelfAdjointEigenSolver<Matrix6d> es(A);
  Matrix6d out = Matrix6d::Zero();
  for (int i = 0; i < 6; ++i) {
    const double var = es.eigenvalues()(i) > 1.0 / cap ? 1.0 / es.eigenvalues()(i) : cap;
    const auto v = es.eigenvectors().col(i);
    out += var * v * v.transpose();
  }
  return out;
}

struct Accumulated
{
  Matrix6d H {Matrix6d::Zero()};
  Vector6d g {Vector6d::Zero()};
  int n {0};
  double sum_r2 {0.0};        // unweighted
  double sum_chi2 {0.0};      // r^2 / sigma^2
  double weight_sum {0.0};
  // Observability information: the same weights with each match's WIDE
  // normal (VoxelMap::judgeNormal); only filled when asked for.
  Matrix6d H_judge {Matrix6d::Zero()};
  double weight_sum_judge {0.0};
};

Accumulated linearise(
  const std::vector<SourcePoint> & source, const VoxelMap & map,
  const Eigen::Isometry3d & T, double max_corr, double kernel_sigmas, bool judge = false,
  bool wide_judge = false)
{
  Accumulated a;
  const Eigen::Vector3d c = T.translation();
  const double k2 = kernel_sigmas * kernel_sigmas;
  for (const auto & s : source) {
    const Eigen::Vector3d p = T * s.p;
    const PlaneMatch m = map.query(p, s.variance);
    if (!m.found) {continue;}
    const double r = m.normal.dot(p - m.point);
    const double var = s.variance + m.point_variance;
    if (std::abs(r) > max_corr + 3.0 * std::sqrt(var)) {continue;}
    // d r / d [rho; phi]: moving the point by rho moves it along n by n.rho;
    // rotating it by phi about the body position moves it by phi x (p - c),
    // whose component along n is phi . ((p - c) x n).
    Vector6d J;
    J.head<3>() = m.normal;
    J.tail<3>() = (p - c).cross(m.normal);
    // Geman-McClure, scale in METRES: KISS-ICP's tau / 3 (how wrong the
    // prediction is expected to be) combined with this point's own noise.
    // A scale in sigmas alone (r / sigma against 3) gives a near, precise
    // point zero weight until the pose is already within a few of ITS
    // sigmas -- 3 cm at 2 m -- so from a 10 cm prediction error only the far,
    // noisy points steered and Gauss-Newton crawled: 30 iterations and a
    // 3.6 cm error on two noise-free renders.
    const double kappa2 = k2 * var + max_corr * max_corr / 9.0;
    const double u2 = r * r / var;
    const double gm = kappa2 / (kappa2 + r * r);
    const double robust = gm * gm;
    const double w = robust / var;
    a.H.noalias() += w * J * J.transpose();
    a.g.noalias() += w * r * J;
    a.n += 1;
    a.sum_r2 += r * r;
    a.sum_chi2 += u2;
    a.weight_sum += w;
    Eigen::Vector3d nj = m.normal;
    if (!judge) {continue;}
    if (wide_judge && !map.judgeNormal(p, s.variance, m, &nj)) {continue;}
    {
      Vector6d Jj;
      Jj.head<3>() = nj;
      Jj.tail<3>() = (p - c).cross(nj);
      a.H_judge.noalias() += w * Jj * Jj.transpose();
      a.weight_sum_judge += w;
    }
  }
  return a;
}

/// Directions the scan does not observe (see RegistrationConfig), judged on
/// `H_judge` / `weight_sum` exactly as analyseDegeneracy() reports them, plus
/// last frame's translation directions that have not clearly come back.
struct Directions
{
  std::vector<Vector6d> all;                 // orthonormal, 6-D
  std::vector<Eigen::Vector3d> translation;  // the translation ones, 3-D
};

Directions findDegenerate(
  const Matrix6d & H_judge, double weight_sum, const RegistrationConfig & cfg)
{
  Directions out;
  if (cfg.degenerate_translation_eig <= 0.0 && cfg.degenerate_rotation_eig <= 0.0) {
    return out;
  }
  // No matches at all: nothing is observed.
  const Matrix6d Hn = weight_sum > 0.0 ? Matrix6d(H_judge / weight_sum) : Matrix6d::Zero();
  const Eigen::Matrix3d Hn_tt = Hn.topLeftCorner<3, 3>();
  std::vector<Eigen::Vector3d> cand;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> et(Hn_tt);
  for (int i = 0; i < 3; ++i) {
    if (et.eigenvalues()(i) < cfg.degenerate_translation_eig) {
      cand.push_back(et.eigenvectors().col(i));
    }
  }
  for (const auto & u : cfg.sticky_translation_dirs) {
    if (u.dot(Hn_tt * u) < cfg.release_factor * cfg.degenerate_translation_eig) {
      cand.push_back(u.normalized());
    }
  }
  for (auto v : cand) {     // orthonormalise, so overlapping ones count once
    for (const auto & b : out.translation) {v -= b * b.dot(v);}
    if (v.norm() < 0.3) {continue;}
    v.normalize();
    out.translation.push_back(v);
    Vector6d u = Vector6d::Zero();
    u.head<3>() = v;
    out.all.push_back(u);
  }
  const Eigen::Matrix3d Htt = Hn_tt + 1e-9 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = Hn.bottomRightCorner<3, 3>() -
    Hn.bottomLeftCorner<3, 3>() * Htt.ldlt().solve(Hn.topRightCorner<3, 3>());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> er(0.5 * (S + S.transpose()));
  for (int i = 0; i < 3; ++i) {
    if (er.eigenvalues()(i) < cfg.degenerate_rotation_eig) {
      Vector6d u = Vector6d::Zero();
      u.tail<3>() = er.eigenvectors().col(i);
      out.all.push_back(u);
    }
  }
  return out;
}

/// Remove the scan's information (and gradient) along `d`: there the prior
/// alone decides.
void project(const Directions & d, Matrix6d * H, Vector6d * g)
{
  if (d.all.empty()) {return;}
  Matrix6d P = Matrix6d::Identity();
  for (const auto & u : d.all) {P -= u * u.transpose();}
  *H = P * (*H) * P;
  if (g != nullptr) {*g = P * (*g);}
}

}  // namespace

RegistrationResult registerScan(
  const std::vector<SourcePoint> & source, const VoxelMap & map,
  const Eigen::Isometry3d & initial, double max_correspondence_m,
  const PosePrior * prior, const RegistrationConfig & cfg)
{
  RegistrationResult res;
  Eigen::Isometry3d T = initial;
  // Which directions does this scan observe? Judged once at the starting
  // pose and held for the solve; in the directions judged unobservable the
  // prior decides (solution remapping, Zhang et al. 2016).
  Accumulated a = linearise(source, map, T, max_correspondence_m, cfg.kernel_sigmas, true,
      cfg.judge_wide_normals);
  const Directions d0 = findDegenerate(a.H_judge, a.weight_sum_judge, cfg);
  for (int it = 0; it < cfg.max_iterations; ++it) {
    if (it > 0) {a = linearise(source, map, T, max_correspondence_m, cfg.kernel_sigmas);}
    Matrix6d A = a.H;
    Vector6d g = a.g;
    project(d0, &A, &g);
    Vector6d b = -g;
    if (prior != nullptr) {
      const Vector6d e = poseError(T, prior->pose);
      A += prior->information;
      b -= prior->information * e;
    }
    const Vector6d dx = solvePsd(A, b);
    T = applyUpdate(T, dx);
    res.iterations = it + 1;
    if (dx.head<3>().norm() < cfg.convergence_eps && dx.tail<3>().norm() < cfg.convergence_eps) {
      res.converged = true;
      break;
    }
  }
  // Re-linearise at the final pose so the reported information, residuals
  // and correspondence count describe the pose actually returned, and
  // re-judge observability there for the verdict and the covariance.
  a = linearise(source, map, T, max_correspondence_m, cfg.kernel_sigmas, true,
      cfg.judge_wide_normals);
  const Directions d1 = findDegenerate(a.H_judge, a.weight_sum_judge, cfg);
  res.pose = T;
  res.hessian = a.H;
  res.correspondences = a.n;
  res.weight_sum = a.weight_sum;
  res.judge_hessian = a.H_judge;
  res.judge_weight_sum = a.weight_sum_judge;
  res.projected_directions = static_cast<int>(d1.all.size());
  res.projected_translation_dirs = d1.translation;
  res.rms_residual_m = a.n > 0 ? std::sqrt(a.sum_r2 / a.n) : 0.0;
  res.chi2_per_dof = a.n > 6 ? a.sum_chi2 / (a.n - 6) : 0.0;
  // If the residuals are bigger than the noise model says, the model is
  // optimistic (correlated stereo errors, map blur, unmodelled geometry):
  // deflate the scan's information by that factor rather than report a
  // covariance the data itself contradicts. Never inflate it.
  const double deflate = 1.0 / std::max(1.0, res.chi2_per_dof);
  Matrix6d info = a.H;
  project(d1, &info, nullptr);
  info *= deflate;
  if (prior != nullptr) {info += prior->information;}
  res.covariance = cappedInverse(info, 1e6);
  return res;
}

Degeneracy analyseDegeneracy(const Matrix6d & hessian, double weight_sum)
{
  Degeneracy d;
  if (!(weight_sum > 0.0)) {return d;}
  const Matrix6d Hn = hessian / weight_sum;
  const Eigen::Matrix3d Htt = Hn.topLeftCorner<3, 3>();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> et(Htt);
  d.translation_min_eig = std::max(0.0, et.eigenvalues()(0));
  d.translation_weak_dir = et.eigenvectors().col(0);

  // Rotation with translation marginalised: Hrr - Hrt Htt^-1 Htr. A small
  // ridge keeps Htt invertible when translation is itself degenerate; the
  // translation verdict has already said so.
  const Eigen::Matrix3d Htt_r = Htt + 1e-9 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = Hn.bottomRightCorner<3, 3>() -
    Hn.bottomLeftCorner<3, 3>() * Htt_r.ldlt().solve(Hn.topRightCorner<3, 3>());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> er(0.5 * (S + S.transpose()));
  d.rotation_min_eig = std::max(0.0, er.eigenvalues()(0));
  d.rotation_weak_dir = er.eigenvectors().col(0);
  return d;
}

}  // namespace dsim_slam
