#include "dsim_slam/voxel_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace dsim_slam
{

VoxelMap::VoxelMap(const VoxelMapConfig & cfg)
: cfg_(cfg)
{
  if (!(cfg_.voxel_size_m > 0.0) || !(cfg_.max_voxel_sigma_m > 0.0) ||
    !(cfg_.min_voxel_sigma_m > 0.0) || cfg_.min_voxel_sigma_m > cfg_.max_voxel_sigma_m)
  {
    throw std::invalid_argument("VoxelMap: voxel size and sigma bounds must be positive and ordered");
  }
  min_weight_ = 1.0 / (cfg_.max_voxel_sigma_m * cfg_.max_voxel_sigma_m);
  max_weight_ = 1.0 / (cfg_.min_voxel_sigma_m * cfg_.min_voxel_sigma_m);
}

VoxelMap::Key VoxelMap::keyOf(const Eigen::Vector3d & p) const
{
  return Key{
    static_cast<int32_t>(std::floor(p.x() / cfg_.voxel_size_m)),
    static_cast<int32_t>(std::floor(p.y() / cfg_.voxel_size_m)),
    static_cast<int32_t>(std::floor(p.z() / cfg_.voxel_size_m))};
}

void VoxelMap::insert(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & variances)
{
  if (points.size() != variances.size()) {
    throw std::invalid_argument("VoxelMap::insert: one variance per point");
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!points[i].allFinite() || !(variances[i] > 0.0)) {continue;}
    const double w = 1.0 / variances[i];
    Voxel & v = voxels_[keyOf(points[i])];
    v.weighted_sum += w * points[i];
    v.weight += w;
    if (v.weight > max_weight_) {
      // Scale both, so the mean is unchanged and only its certainty is capped.
      const double s = max_weight_ / v.weight;
      v.weighted_sum *= s;
      v.weight = max_weight_;
    }
  }
}

void VoxelMap::removeFar(const Eigen::Vector3d & centre)
{
  const double r2 = cfg_.max_distance_m * cfg_.max_distance_m;
  for (auto it = voxels_.begin(); it != voxels_.end(); ) {
    if ((it->second.mean() - centre).squaredNorm() > r2) {
      it = voxels_.erase(it);
    } else {
      ++it;
    }
  }
}

namespace
{
struct Fit
{
  Eigen::Vector3d centroid;
  Eigen::Vector3d lam;       // ascending
  Eigen::Matrix3d axes;      // columns: normal, mid, major
  double wsum;
};

/// Weighted PCA of the means picked by `idx`.
template<class Cands>
Fit fitPlane(const Cands & c, const int * idx, int n)
{
  Fit f;
  f.wsum = 0.0;
  f.centroid.setZero();
  for (int i = 0; i < n; ++i) {f.wsum += c[idx[i]].w; f.centroid += c[idx[i]].w * c[idx[i]].m;}
  f.centroid /= f.wsum;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d d = c[idx[i]].m - f.centroid;
    cov += c[idx[i]].w * d * d.transpose();
  }
  cov /= f.wsum;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
  es.computeDirect(cov);
  f.lam = es.eigenvalues();
  f.axes = es.eigenvectors();
  return f;
}
}  // namespace

PlaneMatch VoxelMap::query(const Eigen::Vector3d & p, double point_variance) const
{
  PlaneMatch out;
  const Key k = keyOf(p);
  struct Cand
  {
    Eigen::Vector3d m;
    double w;
    double d2;
  };
  std::array<Cand, 27> c;
  int n = 0;
  int nearest = -1;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        const auto it = voxels_.find(Key{k.x + dx, k.y + dy, k.z + dz});
        if (it == voxels_.end() || it->second.weight < min_weight_) {continue;}
        const Eigen::Vector3d m = it->second.mean();
        c[n] = Cand{m, it->second.weight, (m - p).squaredNorm()};
        if (nearest < 0 || c[n].d2 < c[nearest].d2) {nearest = n;}
        ++n;
      }
    }
  }
  if (n < cfg_.min_neighbours) {return out;}
  // A query whose nearest mean is more than a voxel away belongs to some
  // other surface.
  if (c[nearest].d2 > cfg_.voxel_size_m * cfg_.voxel_size_m) {return out;}

  // ---- stage 1: the ORIENTATION, from the k means nearest p --------------
  // Not the whole 27-voxel neighbourhood: one that spans two surfaces (a
  // box side and the floor under it) tilts the fit, and the residual then
  // grows with how far p sits from the anchor ALONG the plane -- a 10 deg
  // tilt at 30 cm is 5 cm at the true pose.
  std::array<int, 27> order;
  for (int i = 0; i < n; ++i) {order[i] = i;}
  const int kn = std::min(n, cfg_.plane_points);
  std::partial_sort(order.begin(), order.begin() + kn, order.begin() + n,
    [&c](int a, int b) {return c[a].d2 < c[b].d2;});
  const Fit f1 = fitPlane(c, order.data(), kn);
  if (!(f1.lam.sum() > 0.0) || f1.lam(1) < 0.05 * f1.lam.sum()) {return out;}   // collinear
  const Eigen::Vector3d n1 = f1.axes.col(0);

  // ---- stage 2: the POSITION, from a window that does not follow p's noise
  // "Nearest to p" picks means on the same side of the surface as p's own
  // depth error, so an anchor built from them follows the noise and the
  // residual shrinks towards zero. Stereo noise is along the ray, which for
  // a surface facing the camera is along its normal -- the direction of
  // travel -- so registration under-reacted to forward motion: a 15 cm
  // lag in the first half second of the survey. Re-select every mean near
  // p ALONG the plane (VoxelMapConfig::window_radius_sigmas) and within a
  // noise-scaled band ACROSS it (window_band_sigmas), a window wide enough
  // that p's noise barely moves it.
  const double sp = std::sqrt(std::max(0.0, point_variance));
  const double band = cfg_.max_plane_distance_m + cfg_.window_band_sigmas * sp;
  const double radius = std::max(cfg_.voxel_size_m, cfg_.window_radius_sigmas * sp);
  const double r2 = radius * radius;
  std::array<int, 27> sel;
  int ns = 0;
  // The band is centred on p, not on the stage-1 plane: centring it on the
  // plane lets the anchor follow the stage-1 fit's own noise (the forward
  // overshoot grew from 22 % to 34 %, and survey drift over four noise seeds
  // from 1.8 to 3.1 %/m). The cost: a point further off its surface than the
  // band finds no match, which bounds the pull-in range by the band.
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d d = c[i].m - p;
    const double across = n1.dot(d);
    if (std::abs(across) <= band && (d - n1 * across).squaredNorm() <= r2) {sel[ns++] = i;}
  }
  if (ns < cfg_.min_neighbours) {return out;}
  const Fit f = fitPlane(c, sel.data(), ns);
  const double sum = f.lam.sum();
  if (!(sum > 0.0) || f.lam(1) < 0.05 * sum) {return out;}
  // Thickness against width: sqrt(lambda0 / lambda1) is roughly the angle
  // by which the normal is uncertain. Beyond ~3 m the D435's depth noise
  // scatters a surface over a slab as thick as the patch is wide, and such
  // a patch's "normal" is near-random: accepting it gave spurious
  // information in every direction, which both hid a wall's degeneracy and
  // pinned the pose along it.
  if (f.lam(0) > cfg_.max_thickness_ratio * cfg_.max_thickness_ratio * f.lam(1)) {return out;}
  const Eigen::Vector3d nrm = f.axes.col(0);
  // Every mean must lie on the plane to within the tolerance plus its own
  // uncertainty: an edge or a corner fails here instead of producing a
  // normal that belongs to neither face.
  for (int i = 0; i < ns; ++i) {
    const auto & ci = c[sel[i]];
    // A mean's scatter about the surface is NOT its own 1/sqrt(weight),
    // which shrinks with every re-observation: noise spreads a far surface
    // over neighbouring voxels, and those means sit about a point-sigma off
    // it however often they are seen. Hence the larger of the two.
    const double tol = cfg_.max_plane_distance_m +
      cfg_.plane_sigma_factor * std::max(1.0 / std::sqrt(ci.w), sp);
    if (std::abs(nrm.dot(ci.m - f.centroid)) > tol) {return out;}
  }
  // The plane is only known where its means are. A query outside the
  // patch's own extent -- along EITHER in-plane axis -- is extrapolating it:
  // on a thin cylinder the patch is a tall, narrow strip, and a point 14 cm
  // round the curve was 7 cm off its plane at the TRUE pose. (An isotropic
  // radius let that through: the strip's height hid its narrowness.)
  double ext1 = 0.0, ext2 = 0.0;
  for (int i = 0; i < ns; ++i) {
    const Eigen::Vector3d d = c[sel[i]].m - f.centroid;
    ext1 = std::max(ext1, std::abs(f.axes.col(1).dot(d)));
    ext2 = std::max(ext2, std::abs(f.axes.col(2).dot(d)));
  }
  const Eigen::Vector3d dq = p - f.centroid;
  if (std::abs(f.axes.col(1).dot(dq)) > ext1 || std::abs(f.axes.col(2).dot(dq)) > ext2) {
    return out;
  }

  out.found = true;
  out.point = f.centroid;
  out.normal = nrm;
  // The anchor's uncertainty along the normal: the means' scatter about the
  // plane plus the centroid's own variance.
  out.point_variance = f.lam(0) + 1.0 / f.wsum;
  return out;
}

bool VoxelMap::judgeNormal(
  const Eigen::Vector3d & p, double point_variance, const PlaneMatch & m,
  Eigen::Vector3d * normal) const
{
  if (!m.found) {return false;}
  const double sp = std::sqrt(std::max(0.0, point_variance));
  const double band = cfg_.max_plane_distance_m + cfg_.window_band_sigmas * sp;
  const double r2 = cfg_.judge_radius_m * cfg_.judge_radius_m;
  const int reach = static_cast<int>(std::ceil(cfg_.judge_radius_m / cfg_.voxel_size_m));
  const Key k = keyOf(p);
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  Eigen::Matrix3d s = Eigen::Matrix3d::Zero();
  int n = 0;
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      for (int dz = -reach; dz <= reach; ++dz) {
        const auto it = voxels_.find(Key{k.x + dx, k.y + dy, k.z + dz});
        if (it == voxels_.end() || it->second.weight < min_weight_) {continue;}
        const Eigen::Vector3d q = it->second.mean();
        const Eigen::Vector3d d = q - m.point;
        const double across = m.normal.dot(d);
        if (std::abs(across) > band || (d - m.normal * across).squaredNorm() > r2) {continue;}
        // Unweighted: the point is the means' SPREAD, and near, heavily
        // weighted means would otherwise shrink it back to a small patch.
        c += q;
        s += q * q.transpose();
        ++n;
      }
    }
  }
  if (n < 3 * cfg_.min_neighbours) {return false;}
  c /= n;
  const Eigen::Matrix3d cov = s / n - c * c.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
  es.computeDirect(cov);
  const Eigen::Vector3d lam = es.eigenvalues();
  if (!(lam(1) > 0.0) || lam(0) > cfg_.max_thickness_ratio * cfg_.max_thickness_ratio * lam(1)) {
    return false;
  }
  *normal = es.eigenvectors().col(0);
  return true;
}

std::size_t VoxelMap::confirmedSize() const
{
  std::size_t n = 0;
  for (const auto & kv : voxels_) {
    if (kv.second.weight >= min_weight_) {++n;}
  }
  return n;
}

std::vector<Eigen::Vector3d> VoxelMap::points() const
{
  std::vector<Eigen::Vector3d> out;
  out.reserve(voxels_.size());
  for (const auto & kv : voxels_) {out.push_back(kv.second.mean());}
  return out;
}

}  // namespace dsim_slam
