// Time per frame at the default resolution against a scene of ~30 primitives.
// A measurement, not a test: it prints and always exits 0. Built with the
// tests (BUILD_TESTING) but not registered with ctest, because a timing bound
// that holds on this machine is a flaky failure on a loaded one.
//
//   /tmp/b/dsim_depthcam/bench_depthcam
#include <chrono>
#include <cstdio>

#include "dsim_depthcam/depthcam.hpp"

using V = Eigen::Vector3d;

int main()
{
  dsim_depthcam::World w;      // ground: 1
  // fence3d's thirteen moving spheres, laid out as in config/obstacles_fence3d.yaml.
  for (int i = 0; i < 11; ++i) {
    w.obstacles.push_back(dsim_depthcam::obstacleFromFlat(
        "F" + std::to_string(i), "sphere", 0.9,
        {4.0, 0.5 + 0.9 * i - 4.5, 0.0, 0.0, 6.5, 0.5 + 0.9 * i - 4.5, 0.0, 10.0}));
  }
  w.obstacles.push_back(dsim_depthcam::obstacleFromFlat("M1", "column", 0.4, {8, -2, 1.5, 0, 8, 2, 1.5, 10}));
  w.obstacles.push_back(dsim_depthcam::obstacleFromFlat("M2", "column", 0.4, {9, 2, 1.5, 0, 9, -2, 1.5, 10}));
  // Sixteen landmarks, a mix of the three types: 1 + 13 + 16 = 30.
  for (int i = 0; i < 16; ++i) {
    const V c(3.0 + 0.5 * i, (i % 2 ? 3.0 : -3.0), 1.0);
    if (i % 3 == 0) {
      w.landmarks.push_back(dsim_depthcam::makeBox(c, V(0.6, 0.6, 2.0), 0.1 * i));
    } else if (i % 3 == 1) {
      w.landmarks.push_back(dsim_depthcam::makeCylinder(c, 0.3, 2.0));
    } else {
      w.landmarks.push_back(dsim_depthcam::makeSphere(c, 0.4));
    }
  }
  dsim_depthcam::CameraConfig cfg;
  cfg.dropout = 0.02;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = V(0, 0, 1.5);
  const auto Topt = dsim_depthcam::worldToOptical(T, dsim_depthcam::Mount{});

  const int frames = 60;
  size_t points = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < frames; ++i) {
    const auto scene = dsim_depthcam::sceneAt(w, 3.0 + i / 30.0);
    points += dsim_depthcam::render(scene, Topt, cfg, i).points.size();
  }
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count() / frames;
  std::printf(
    "%dx%d, %zu primitives: %.2f ms/frame (%.0f Hz possible), %.0f points/frame\n",
    cfg.width, cfg.height, dsim_depthcam::sceneAt(w, 3.0).size(), ms, 1000.0 / ms,
    double(points) / frames);
  return 0;
}
