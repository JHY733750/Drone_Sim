#pragma once
// Static landmarks from YAML. Separate from depthcam.hpp because it needs
// yaml-cpp, which the host the mutation harness runs on does not have; the
// ray casting it feeds is tested without it.
//
// Format (one list, every entry fully specified, no defaults but yaw):
//
//   landmarks:
//     - {name: L0, type: box, center: [x, y, z], size: [sx, sy, sz], yaw: 0.3}
//     - {name: L1, type: cylinder, center: [x, y, z], radius: 0.4, height: 2.0}
//     - {name: L2, type: sphere, center: [x, y, z], radius: 0.5}
//
// A cylinder's axis is vertical and its centre is at MID-height (so a 2 m
// cylinder standing on the ground has center z = 1). A box's yaw is radians
// about world z, right-handed; it is optional and defaults to 0.
//
// Anything malformed is an exception naming the entry, never a skipped
// landmark: a missing obstacle in a depth image looks exactly like free space.
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "dsim_depthcam/geometry.hpp"

namespace dsim_depthcam
{

namespace detail
{
inline Eigen::Vector3d vec3(const YAML::Node & e, const char * key, const std::string & who)
{
  const auto n = e[key];
  if (!n || !n.IsSequence() || n.size() != 3) {
    throw std::runtime_error("landmark " + who + ": '" + key + "' must be a list of 3 numbers");
  }
  return Eigen::Vector3d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>());
}

inline double num(const YAML::Node & e, const char * key, const std::string & who)
{
  const auto n = e[key];
  if (!n || !n.IsScalar()) {
    throw std::runtime_error("landmark " + who + ": missing number '" + key + "'");
  }
  return n.as<double>();
}
}  // namespace detail

/// Parse landmarks from an already-loaded YAML document.
inline std::vector<Primitive> landmarksFromYaml(const YAML::Node & doc)
{
  std::vector<Primitive> out;
  if (!doc || doc.IsNull()) {return out;}
  const auto list = doc["landmarks"];
  if (!list || list.IsNull()) {return out;}
  if (!list.IsSequence()) {throw std::runtime_error("'landmarks' must be a list");}
  for (size_t i = 0; i < list.size(); ++i) {
    const auto e = list[i];
    const std::string who = e["name"] ? e["name"].as<std::string>() : "#" + std::to_string(i);
    if (!e["type"]) {throw std::runtime_error("landmark " + who + ": missing 'type'");}
    const std::string type = e["type"].as<std::string>();
    const Eigen::Vector3d c = detail::vec3(e, "center", who);
    if (type == "box") {
      const double yaw = e["yaw"] ? e["yaw"].as<double>() : 0.0;
      out.push_back(makeBox(c, detail::vec3(e, "size", who), yaw, who));
    } else if (type == "cylinder") {
      out.push_back(makeCylinder(
          c, detail::num(e, "radius", who), detail::num(e, "height", who), who));
    } else if (type == "sphere") {
      out.push_back(makeSphere(c, detail::num(e, "radius", who), who));
    } else {
      throw std::runtime_error(
              "landmark " + who + ": unknown type '" + type + "' (box | cylinder | sphere)");
    }
  }
  return out;
}

/// Load landmarks from a file. An empty path means "no landmarks"; a path
/// that does not open is an error, not an empty world.
inline std::vector<Primitive> loadLandmarks(const std::string & path)
{
  if (path.empty()) {return {};}
  YAML::Node doc;
  try {
    doc = YAML::LoadFile(path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("landmarks file '" + path + "': " + e.what());
  }
  return landmarksFromYaml(doc);
}

}  // namespace dsim_depthcam
