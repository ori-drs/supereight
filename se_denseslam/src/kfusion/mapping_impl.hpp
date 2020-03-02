/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 *
 * */
#ifndef KFUSION_MAPPING_HPP
#define KFUSION_MAPPING_HPP
#include <se/node.hpp>

struct sdf_update {

  template <typename DataHandlerT>
  void operator()(DataHandlerT& handler, const Eigen::Vector3i&, 
      const Eigen::Vector3f& pos, const Eigen::Vector2f& pixel) {

    const Eigen::Vector2i px = pixel.cast<int>();
    const float depthSample = depth[px(0) + depthSize(0)*px(1)];
    if (depthSample <=  0) return;
    const float diff = (depthSample - pos(2)) 
      * std::sqrt( 1 + se::math::sq(pos(0) / pos(2)) + se::math::sq(pos(1) / pos(2)));
    if (diff > -mu) {
      const float sdf = fminf(1.f, diff / mu);
      auto data = handler.get();
      data.x = se::math::clamp(
          (static_cast<float>(data.y) * data.x + sdf) / (static_cast<float>(data.y) + 1.f), 
          -1.f,
          1.f);
      data.y = fminf(data.y + 1, maxweight);
      handler.set(data);
    }
  } 

  sdf_update(const float * d, const Eigen::Vector2i framesize, float m, int mw) : 
    depth(d), depthSize(framesize), mu(m), maxweight(mw){};

  const float * depth;
  Eigen::Vector2i depthSize;
  float mu;
  int maxweight;
};

struct sdf_update_cloud {

  template <typename DataHandlerT>
  void operator()(DataHandlerT& handler, const Eigen::Vector3i&, const Eigen::Vector3f& pos, const Eigen::Vector2f& pixel) {
    // pos - the coordinate of a voxel within a block in camera frame
    // pixel - the pixel coordinate of a voxel within a block in picture frame

    Eigen::Vector3f ray_unit = pos.normalized();

    std::vector<double> angles;
    angles.resize(units.size());
#pragma omp parallel for
    for (int i = 0; i < units.size(); i++){
      angles[i] = units[i].dot(ray_unit);
    }

    int max_index = std::distance(angles.begin(), max_element(angles.begin(), angles.end()));

    const float depthSample = depths[max_index];

    if (std::acos(angles[max_index])/M_PI*180.0 > 0.3){
      return;
    }

    if (depthSample <=  0) {
      return;
    }

    const float diff = (depthSample - pos(2)) * std::sqrt( 1 + se::math::sq(pos(0) / pos(2)) + se::math::sq(pos(1) / pos(2)));
    if (diff > -mu) {
      const float sdf = fminf(1.f, diff / mu);
      auto data = handler.get();
      data.x = se::math::clamp((static_cast<float>(data.y) * data.x + sdf) / (static_cast<float>(data.y) + 1.f), -1.f, 1.f);
      data.y = fminf(data.y + 1, maxweight);
      handler.set(data);
    } 
  } 

  sdf_update_cloud(const std::vector<Eigen::Vector3f> cloud, const Eigen::Matrix4f pose, const Eigen::Vector2i framesize, float m, int mw) : 
                  points(cloud), 
                  lidar_pose(pose), 
                  depthSize(framesize), 
                  mu(m), 
                  maxweight(mw)
  {
    depths.resize(points.size());
    units.resize(points.size());

    Eigen::Isometry3f sensor_pose;
    sensor_pose.matrix() = lidar_pose;

    for (int i = 0; i < points.size(); i ++){
      Eigen::Vector3f local_vector = (pose.inverse() * points[i].homogeneous()).head<3>();

      depths[i] = local_vector.norm();
      units[i] = local_vector.normalized();

    }

  };

  std::vector<Eigen::Vector3f> points;
  Eigen::Matrix4f lidar_pose;
  std::vector<float> depths;
  std::vector<Eigen::Vector3f> units;

  Eigen::Vector2i depthSize;
  float mu;
  int maxweight;
};

#endif
