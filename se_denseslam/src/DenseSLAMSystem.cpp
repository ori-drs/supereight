/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.


 Copyright 2016 Emanuele Vespa, Imperial College London

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <se/DenseSLAMSystem.h>
#include <se/ray_iterator.hpp>
#include <se/algorithms/meshing.hpp>
#include <se/geometry/octree_collision.hpp>
#include <se/vtk-io.h>
#include "timings.h"
#include <perfstats.h>
#include "preprocessing.cpp"
#include "tracking.cpp"
#include "rendering.cpp"
#include "bfusion/mapping_impl.hpp"
#include "kfusion/mapping_impl.hpp"
#include "bfusion/alloc_impl.hpp"
#include "kfusion/alloc_impl.hpp"

#include <iostream>

extern PerfStats Stats;
static bool print_kernel_timing = false;

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& inputSize,
                                 const Eigen::Vector3i& volumeResolution,
                                 const Eigen::Vector3f& volumeDimensions,
			                           const Eigen::Vector3f& initPose,
                                 std::vector<int> & pyramid,
                                 const Configuration& config):
      DenseSLAMSystem(inputSize, volumeResolution, volumeDimensions,
          se::math::toMatrix4f(initPose), pyramid, config) { }

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& inputSize,
                                 const Eigen::Vector3i& volumeResolution,
                                 const Eigen::Vector3f& volumeDimensions,
                                 const Eigen::Matrix4f& initPose,
                                 std::vector<int> & pyramid,
                                 const Configuration& config) :
  computation_size_(inputSize),
  vertex_(computation_size_.x(), computation_size_.y()),
  normal_(computation_size_.x(), computation_size_.y()),
  float_depth_(computation_size_.x(), computation_size_.y())
  {

    this->init_pose_ = initPose.block<3,1>(0,3);
    this->volume_dimension_ = volumeDimensions;
    this->volume_resolution_ = volumeResolution;
    this->mu_ = config.mu;
    pose_ = initPose;
    raycast_pose_ = initPose;

    this->iterations_.clear();
    for (std::vector<int>::iterator it = pyramid.begin();
        it != pyramid.end(); it++) {
      this->iterations_.push_back(*it);
    }

    viewPose_ = &pose_;

    if (getenv("KERNEL_TIMINGS"))
      print_kernel_timing = true;

    // internal buffers to initialize
    reduction_output_.resize(8 * 32);
    tracking_result_.resize(computation_size_.x() * computation_size_.y());

    for (unsigned int i = 0; i < iterations_.size(); ++i) {
      int downsample = 1 << i;
      scaled_depth_.push_back(se::Image<float>(computation_size_.x() / downsample,
            computation_size_.y() / downsample));

      input_vertex_.push_back(se::Image<Eigen::Vector3f>(computation_size_.x() / downsample,
            computation_size_.y() / downsample));

      input_normal_.push_back(se::Image<Eigen::Vector3f>(computation_size_.x() / downsample,
            computation_size_.y() / downsample));
    }

    // ********* BEGIN : Generate the gaussian *************
    size_t gaussianS = radius * 2 + 1;
    gaussian_.reserve(gaussianS);
    int x;
    for (unsigned int i = 0; i < gaussianS; i++) {
      x = i - 2;
      gaussian_[i] = expf(-(x * x) / (2 * delta * delta));
    }

    // ********* END : Generate the gaussian *************

    discrete_vol_ptr_ = std::make_shared<se::Octree<FieldType> >();
    discrete_vol_ptr_->init(volume_resolution_.x(), volume_dimension_.x());
    volume_ = Volume<FieldType>(volume_resolution_.x(), volume_dimension_.x(),
        discrete_vol_ptr_.get());

    lidar_k_ = Eigen::Matrix4f::Identity();
    frame_ = 1;
}

bool DenseSLAMSystem::preprocessing(const unsigned short * inputDepth,
    const Eigen::Vector2i& inputSize, const bool filterInput){

    mm2metersKernel(float_depth_, inputDepth, inputSize);
    if(filterInput){
        bilateralFilterKernel(scaled_depth_[0], float_depth_, gaussian_,
            e_delta, radius);
    }
    else {
      std::memcpy(scaled_depth_[0].data(), float_depth_.data(),
          sizeof(float) * computation_size_.x() * computation_size_.y());
    }
	return true;
}

bool DenseSLAMSystem::tracking(const Eigen::Vector4f& k,
    float icp_threshold, unsigned tracking_rate, unsigned frame) {

	if (frame % tracking_rate != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (unsigned int i = 1; i < iterations_.size(); ++i) {
		halfSampleRobustImageKernel(scaled_depth_[i], scaled_depth_[i - 1], e_delta * 3, 1);
	}

	// prepare the 3D information from the input depth maps
  Eigen::Vector2i localimagesize = computation_size_;
	for (unsigned int i = 0; i < iterations_.size(); ++i) {
    Eigen::Matrix4f invK = getInverseCameraMatrix(k / float(1 << i));
		depth2vertexKernel(input_vertex_[i], scaled_depth_[i], invK);
    if(k.y() < 0)
      vertex2normalKernel<true>(input_normal_[i], input_vertex_[i]);
    else
      vertex2normalKernel<false>(input_normal_[i], input_vertex_[i]);
		localimagesize /= 2;;
	}

	old_pose_ = pose_;
	const Eigen::Matrix4f projectReference = getCameraMatrix(k) * raycast_pose_.inverse();

	for (int level = iterations_.size() - 1; level >= 0; --level) {
    Eigen::Vector2i localimagesize(
				computation_size_.x() / (int) pow(2, level),
				computation_size_.y() / (int) pow(2, level));
		for (int i = 0; i < iterations_[level]; ++i) {

      trackKernel(tracking_result_.data(), input_vertex_[level], input_normal_[level],
          vertex_, normal_, pose_, projectReference,
          dist_threshold, normal_threshold);

			reduceKernel(reduction_output_.data(), tracking_result_.data(), computation_size_,
					localimagesize);

			if (updatePoseKernel(pose_, reduction_output_.data(), icp_threshold))
				break;

		}
	}
	return checkPoseKernel(pose_, old_pose_, reduction_output_.data(),
      computation_size_, track_threshold);
}

bool DenseSLAMSystem::raycasting(const Eigen::Vector4f& k, float mu, unsigned int frame) {

  bool doRaycast = false;

  if(frame > 2) {
    // raycast_pose_ = pose_;
    raycast_pose_ = sensor_pose_.matrix();

    float step = volume_dimension_.x() / volume_resolution_.x();

    // raycastKernel(volume_, vertex_, normal_,
    //     raycast_pose_ * getInverseCameraMatrix(k), nearPlane,
    //     farPlane, mu, step, step*BLOCK_SIDE);
    // Didn't quite figure out how to resize se::Image so I created two additional ones temporarily
    se::Image<Eigen::Vector3f> vertex_lidar(computation_size_.x(), computation_size_.y());
    se::Image<Eigen::Vector3f> normal_lidar(computation_size_.x(), computation_size_.y());
    raycastKernelLidar(volume_, vertex_lidar, normal_lidar, raycast_pose_ * lidar_k_.inverse(), nearPlane, farPlane, mu, step, step*BLOCK_SIDE);

    doRaycast = true;
  }
  return doRaycast;
}

bool DenseSLAMSystem::integration(const Eigen::Vector4f& k, unsigned int integration_rate,
    float mu, unsigned int frame) {

  if (((frame % integration_rate) == 0) || (frame <= 3)) {

    float voxelsize =  volume_._dim/volume_._size;
    int num_vox_per_pix = volume_._dim/((se::VoxelBlock<FieldType>::side)*voxelsize);
    // size_t total = num_vox_per_pix * computation_size_.x() *
    //   computation_size_.y();
    size_t total;
    if(std::is_same<FieldType, SDF>::value) {
      total = (4.0 * mu / voxelsize) * computation_size_.x() * computation_size_.y();
    } else if(std::is_same<FieldType, OFusion>::value) {
      total = (12.0 * mu / voxelsize) * computation_size_.x() * computation_size_.y();
    }
    allocation_list_.reserve(total);

    unsigned int allocated = 0;
    if(std::is_same<FieldType, SDF>::value) {
     // allocated  = buildAllocationList(allocation_list_.data(),
     //     allocation_list_.capacity(),
     //    *volume_._map_index, pose_, getCameraMatrix(k), float_depth_.data(),
     //    computation_size_, volume_._size,
     //  voxelsize, 2*mu);
      allocated = buildAllocationListCloud(allocation_list_.data(), allocation_list_.capacity(), *volume_._map_index, sensor_pose_.matrix(), points_, computation_size_, volume_._size, voxelsize, 2*mu);
      
    } else if(std::is_same<FieldType, OFusion>::value) {
     // allocated = buildOctantList(allocation_list_.data(), allocation_list_.capacity(),
     //     *volume_._map_index,
     //     pose_, getCameraMatrix(k), float_depth_.data(), computation_size_, voxelsize,
     //     compute_stepsize, step_to_depth, 6*mu);

      allocated = buildOctantListCloud(allocation_list_.data(), allocation_list_.capacity(), *volume_._map_index, sensor_pose_.matrix(), points_, computation_size_, voxelsize, compute_stepsize, step_to_depth, 6*mu);
    }

    volume_._map_index->allocate(allocation_list_.data(), allocated);

    if(std::is_same<FieldType, SDF>::value) {
      // struct sdf_update funct(float_depth_.data(),
      //     Eigen::Vector2i(computation_size_.x(), computation_size_.y()), mu, 100);
      // se::functor::projective_map(*volume_._map_index,
      //     Sophus::SE3f(pose_).inverse(),
      //     getCameraMatrix(k),
      //     Eigen::Vector2i(computation_size_.x(), computation_size_.y()),
      //     funct);

      struct sdf_update_cloud funct(points_, sensor_pose_.matrix(), Eigen::Vector2i(computation_size_.x(), computation_size_.y()), mu, 100);
      se::functor::projective_map(*volume_._map_index, Sophus::SE3f(sensor_pose_.matrix()).inverse(), lidar_k_, Eigen::Vector2i(computation_size_.x(), computation_size_.y()), funct);
    } else if(std::is_same<FieldType, OFusion>::value) {

      // float timestamp = (1.f/30.f)*frame;
      // struct bfusion_update funct(float_depth_.data(),
      //     Eigen::Vector2i(computation_size_.x(), computation_size_.y()), 
      //     mu, timestamp, voxelsize);

      // se::functor::projective_map(*volume_._map_index,
      //     Sophus::SE3f(pose_).inverse(),
      //     getCameraMatrix(k),
      //     Eigen::Vector2i(computation_size_.x(), computation_size_.y()),
      //     funct);

      float timestamp = (2.f)*frame;
      struct bfusion_update_cloud funct(points_, sensor_pose_.matrix(), Eigen::Vector2i(computation_size_.x(), computation_size_.y()), mu, timestamp, voxelsize);
      se::functor::projective_map(*volume_._map_index, Sophus::SE3f(sensor_pose_.matrix()).inverse(), lidar_k_, Eigen::Vector2i(computation_size_.x(), computation_size_.y()), funct);
    }

    // if(frame % 15 == 0) {
    //   std::stringstream f;
    //   f << "./slices/integration_" << frame << ".vtk";
    //   save3DSlice(*volume_._map_index, Eigen::Vector3i(0, 200, 0),
    //       Eigen::Vector3i(volume_._size, 201, volume_._size),
    //       Eigen::Vector3i::Constant(volume_._size), f.str().c_str());
    //   f.str("");
    //   f.clear();
    // }
  } else {
    return false;
  }
  return true;
}

void DenseSLAMSystem::dump_volume(std::string ) {

}

void DenseSLAMSystem::renderVolume(unsigned char* out,
    const Eigen::Vector2i& outputSize,
    int frame,
		int raycast_rendering_rate,
    const Eigen::Vector4f& k,
    float largestep) {

	if (frame % raycast_rendering_rate == 0) {
    const float step = volume_dimension_.x() / volume_resolution_.x();
		renderVolumeKernel(volume_, out, outputSize,
	*(this->viewPose_) * getInverseCameraMatrix(k), nearPlane,
	farPlane * 2.0f, mu_, step, largestep,
        this->viewPose_->topRightCorner<3, 1>(), ambient,
        !(this->viewPose_->isApprox(raycast_pose_)), vertex_,
        normal_);
  }
}

void DenseSLAMSystem::renderTrack(unsigned char* out,
    const Eigen::Vector2i& outputSize) {
        renderTrackKernel(out, tracking_result_.data(), outputSize);
}

void DenseSLAMSystem::renderDepth(unsigned char* out,
    const Eigen::Vector2i& outputSize) {
        renderDepthKernel(out, float_depth_.data(), outputSize, nearPlane, farPlane);
}

void DenseSLAMSystem::dump_mesh(const std::string filename){

  std::vector<Triangle> mesh;
  auto inside = [](const Volume<FieldType>::value_type& val) {
    // meshing::status code;
    // if(val.y == 0.f)
    //   code = meshing::status::UNKNOWN;
    // else
    //   code = val.x < 0.f ? meshing::status::INSIDE : meshing::status::OUTSIDE;
    // return code;
    // std::cerr << val.x << " ";
    return val.x < 0.f;
  };

  auto select = [](const Volume<FieldType>::value_type& val) {
    return val.x;
  };

  se::algorithms::marching_cube(*volume_._map_index, select, inside, mesh);
  writeVtkMesh(filename.c_str(), mesh);
}

void DenseSLAMSystem::readPcdFile(int frame){
  points_.clear();

  std::string cloud_name = std::to_string(frame);
  cloud_name = std::string(3 - cloud_name.length(), '0') + cloud_name;
  std::stringstream ss_cloud_filename;

  char const* home_directory = getenv("HOME");

  ss_cloud_filename << home_directory << "/object_scan_data/18-11-2019-radcliffe-yiduo/flipped/pointclouds/" << cloud_name << ".pcd";
  // ss_cloud_filename << home_directory << "/object_scan_data/test_point_cloud_4/" << cloud_name << ".pcd";

  std::ifstream cloud_file_stream(ss_cloud_filename.str(), std::ios::in);

  std::string line;
  int line_num = 0;
  int point_num = 0;
  while (std::getline(cloud_file_stream, line)) {
      line_num ++;

      std::istringstream iss(line);

      if (line_num <=11){
        std::string header_info;
        int header_content;
        if (!(iss >> header_info >> header_content)){
          continue;
        }
        if (header_info.compare("POINTS") != 0){
          continue;
        }
        // std::cout << "There are " << header_content << " points in this pcd file according to its header. \n";
      }

      float x, y, z;

      // std::string rgb; 
      // std::string holder0, holder1, holder2; 
      // if (!(iss >> x >> y >> z >> rgb >> holder0 >> holder1 >> holder2)) { 
      //   continue; 
      // }
      if (!(iss >> x >> y >> z)) { 
        continue; 
      }
      point_num ++;

      Eigen::Vector3f point(x, y, z);
      points_.push_back(point);
  }

  // std::cout << "Read " << line_num << " lines. \n";
  // std::cout << "There are " << point_num << "/" << line_num << " on points. \n";
  // std::cout << "There are " << points_.size() << " points in this pcd file according to content. \n";

  computation_size_.y() = points_.size();
}

void DenseSLAMSystem::readPoseFile(int frame){
  Eigen::Isometry3f sensor_pose = Eigen::Isometry3f::Identity();

  std::string cloud_name = std::to_string(frame);
  cloud_name = std::string(3 - cloud_name.length(), '0') + cloud_name;
  std::stringstream ss_pose_filename;

  char const* home_directory = getenv("HOME");

  ss_pose_filename << home_directory << "/object_scan_data/18-11-2019-radcliffe-yiduo/flipped/poses/" << cloud_name << ".txt";
  // ss_pose_filename << home_directory << "/object_scan_data/test_point_cloud_4/" << cloud_name << ".txt";

  std::ifstream pose_file_stream(ss_pose_filename.str(), std::ios::in);

  std::vector<double> tmp_sensor_pose; 
  float num = 0.0;
  while (pose_file_stream >> num) {
    tmp_sensor_pose.push_back(num);
  }

  Eigen::Vector3f translation(tmp_sensor_pose[0], tmp_sensor_pose[1], tmp_sensor_pose[2]);
  sensor_pose.translate(translation);

  Eigen::Quaternionf rotation(tmp_sensor_pose[3], tmp_sensor_pose[4], tmp_sensor_pose[5], tmp_sensor_pose[6]);
  sensor_pose.rotate(rotation);

  sensor_pose_ = sensor_pose;
  // std::cout << sensor_pose_.matrix() << "\n";
}

void DenseSLAMSystem::fullVolume(void){
  std::string cloud_name = std::to_string(frame_-1);
  cloud_name = std::string(3 - cloud_name.length(), '0') + cloud_name;
  std::stringstream ss_path_filename;

  char const* work_directory = getenv("PWD");

  ss_path_filename << work_directory << "/slices/" << cloud_name << ".csv";
  std::ofstream cloud_file;
  cloud_file.open(ss_path_filename.str());
  cloud_file << sensor_pose_.translation().x() << "," 
             << sensor_pose_.translation().y() << ","
             << sensor_pose_.translation().z() << ","
             << Eigen::Quaternionf(sensor_pose_.rotation()).w() << ","
             << Eigen::Quaternionf(sensor_pose_.rotation()).x() << ","
             << Eigen::Quaternionf(sensor_pose_.rotation()).y() << ","
             << Eigen::Quaternionf(sensor_pose_.rotation()).z() << ","
             << 0.0 << "\n";

  for (int x = 0; x < volume_._size; x++){
    for (int y = 0; y < volume_._size; y++){
      for (int z = 0; z < volume_._size; z++){
        Eigen::Vector3f this_coord = volume_.pos(Eigen::Vector3i(x, y, z));
    
        se::VoxelBlock<FieldType>* block = volume_._map_index->fetch(x, y, z);
        voxel_traits<FieldType>::value_type content = volume_.get(this_coord);

        if (content.x != 1.0 || content.y != 0.0){
          cloud_file << x << "," 
                     << y << ","
                     << z << ",";
          cloud_file << this_coord.x() << ","
                     << this_coord.y() << ","
                     << this_coord.z() << ",";
          cloud_file << content.x << "," << content.y << "\n";
        }

      }
    }
  }

  cloud_file.close();

}
