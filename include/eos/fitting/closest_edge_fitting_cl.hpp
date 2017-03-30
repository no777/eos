/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/closest_edge_fitting.hpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef CLOSESTEDGEFITTING_CL_HPP_
#define CLOSESTEDGEFITTING_CL_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/utils.hpp"

#include "nanoflann.hpp"

#include "glm/common.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include "Eigen/Dense" // Need only Vector2f actually.

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <vector>
#include <algorithm>
#include <utility>
#include <cstddef>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // eliminate build warning

#if __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif


#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/ocl.hpp"



#include <opencv2/core/ocl.hpp>


// 1. Add the af/opencl.h include to your project
//#include <af/opencl.h>




namespace eos {
	namespace fitting {
        
        struct clinfo{
            cl_platform_id platform;
            cl_context context;
            cl_command_queue queue;
            cl_device_id device;

        };
        

        int testcl();
        
        int initOpenCL();
        

/**
 * @brief Computes the intersection of the given ray with the given triangle.
 *
 * Uses the Möller-Trumbore algorithm algorithm "Fast Minimum Storage
 * Ray/Triangle Intersection". Independent implementation, inspired by:
 * http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
 * The default eps (1e-6f) is from the paper.
 * When culling is on, rays intersecting triangles from the back will be discarded -
 * otherwise, the triangles normal direction w.r.t. the ray direction is just ignored.
 *
 * Note: The use of optional might turn out as a performance problem, as this
 * function is called loads of time - how costly is it to construct a boost::none optional?
 *
 * @param[in] ray_origin Ray origin.
 * @param[in] ray_direction Ray direction.
 * @param[in] v0 First vertex of a triangle.
 * @param[in] v1 Second vertex of a triangle.
 * @param[in] v2 Third vertex of a triangle.
 * @param[in] enable_backculling When culling is on, rays intersecting triangles from the back will be discarded.
 * @return Whether the ray intersects the triangle, and if yes, including the distance.
 */
        inline std::pair<bool, boost::optional<float>> ray_triangle_intersect_cl(const glm::vec3* ray_origin, const glm::vec3* ray_direction, const glm::vec3* v0, const glm::vec3* v1, const glm::vec3* v2,int size, bool enable_backculling);
        

/**
 * @brief Computes the vertices that lie on occluding boundaries, given a particular pose.
 *
 * This algorithm computes the edges that lie on occluding boundaries of the mesh.
 * It performs a visibility text of each vertex, and returns a list of the (unique)
 * vertices that make the boundary edges.
 * An edge is defined as the line whose two adjacent faces normals flip the sign.
 *
 * @param[in] mesh The mesh to use.
 * @param[in] edge_topology The edge topology of the given mesh.
 * @param[in] R The rotation (pose) under which the occluding boundaries should be computed.
 * @return A vector with unique vertex id's making up the edges.
 */
         std::vector<int>  occluding_boundary_vertices_cl(const core::Mesh& mesh, const morphablemodel::EdgeTopology& edge_topology, glm::mat4x4 R);
//        
//{
//   
//    
//	// Rotate the mesh:
//	std::vector<glm::vec4> rotated_vertices;
//	std::for_each(begin(mesh.vertices), end(mesh.vertices), [&rotated_vertices, &R](auto&& v) { rotated_vertices.push_back(R * v); });
//
//	// Compute the face normals of the rotated mesh:
//	std::vector<glm::vec3> facenormals;
//	for (auto&& f : mesh.tvi) { // for each face (triangle):
//		auto n = render::compute_face_normal(glm::vec3(rotated_vertices[f[0]]), glm::vec3(rotated_vertices[f[1]]), glm::vec3(rotated_vertices[f[2]]));
//		facenormals.push_back(n);
//	}
//
//	// Find occluding edges:
//	std::vector<int> occluding_edges_indices;
//	for (int edge_idx = 0; edge_idx < edge_topology.adjacent_faces.size(); ++edge_idx) // For each edge... Ef contains the indices of the two adjacent faces
//	{
//		const auto& edge = edge_topology.adjacent_faces[edge_idx];
//		if (edge[0] == 0) // ==> NOTE/Todo Need to change this if we use 0-based indexing!
//		{
//			// Edges with a zero index lie on the mesh boundary, i.e. they are only
//			// adjacent to one face.
//			continue;
//		}
//		// Compute the occluding edges as those where the two adjacent face normals
//		// differ in the sign of their z-component:
//		// Changing from 1-based indexing to 0-based!
//		if (glm::sign(facenormals[edge[0] - 1].z) != glm::sign(facenormals[edge[1] - 1].z))
//		{
//			// It's an occluding edge, store the index:
//			occluding_edges_indices.push_back(edge_idx);
//		}
//	}
//	// Select the vertices lying at the two ends of the occluding edges and remove duplicates:
//	// (This is what EdgeTopology::adjacent_vertices is needed for).
//	std::vector<int> occluding_vertices; // The model's contour vertices
//	for (auto&& edge_idx : occluding_edges_indices)
//	{
//		// Changing from 1-based indexing to 0-based!
//		occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][0] - 1);
//		occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][1] - 1);
//	}
//	// Remove duplicate vertex id's (std::unique works only on sorted sequences):
//	std::sort(begin(occluding_vertices), end(occluding_vertices));
//	occluding_vertices.erase(std::unique(begin(occluding_vertices), end(occluding_vertices)), end(occluding_vertices));
//
//	// Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
//	std::vector<bool> visibility;
//	for (const auto& vertex_idx : occluding_vertices)
//	{
//		bool visible = true;
//		// For every tri of the rotated mesh:
//        int count =  mesh.tvi.size();
//		for (auto&& tri : mesh.tvi)
//		{
//			auto& v0 = rotated_vertices[tri[0]];
//			auto& v1 = rotated_vertices[tri[1]];
//			auto& v2 = rotated_vertices[tri[2]];
//
//			glm::vec3 ray_origin(rotated_vertices[vertex_idx]);
//			glm::vec3 ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
//            
//            glm::vec3 * pV0 = new glm::vec3(count);
//            pV0[0] =glm::vec3(v0);
//            
//            glm::vec3 * pV1 = new glm::vec3(count);
//            pV1[0] =glm::vec3(v1);
//            
//            glm::vec3 * pV2 = new glm::vec3(count);
//            pV2[0] =glm::vec3(v2);
//            
//            auto intersect = ray_triangle_intersect_cl(&ray_origin, &ray_direction,pV0,pV1,pV2, count,false);
//			// first is bool intersect, second is the distance t
//			if (intersect.first == true)
//			{
//				// We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the intersection:
//				// Check if in front or behind?
//				if (intersect.second.get() <= 1e-4)
//				{
//					continue; // the intersection is behind the vertex, we don't care about it
//				}
//				// Otherwise, we've hit a genuine triangle, and the vertex is not visible:
//				visible = false;
//				break;
//			}
//		}
//		visibility.push_back(visible);
//	}
//
//	// Remove vertices from occluding boundary list that are not visible:
//	std::vector<int> final_vertex_ids;
//	for (int i = 0; i < occluding_vertices.size(); ++i)
//	{
//		if (visibility[i] == true)
//		{
//			final_vertex_ids.push_back(occluding_vertices[i]);
//		}
//	}
//	return final_vertex_ids;
//};

        
	} /* namespace fitting */
} /* namespace eos */

#endif /* CLOSESTEDGEFITTING_CL_HPP_ */
