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
 
#include "eos/fitting/closest_edge_fitting_cl.hpp"
//#include "eos/fitting/closest_edge_fitting.hpp"


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // eliminate build warning

#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/ocl.hpp"



#include <opencv2/core/ocl.hpp>


//#include <arrayfire.h>

// 1. Add the af/opencl.h include to your project
//#include <af/opencl.h>

namespace eos {
	namespace fitting {
        
        int complKernel(){
        
        }
        
        
        int testcl() {
            
//            using namespace af;
//            int device = 0;
//            af::setDevice(device);
//            af::info();

        }
        

        void inline  occluding_boundary_vertices_cl(clinfo clinfo,const core::Mesh& mesh, std::vector<glm::vec4> & rotated_vertices,std::vector<int> &occluding_vertices,std::vector<bool> & visibility)
        {
            
         
            int meshSize = mesh.tvi.size();
            int verSize = occluding_vertices.size();
            int size = meshSize;
            size_t length =verSize;

            // Create ArrayFire array objects:
            // ... additional ArrayFire operations here
            // 2. Obtain the device, clinfo.context, and queue used by ArrayFire
//            static cl_clinfo.context clinfo.context = afcl::getclinfo.context();
//            static cl_device_id af_device_id = afcl::getDeviceId();
//            static cl_command_queue queue = afcl::getQueue();
            // 3. Obtain cl_mem references to af::array objects
            // 4. Load, build, and use your kernels.
            //    For the sake of readability, we have omitted error checking.
            int status = CL_SUCCESS;
            // A simple copy kernel, uses C++11 syntax for multi-line strings.
            const char * kernel_name = "copy_kernel";
            
            using namespace std;
            
            
            
            const char * source = R"(
            
            float check( float3 ray_direction, float3  ray_origin, float3 v0,  float3  v1,  float3 v2)
            {
                
                float epsilon = 0.000001;
                
                float3 v0v1 = v1 - v0;
                float3 v0v2 = v2- v0;
                float3 pvec = cross(ray_direction,v0v2);
                float det = dot(v0v1,pvec);
                
                float test = det<0?-det:det;
                if ( test < epsilon){
                    return epsilon;
                    
                }
                
                
                float inv_det = 1 / det;
                
                float3 tvec = ray_origin - v0;
                float  u = dot(tvec, pvec) * inv_det;
                if (u < 0 || u > 1){
                    return epsilon;
                }
                
                float3 qvec = cross(tvec, v0v1);
                float v = dot(ray_direction, qvec) * inv_det;
                if (v < 0 || u + v > 1)
                {
                    return epsilon;
                    
                }
                
                
                float  t = dot(v0v2, qvec) * inv_det;
                return t;
                
            }

            
            void __kernel
            copy_kernel(__global float3 * ray_direction,__global float3 *ray_origin, __global float3 * v0, __global float3 * v1, __global float3 * v2,__global int *meshSize,__global bool * c)
            {
                
                int id = get_global_id(0);
                float epsilon = 0.000001;
                int i=0;
                int size = meshSize[0];
                
                for(i=0;i<size;i++)
                {
                    int idx = size*id+i;
                    float r = check(ray_direction[id],ray_origin[idx],v0[id],v1[id],v2[id]);
                    
                    if(r>epsilon){
                        c[id]=  false;
                        return;
                    }
                }
                
                c[id]=  true;
                
                
            }
            )";
            // Create the program, build the executable, and extract the entry point
            // for the kernel.
            cl_program program = clCreateProgramWithSource(clinfo.context, 1, &source, NULL, &status);
            status = clBuildProgram(program, 1, &clinfo.device, NULL, NULL, NULL);
            cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
            // Set arguments and launch your kernels
            
//            std::auto_ptr<bool> result( new bool[size] );
//            std::auto_ptr<cl_float3> ray_direction( new cl_float3[size] );

            bool * result = new bool[size];//(bool*)malloc(sizeof(bool)*size);
            cl_float3 * ray_direction = new cl_float3[size];//(cl_float3*)malloc(sizeof(cl_float3)*size);
            cl_float3 * ray_origin = (cl_float3*)malloc(sizeof(cl_float3)*meshSize*verSize);
            cl_float3 * v0 = (cl_float3*)malloc(sizeof(cl_float3)*size);
            cl_float3 * v1 = (cl_float3*)malloc(sizeof(cl_float3)*size);
            cl_float3 * v2 = (cl_float3*)malloc(sizeof(cl_float3)*size);
            cl_int * flag = (cl_int*)malloc(sizeof(cl_int)*size);
            
            
            
            const int mem_size = sizeof(cl_float3)*size;
            
            
            
            
            
            int idx = 0;
            
            for (auto&& tri : mesh.tvi)
            {
                auto& _v0 = rotated_vertices[tri[0]];
                auto& _v1 = rotated_vertices[tri[1]];
                auto& _v2 = rotated_vertices[tri[2]];
                
                v0[idx] = {{_v0.x,_v0.y,_v0.z}};
                v1[idx] = {{_v1.x,_v1.y,_v1.z}};
                v2[idx] = {{_v2.x,_v2.y,_v2.z}};
                
                glm::vec3 _ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
                
                ray_direction[idx] ={{_ray_direction.x,_ray_direction.y,_ray_direction.z}};
                idx++;
            }
            
            int vIdx = 0;
            for (const auto& vertex_idx : occluding_vertices)
            {
                int idx = 0;
                for (auto&& tri : mesh.tvi)
                {
                    glm::vec3 _ray_origin(rotated_vertices[vertex_idx]);
                    int i = meshSize*vIdx+idx;
                    ray_origin[i] = {{_ray_origin.x,_ray_origin.y,_ray_origin.z}};
                    idx++;
                    
                }
                vIdx++;
                
            }
            
            
//            for(int i=0;i<size;i++){
//                ray_direction[i] = {{0,0,1}};
//                ray_origin[i] = {{(float)rand(),(float)rand(),(float)rand()}};
//                v0[i] = {{(float)rand(),(float)rand(),(float)rand()}};
//                v1[i] = {{(float)rand(),(float)rand(),(float)rand()}};
//                v2[i] = {{(float)rand(),(float)rand(),(float)rand()}};
//            }
            
            flag[0] = meshSize;
            

            int error;
            cl_mem src_ray_direction = clCreateBuffer(clinfo.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, ray_direction, &error);
            cl_mem src_ray_origin = clCreateBuffer(clinfo.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3)*meshSize*verSize, ray_origin, &error);
            cl_mem src_v0 = clCreateBuffer(clinfo.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, v0, &error);
            cl_mem src_v1 = clCreateBuffer(clinfo.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, v1, &error);
            cl_mem src_v2 = clCreateBuffer(clinfo.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, v2, &error);
            cl_mem src_flag = clCreateBuffer(clinfo.context, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*size, flag, &error);
            cl_mem res_d = clCreateBuffer(clinfo.context, CL_MEM_WRITE_ONLY, sizeof(cl_bool)*size, NULL, &error);
            
            
            
            

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_ray_direction);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_ray_origin);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_v0);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &src_v1);
            clSetKernelArg(kernel, 4, sizeof(cl_mem), &src_v2);
            clSetKernelArg(kernel, 5, sizeof(cl_mem), &src_flag);
            clSetKernelArg(kernel, 6, sizeof(cl_mem), &res_d);
            clEnqueueNDRangeKernel(clinfo.queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);
            
            clFinish(clinfo.queue);
            
//            clEnqueueReadBuffer(command_queue, memobjOutput, CL_TRUE, 0, outputSize, bufferOut, 0, NULL, NULL);

            clEnqueueReadBuffer(clinfo.queue, res_d, CL_TRUE, 0, sizeof(cl_bool)*verSize, result, 0, NULL, NULL);
            
            
            
            

            
            
            for(int i=0;i<verSize;i++){
                visibility.push_back(result[i]);
            }
//                printf("res:%d,%d\n",i,result[i]);
            // 5. Return control of af::array memory to ArrayFire

            
            

            // ... resume ArrayFire operations
            // Because the device pointers, d_x and d_y, were returned to ArrayFire's
            // control by the unlock function, there is no need to free them using
            // clReleaseMemObject()
            
            

//            free( ray_direction);
            free( ray_origin);
            free( v0);
            free( v1);
            free( v2);
            free( flag);
//            free(result);
            delete[] result;
            delete[] ray_direction;
            
//            clReleaseProgram(program);
//
//
            clReleaseKernel(kernel);
//            
//            clReleaseCommandQueue(queue);
//            
//            clReleaseclinfo.context(clinfo.context);
            
            clReleaseMemObject(src_ray_direction);
            clReleaseMemObject(src_ray_origin);
            clReleaseMemObject(src_v0);
            clReleaseMemObject(src_v1);
            clReleaseMemObject(src_v2);
            clReleaseMemObject(src_flag);
            clReleaseMemObject(res_d);


            return;
        }
        
        static clinfo gClInfo;
        int initOpenCL()
        {
            
            cl_int error = 0;   // Used to handle error codes
            cl_platform_id platform;
            cl_context context;
            cl_command_queue queue;
            cl_device_id device;
            
            // Platform
            //            error = clGetPlatformIDs(&platform);
            cl_uint* numPlatforms=NULL;
            
            cl_platform_id platforms[10];
            
            //            error = clGetPlatformIDs(0, platforms, numPlatforms);
            //
            //            error = clGetPlatformIDs(0, NULL,numPlatforms);
            //
            //            if (error != CL_SUCCESS) {
            ////                cout << "Error getting platform id: " << errorMessage(error) << endl;
            ////                exit(error);
            //            }
            // Device
            int gpu = 1;
            error = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            if (error != CL_SUCCESS)
            {
                printf("Error: Failed to create a device group!\n");
                return EXIT_FAILURE;
            }
            // clinfo.context
            context = clCreateContext(0, 1, &device, NULL, NULL, &error);
            if (error != CL_SUCCESS) {
                //                cout << "Error creating clinfo.context: " << errorMessage(error) << endl;
                exit(error);
            }
            // Command-queue
            queue = clCreateCommandQueue(context, device, 0, &error);
            if (error != CL_SUCCESS) {
                //                cout << "Error creating command queue: " << errorMessage(error) << endl;
                exit(error);
            }
            
            gClInfo.context = context;
            gClInfo.device = device;
//            gClInfo.platform = platform;
            gClInfo.queue = queue;

//            using namespace af;
//            int device = 0;
//            af::setDevice(device);
//            af::info();
////            
//            printf("Create a 5-by-3 matrix of random floats on the GPU\n");
//            af::array A = af::randu(5,3, f32);
//            af_print(A);
//            
//            printf("Element-wise arithmetic\n");
//            af::array B = af::sin(A) + 1.5;
//            af_print(B);
//            
//            printf("Negate the first three elements of second column\n");
//            B(seq(0, 2), 1) = B(af::seq(0, 2), 1) * -1;
//            af_print(B);
//            
//            printf("Fourier transform the result\n");
//            af::array C = fft(B);
//            af_print(C);
//            
//            printf("Grab last row\n");
//            af::array c = C.row(end);
//            af_print(c);
//            
//            printf("Scan Test\n");
//            dim4 dims(16, 4, 1, 1);
//            af::array r = constant(2, dims);
//            af_print(r);
//            
//            //            printf("Scan\n");
//            //            af::array S = scan(r, 0, AF_BINARY_MUL);
//            //            af_print(S);
//            //
//            printf("Create 2-by-3 matrix from host data\n");
//            float d[] = { 1, 2, 3, 4, 5, 6 };
//            af::array D(2, 3, d, afHost);
//            af_print(D);
//            
//            printf("Copy last column onto first\n");
//            D.col(0) = D.col(end);
//            af_print(D);
//            
//            // Sort A
//            printf("Sort A and print sorted af::array and corresponding indices\n");
//            af::array vals, inds;
//            sort(vals, inds, A);
//            af_print(vals);
//            af_print(inds);
        }
        
        
//        af::array af_cross (const af::array &x, const af::array &y){
//            af::array result = af::matmulNT(x,y);
//            float tmp[] = {result(1)(0),result(0)(2),result(0)(3)};
//            return aray(1,3,tmp);
//        }
        

        
        
 
        
        
 

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
        std::vector<int>  occluding_boundary_vertices_cl(const core::Mesh& mesh, const morphablemodel::EdgeTopology& edge_topology, glm::mat4x4 R)
        
        {
        
        	// Rotate the mesh:
        	std::vector<glm::vec4> rotated_vertices;
        	std::for_each(begin(mesh.vertices), end(mesh.vertices), [&rotated_vertices, &R](auto&& v) { rotated_vertices.push_back(R * v); });
        
        	// Compute the face normals of the rotated mesh:
        	std::vector<glm::vec3> facenormals;
        	for (auto&& f : mesh.tvi) { // for each face (triangle):
        		auto n = render::compute_face_normal(glm::vec3(rotated_vertices[f[0]]), glm::vec3(rotated_vertices[f[1]]), glm::vec3(rotated_vertices[f[2]]));
        		facenormals.push_back(n);
        	}
        
        	// Find occluding edges:
        	std::vector<int> occluding_edges_indices;
        	for (int edge_idx = 0; edge_idx < edge_topology.adjacent_faces.size(); ++edge_idx) // For each edge... Ef contains the indices of the two adjacent faces
        	{
        		const auto& edge = edge_topology.adjacent_faces[edge_idx];
        		if (edge[0] == 0) // ==> NOTE/Todo Need to change this if we use 0-based indexing!
        		{
        			// Edges with a zero index lie on the mesh boundary, i.e. they are only
        			// adjacent to one face.
        			continue;
        		}
        		// Compute the occluding edges as those where the two adjacent face normals
        		// differ in the sign of their z-component:
        		// Changing from 1-based indexing to 0-based!
        		if (glm::sign(facenormals[edge[0] - 1].z) != glm::sign(facenormals[edge[1] - 1].z))
        		{
        			// It's an occluding edge, store the index:
        			occluding_edges_indices.push_back(edge_idx);
        		}
        	}
        	// Select the vertices lying at the two ends of the occluding edges and remove duplicates:
        	// (This is what EdgeTopology::adjacent_vertices is needed for).
        	std::vector<int> occluding_vertices; // The model's contour vertices
        	for (auto&& edge_idx : occluding_edges_indices)
        	{
        		// Changing from 1-based indexing to 0-based!
        		occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][0] - 1);
        		occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][1] - 1);
        	}
        	// Remove duplicate vertex id's (std::unique works only on sorted sequences):
        	std::sort(begin(occluding_vertices), end(occluding_vertices));
        	occluding_vertices.erase(std::unique(begin(occluding_vertices), end(occluding_vertices)), end(occluding_vertices));
        
        	// Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
        	std::vector<bool> visibility;
            
            occluding_boundary_vertices_cl( gClInfo,mesh,  rotated_vertices,occluding_vertices,visibility);
            
//            visibility.clear();
//            for (const auto& vertex_idx : occluding_vertices)
//            {
//                bool visible = true;
//                // For every tri of the rotated mesh:
//                for (auto&& tri : mesh.tvi)
//                {
//                    auto& v0 = rotated_vertices[tri[0]];
//                    auto& v1 = rotated_vertices[tri[1]];
//                    auto& v2 = rotated_vertices[tri[2]];
//                    
//                    glm::vec3 ray_origin(rotated_vertices[vertex_idx]);
//                    glm::vec3 ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
//                    auto intersect = ray_triangle_intersect(ray_origin, ray_direction, glm::vec3(v0), glm::vec3(v1), glm::vec3(v2), false);
//                    // first is bool intersect, second is the distance t
//                    if (intersect.first == true)
//                    {
//                        // We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the intersection:
//                        // Check if in front or behind?
//                        if (intersect.second.get() <= 1e-4)
//                        {
//                            continue; // the intersection is behind the vertex, we don't care about it
//                        }
//                        // Otherwise, we've hit a genuine triangle, and the vertex is not visible:
//                        visible = false;
//                        break;
//                    }
//                }
//                visibility.push_back(visible);
//            }

            
        	// Remove vertices from occluding boundary list that are not visible:
        	std::vector<int> final_vertex_ids;
        	for (int i = 0; i < occluding_vertices.size(); ++i)
        	{
        		if (visibility[i] == true)
        		{
        			final_vertex_ids.push_back(occluding_vertices[i]);
        		}
        	}
        	return final_vertex_ids;
        };
        
        
	} /* namespace fitting */
} /* namespace eos */


