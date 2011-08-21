/*
 * cost_functions.c
 *
 *  Created on: Aug 20, 2009
 *      Author: destinaton
 */

#ifdef GPU_OPTIMIZE

#ifdef MAC_DARWIN

#define pthread_yield sched_yield

#define WAIT() {\
    clFinish(*queue);\
    }

#else

#define WAIT() {\
    }


#endif

#include "common/gpu/cost_functions.h"
#include "x264.h"
#include "encoder/me.h"
#include "common/frame.h"

#include <stdio.h>
#include <string.h>
#include <sched.h>

static const char *FILENAME = "./common/gpu/computation_kernels.c";


//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platoform ID
//////////////////////////////////////////////////////////////////////////////
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms; 
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        gpu_log(" Error in clGetPlatformIDs Call !!!");
        return -1000;
    }
    else 
    {
        if(num_platforms == 0)
        {
            gpu_log("No OpenCL platform found!");
            return -2000;
        }
        else 
        {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                gpu_log("Failed to allocate memory for cl_platform ID's!");
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
	    cl_uint i=0;
            for(i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS)
                {
                    if(strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL)
            {
                gpu_log("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!");
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}



#define CHECK_CREATE_CONTEXT(error) {\
	if(error != CL_SUCCESS) {\
		switch(error) {\
			case CL_INVALID_PLATFORM: gpu_log("CL_INVALID_PLATFORM"); break;\
			case CL_INVALID_VALUE: gpu_log("INVALID_VALUE"); break;\
			case CL_DEVICE_NOT_AVAILABLE: gpu_log("DEVICE_NOT_AVAILABLE"); break;\
			case CL_DEVICE_NOT_FOUND: gpu_log("DEVICE_NOT_FOUND"); break;\
			case CL_OUT_OF_HOST_MEMORY: gpu_log("OUT_OF_HOST_MEMORY"); break;\
			case CL_INVALID_DEVICE_TYPE: gpu_log("INVALID_DEVICE_TYPE"); break;\
		} \
	}\
}

#define CHECK_SET_KERNEL_ARG(error) {\
	if(error != CL_SUCCESS) {			\
		switch(error) {					\
			case CL_INVALID_KERNEL: gpu_log("CL_INVALID_KERNEL"); break;\
			case CL_INVALID_ARG_INDEX: gpu_log("CL_INVALID_ARG_INDEX"); break;\
			case CL_INVALID_ARG_VALUE: gpu_log("CL_INVALID_ARG_VALUE"); break;\
			case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;\
			case CL_INVALID_SAMPLER: gpu_log("CL_INVALID_SAMPLER"); break;		\
			case CL_INVALID_ARG_SIZE: gpu_log("CL_INVALID_ARG_SIZE"); break;\
		}\
		gpu_log("wasn't able to set kernel arg 0");\
		exit(-1);\
	}\
}

#define CHECK_SET_KERNEL_ARGMSG(error,msg) {\
	if(error != CL_SUCCESS) {			\
		switch(error) {					\
			case CL_INVALID_KERNEL: gpu_log("CL_INVALID_KERNEL"); break;\
			case CL_INVALID_ARG_INDEX: gpu_log("CL_INVALID_ARG_INDEX"); break;\
			case CL_INVALID_ARG_VALUE: gpu_log("CL_INVALID_ARG_VALUE"); break;\
			case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;\
			case CL_INVALID_SAMPLER: gpu_log("CL_INVALID_SAMPLER"); break;		\
			case CL_INVALID_ARG_SIZE: gpu_log("CL_INVALID_ARG_SIZE"); break;\
		}\
		printf("wasn't able to set kernel arg %s",msg);\
		exit(-1);\
	}\
}

#define CHECK_ENQUEUE_KERNELMSG(error,msg) {\
	if(error != CL_SUCCESS) {\
		switch(error) {\
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE"); break;\
			case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;\
			case CL_INVALID_KERNEL: gpu_log("CL_INVALID_KERNEL"); break;\
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;\
			case CL_INVALID_KERNEL_ARGS: gpu_log("CL_INVALID_KERNEL_ARGS"); break;\
			case CL_INVALID_WORK_DIMENSION: gpu_log("CL_INVALID_WORK_DIMENSION"); break;\
			case CL_INVALID_WORK_GROUP_SIZE: gpu_log("CL_INVALID_WORK_GROUP_SIZE"); break;\
			case CL_INVALID_WORK_ITEM_SIZE: gpu_log("CL_INVALID_WORK_ITEM_SIZE"); break;\
			case CL_INVALID_GLOBAL_OFFSET: gpu_log("CL_INVALID_GLOBAL_OFFSET"); break;\
			case CL_OUT_OF_RESOURCES: gpu_log("CL_OUT_OF_RESOURCES"); break;\
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;\
			case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;\
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;\
			case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;\
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;\
		}\
		printf("wasn't able to enqueue kernel %s",msg);\
		exit(0);\
	}\
}

#define CHECK_ENQUEUE_KERNEL(error) {\
	if(error != CL_SUCCESS) {\
		switch(error) {\
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE"); break;\
			case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;\
			case CL_INVALID_KERNEL: gpu_log("CL_INVALID_KERNEL"); break;\
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;\
			case CL_INVALID_KERNEL_ARGS: gpu_log("CL_INVALID_KERNEL_ARGS"); break;\
			case CL_INVALID_WORK_DIMENSION: gpu_log("CL_INVALID_WORK_DIMENSION"); break;\
			case CL_INVALID_WORK_GROUP_SIZE: gpu_log("CL_INVALID_WORK_GROUP_SIZE"); break;\
			case CL_INVALID_WORK_ITEM_SIZE: gpu_log("CL_INVALID_WORK_ITEM_SIZE"); break;\
			case CL_INVALID_GLOBAL_OFFSET: gpu_log("CL_INVALID_GLOBAL_OFFSET"); break;\
			case CL_OUT_OF_RESOURCES: gpu_log("CL_OUT_OF_RESOURCES"); break;\
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;\
			case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;\
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;\
			case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;\
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;\
		}\
		gpu_log("wasn't able to enqueue kernel");\
		exit(0);\
	}\
}

int x264_init_me_reset_grid(x264_gpu_t *gpu) {
    printf("x264_init_me_reset_grid init.....\n");
	/*gpu->error = clSetKernelArg(gpu->me_reset_grid,0,sizeof(cl_mem),&gpu->valid_grid);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_reset_grid,1,sizeof(cl_mem),&gpu->result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_reset_grid,2,sizeof(cl_mem),&gpu->device_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_predict_nieghbors,0,sizeof(cl_mem),&gpu->device_result);
	CHECK_SET_KERNEL_ARG(gpu->error);*/
}

int x264_init_me_selection_kernel(x264_gpu_t *gpu) {
    printf("x264_init_me_selection_kernel init.....\n");
	size_t size;
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,0,sizeof(cl_mem),&gpu->result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->image_info.image_mb_width = gpu->image_info.image_width / gpu->image_info.mb_width;
	gpu->image_info.image_mb_height = gpu->image_info.image_height / gpu->image_info.mb_height;
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,1,sizeof(cl_int),&gpu->iteration);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,2,sizeof(cl_int),&gpu->image_info.image_mb_width);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,3,sizeof(cl_int),&gpu->image_info.image_mb_height);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,4,sizeof(cl_int),&gpu->assistance_points);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,5,sizeof(cl_mem),&gpu->debug);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,6,sizeof(cl_mem),&gpu->device_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	/*gpu->error = clSetKernelArg(gpu->me_selection_kernel,7,sizeof(cl_int2)*(120*3),NULL);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,8,sizeof(workItemResult)*(1*4),NULL);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,9,sizeof(cl_mem),&gpu->valid_grid);
	CHECK_SET_KERNEL_ARG(gpu->error);*/
	gpu->sel_local_size[0] = 1;
	printf("selection init successfull\n");
}

int x264_init_me_gather_kernel(x264_gpu_t *gpu) {
    printf("x264_init_me_gather_kernel init.....\n");
	size_t size;
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,0,sizeof(cl_int),&gpu->image_info.refIdx);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,1,sizeof(cl_int),&gpu->image_info.image_width);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,2,sizeof(cl_int),&gpu->image_info.image_height);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,3,sizeof(cl_int),&gpu->image_info.mb_width);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,4,sizeof(cl_int),&gpu->image_info.mb_height);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,5,sizeof(cl_int),&gpu->image_info.search_region);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,6,sizeof(cl_mem),&gpu->cost_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,7,sizeof(cl_mem),&gpu->image2D);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_gather_kernel,8,sizeof(cl_mem),&gpu->reference2D);
	CHECK_SET_KERNEL_ARG(gpu->error);
	printf("x264_init_me_gather_kernel init.....\n");
	gpu->error = clSetKernelArg(gpu->me_select_best,0,sizeof(cl_mem),&gpu->cost_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_select_best,1,sizeof(cl_mem),&gpu->device_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
    printf("x264_init_me_gather_kernel init.....\n");
	/*gpu->error = clSetKernelArg(gpu->me_select_best_mv,0,sizeof(cl_mem),&gpu->cost_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
	gpu->error = clSetKernelArg(gpu->me_select_best_mv,1,sizeof(cl_mem),&gpu->device_result);
	CHECK_SET_KERNEL_ARG(gpu->error);*/
	printf("x264_init_me_gather_kernel init.....\n");
	gpu->error = clSetKernelArg(gpu->dct16x16_kernel,0,sizeof(cl_mem),&gpu->image2D);
	CHECK_SET_KERNEL_ARG(gpu->error);
	printf("x264_init_me_gather_kernel init ref.....\n");
	gpu->error = clSetKernelArg(gpu->dct16x16_kernel,1,sizeof(cl_mem),&gpu->reference2D);
	CHECK_SET_KERNEL_ARG(gpu->error);
    printf("x264_init_me_gather_kernel init device_result.....\n");
	gpu->error = clSetKernelArg(gpu->dct16x16_kernel,2,sizeof(cl_mem),&gpu->device_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
    printf("x264_init_me_gather_kernel init residual.....\n");
	gpu->error = clSetKernelArg(gpu->dct16x16_kernel,3,sizeof(cl_mem),&gpu->residual_result);
	CHECK_SET_KERNEL_ARG(gpu->error);
    printf("x264_init_me_gather_kernel init nz_result.....\n");
	/*gpu->error = clSetKernelArg(gpu->dct16x16_kernel,4,sizeof(cl_mem),&gpu->nz_result);
	CHECK_SET_KERNEL_ARG(gpu->error);*/
		printf("x264_init_me_gather_kernel finished.....\n");
}

void x264_gpu_doSelection(x264_gpu_t *gpu, int iteration) {
	//printf("trying to enqueue kernel with size: %d\n",gpu->local_size[0]*gpu->global_size[0]*gpu->global_size[1]);
	
	size_t size=0;
	int image_mb_width = gpu->image_info.image_width / gpu->image_info.mb_width;
	int image_mb_height = (gpu->image_info.image_height+gpu->image_info.mb_height-1)/ gpu->image_info.mb_height;

	//printf("size: (%d %d) (%d %d)\n",global_size[0],global_size[1],local_size[0],local_size[1]);

	//cl_event event[1];
	/*gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_reset_grid,
						2,NULL,global_size,local_size,0,NULL,NULL);
	CHECK_ENQUEUE_KERNEL(gpu->error);
	size_t global[2] = {60,68};
	size_t local[2] = {60,1};
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_predict_nieghbors,
										2,NULL,global,local,0,NULL,NULL);
	
	CHECK_ENQUEUE_KERNEL(gpu->error);*/
	/*gpu->error = clWaitForEvents(1,&event[0]);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	
	clGetEventProfilingInfo(event[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	
	printf("reset kernel time needed: %ld ns\n",endTime-startTime);
	clReleaseEvent(event[0]);*/
	gpu->sel_global_size[0] = 120;
	gpu->sel_global_size[1] = 68;

	gpu->sel_local_size[0] = 120;
	gpu->sel_local_size[1] = 1;
	/*printf("size: (%d %d) (%d %d)\n",gpu->sel_global_size[0],gpu->sel_global_size[1],gpu->sel_local_size[0],gpu->sel_local_size[1]);	*/

	cl_event event2[1];
        gpu->iteration = 0;
	gpu->error = clSetKernelArg(gpu->me_selection_kernel,1,sizeof(cl_int),&gpu->iteration);
	CHECK_SET_KERNEL_ARG(gpu->error);
	//printf("enqueueing me_selection_kernel\n");
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_selection_kernel,
					2,NULL,gpu->sel_global_size,gpu->sel_local_size,0,NULL,&event2[0]);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured selection");
	CHECK_ENQUEUE_KERNEL(gpu->error);
	gpu->error = clWaitForEvents(1,&event2[0]);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(event2[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event2[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	printf("selection kernel time needed: %ld ns\n",endTime-startTime);

	cl_event event3[1];

	size_t global_size[2];
	global_size[0] = 30720;
	global_size[1] = 68;
	size_t local_size[2];
	local_size[0] =  256;
	local_size[1] = 1;
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->dct16x16_kernel,
					2,NULL,global_size,local_size,0,NULL,&event3[0]);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured selection");
	CHECK_ENQUEUE_KERNEL(gpu->error);
	gpu->error = clWaitForEvents(1,&event3[0]);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	clGetEventProfilingInfo(event3[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event3[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	printf("dct16x16 kernel time needed: %ld ns\n",endTime-startTime);
}

int MIN_VALUE(int x, int y) {
	if(x < y) return x;
	return y;
}

int MAX_VALUE(int x, int y) {
	if(x < y) return y;
	return x;
}

void MATCH(int *cost, unsigned char* img, unsigned char *r, int i, int j, int t2) {
	int x, y;
	for(y=0;y<16;y++) {
		for(x=0;x<16;x++) {
			cost[t2] += MAX_VALUE(*(img+i+x+y*1984),*(r+j+x+y*1984)) - MIN_VALUE(*(img+i+x+y*1984),*(r+j+x+y*1984));
		}
	}
}

void x264_gpu_processImage_char(x264_gpu_t *gpu, char *ref, char *image, int refIdx) {
	// transfer data to device
	size_t size = gpu->image_info.image_width * gpu->image_info.image_height;

	unsigned char *img = image;//image->plane[0];
	//printf("src plane: %d %d %d\n",image->plane[0],image->i_stride[0],image->i_lines[0]);
	//printf("ref plane: %d %d %d\n",ref->plane[0],ref->i_stride[0],ref->i_lines[0]);

	size_t origin[3] = {0,0,0};
	size_t region[3] = {1920,1088,1};
	size_t inRowPitch = 1920;

	/*int x,y;
	int cost[256];

	for(x=0;x<256;x++) cost[x] = 0;

	int x_off, y_off;
	int t=0;
	int offset = 0*1920 + 16;
	for(y_off=0;y_off<32;y_off+=2) {
		for(x_off=0;x_off<32;x_off+=2) {
			MATCH(cost,image,ref,offset,y_off*1920+x_off,t);
			t++;
		}
	}

	int minCost = 1<<30;
	y = 0;
	for(x=0;x<256;x++) {
		printf("(%3d) %4d |",x,cost[x]);
		if(cost[x] < minCost) {
			minCost = cost[x];
			y = x;
		}
		if(((x+1) & 15) == 0 ) printf("\n");
	}
	printf("\n");

	printf("MIN(test) cost: %d (%d) \n",minCost,y);*/

	gpu->error = clEnqueueWriteImage(gpu->cmd_queue, gpu->image2D, CL_FALSE,origin,region,
									inRowPitch, 0, (void *)image,
									  0, NULL, NULL);

	gpu->error = clEnqueueWriteImage(gpu->cmd_queue, gpu->reference2D, CL_FALSE,origin,region,
									inRowPitch, 0, (void *)ref,
									  0, NULL, NULL);
	if(gpu->error != CL_SUCCESS) gpu_log("can't transfer image to device");
	
	/*char temp[10];
	char *testW = "P2\n";
	char testH[40];
	sprintf(testH,"%d %d\n",gpu->image_info.image_width/2,gpu->image_info.image_height/2);
	char *testString = "255\n";
	 FILE * pFile;
  	pFile = fopen ( "source_image.pgm" , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
	char hex[10];
	int i,j;
	int index;
	unsigned int pixel;
  	for(i=0;i<(gpu->image_info.image_height);i+=2){
		for(j=0;j<(gpu->image_info.image_width);j+=2) {
			index = i*gpu->image_info.image_width + j;
			pixel = img[index]+img[index+1];
			index = (i+1)*gpu->image_info.image_width + j;
			pixel += img[index]+img[index+1];
			pixel = pixel/4;
			sprintf(hex,"%3d", (unsigned char)pixel);
			fwrite(hex,1,strlen(hex),pFile);
			fputc(' ',pFile);
		}
  		fputc('\n',pFile);
 
  		
  	}
  	fclose (pFile);
	sprintf(testH,"%d %d\n",gpu->image_info.image_width,gpu->image_info.image_height);
	pFile = fopen ( "source_image2.pgm" , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
  	for(i=0;i<(gpu->image_info.image_width*gpu->image_info.image_height);i++){
		//unsigned char pixel = ((unsigned int)img[i]+(unsigned int)img[i+1])>>1;
  		sprintf(hex,"%3d", (unsigned char)img[i]);
		fwrite(hex,1,strlen(hex),pFile);
  		//fputc((int)',',pFile);
  		//fputc((unsigned char)img[i],pFile);
  		if(i % (gpu->image_info.image_width-1) == 0 && i > 0) fputc('\n',pFile);
  		else fputc(' ',pFile);
  		
  	}
  	fclose (pFile);
	
	// copy 
	gpu->error = clEnqueueWriteBuffer(gpu->cmd_queue, gpu->image, CL_FALSE,0,
									  size, (void *)image,
									  0, NULL, NULL);
	if(gpu->error != CL_SUCCESS) gpu_log("can't transfer image to device");
	gpu->error = clEnqueueWriteBuffer(gpu->cmd_queue, gpu->reference, CL_FALSE, 0,
									  size, (void *)ref,
									  0, NULL, NULL);
	if(gpu->error != CL_SUCCESS) gpu_log("can't transfer reference image to device");*/
	gpu->image_info.refIdx = refIdx;
	
	if(gpu->mg_gather_set == false) {
		gpu->mg_gather_set = true;
		int result = x264_init_me_gather_kernel(gpu);
		if(result != 0) return result;
	}
	
	// put first kernel to the command queue
	//printf("trying to enqueue kernel with size: %d\n",gpu->local_size[0]*gpu->global_size[0]*gpu->global_size[1]);
	cl_event event[1];

	size_t global_size[2] = {30720,68};
	size_t local_size[2] = {256,1};
	printf("global(%d %d) local(%d %d)\n",global_size[0],global_size[1],gpu->local_size[0],gpu->local_size[1]);
	
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_gather_kernel,
										2,NULL,global_size,local_size,0,NULL,&event[0]);

	CHECK_ENQUEUE_KERNEL(gpu->error);
	gpu->error = clWaitForEvents(1,event);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(event[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	
	printf("me_gather_kernel time needed: %ld ns\n",endTime-startTime);

	size_t global_size2[2] = {30720,68};
	size_t local_size2[2] = {256,1};
	cl_event event2[1];
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_select_best,
										2,NULL,global_size2,local_size2,0,NULL,&event2[0]);

	CHECK_ENQUEUE_KERNEL(gpu->error);
	gpu->error = clWaitForEvents(1,event2);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	clGetEventProfilingInfo(event2[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event2[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	
	printf("me_select_best time needed: %ld ns\n",endTime-startTime);

	/*size_t global_size3[2] = {15360,68};
	size_t local_size3[2] = {256,1};
	cl_event event3[1];
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_select_best_mv,
										2,NULL,global_size3,local_size3,0,NULL,&event3[0]);

	CHECK_ENQUEUE_KERNEL(gpu->error);
	gpu->error = clWaitForEvents(1,event3);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	clGetEventProfilingInfo(event3[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event3[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	
	printf("me_select_best_mv time needed: %ld ns\n",endTime-startTime);*/

	
}

void x264_gpu_reset(x264_gpu_t *gpu) {
	clFlush(gpu->cmd_queue);
	clFinish(gpu->cmd_queue);
}

void x264_gpu_processImage(x264_gpu_t *gpu, x264_frame_t *ref, x264_frame_t *image, int refIdx) {
	size_t origin[3] = {0,0,0};
	size_t region[3] = {1984,1088,1};
	size_t inRowPitch = 1984;

	/*int x,y;
	int cost[256];

	for(x=0;x<256;x++) cost[x] = 0;

	int x_off, y_off;
	int t=0;
	int offset = 0*1920 + 16;
	for(y_off=0;y_off<32;y_off+=2) {
		for(x_off=0;x_off<32;x_off+=2) {
			MATCH(cost,image,ref,offset,y_off*1920+x_off,t);
			t++;
		}
	}

	int minCost = 1<<30;
	y = 0;
	for(x=0;x<256;x++) {
		printf("(%3d) %4d |",x,cost[x]);
		if(cost[x] < minCost) {
			minCost = cost[x];
			y = x;
		}
		if(((x+1) & 15) == 0 ) printf("\n");
	}
	printf("\n");

	printf("MIN(test) cost: %d (%d) \n",minCost,y);*/

	gpu->error = clEnqueueWriteImage(gpu->cmd_queue, gpu->image2D, CL_FALSE,origin,region,
									inRowPitch, 0, (void *)image->plane[0],
									  0, NULL, NULL);

	gpu->error = clEnqueueWriteImage(gpu->cmd_queue, gpu->reference2D, CL_FALSE,origin,region,
									inRowPitch, 0, (void *)ref->plane[0],
									  0, NULL, NULL);
	if(gpu->error != CL_SUCCESS) gpu_log("can't transfer image to device");
	
	/*char temp[10];
	char *testW = "P2\n";
	char testH[40];
	sprintf(testH,"%d %d\n",gpu->image_info.image_width/2,gpu->image_info.image_height/2);
	char *testString = "255\n";
	 FILE * pFile;
  	pFile = fopen ( "source_image.pgm" , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
	char hex[10];
	int i,j;
	int index;
	unsigned int pixel;
  	for(i=0;i<(gpu->image_info.image_height);i+=2){
		for(j=0;j<(gpu->image_info.image_width);j+=2) {
			index = i*gpu->image_info.image_width + j;
			pixel = img[index]+img[index+1];
			index = (i+1)*gpu->image_info.image_width + j;
			pixel += img[index]+img[index+1];
			pixel = pixel/4;
			sprintf(hex,"%3d", (unsigned char)pixel);
			fwrite(hex,1,strlen(hex),pFile);
			fputc(' ',pFile);
		}
  		fputc('\n',pFile);
 
  		
  	}
  	fclose (pFile);
	sprintf(testH,"%d %d\n",gpu->image_info.image_width,gpu->image_info.image_height);
	pFile = fopen ( "source_image2.pgm" , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
  	for(i=0;i<(gpu->image_info.image_width*gpu->image_info.image_height);i++){
		//unsigned char pixel = ((unsigned int)img[i]+(unsigned int)img[i+1])>>1;
  		sprintf(hex,"%3d", (unsigned char)img[i]);
		fwrite(hex,1,strlen(hex),pFile);
  		//fputc((int)',',pFile);
  		//fputc((unsigned char)img[i],pFile);
  		if(i % (gpu->image_info.image_width-1) == 0 && i > 0) fputc('\n',pFile);
  		else fputc(' ',pFile);
  		
  	}
  	fclose (pFile);
	
	// copy 
	gpu->error = clEnqueueWriteBuffer(gpu->cmd_queue, gpu->image, CL_FALSE,0,
									  size, (void *)image,
									  0, NULL, NULL);
	if(gpu->error != CL_SUCCESS) gpu_log("can't transfer image to device");
	gpu->error = clEnqueueWriteBuffer(gpu->cmd_queue, gpu->reference, CL_FALSE, 0,
									  size, (void *)ref,
									  0, NULL, NULL);
	if(gpu->error != CL_SUCCESS) gpu_log("can't transfer reference image to device");*/
	gpu->image_info.refIdx = refIdx;
	
	if(gpu->mg_gather_set == false) {
		gpu->mg_gather_set = true;
		int result = x264_init_me_gather_kernel(gpu);
		if(result != 0) return result;
	}
	
	// put first kernel to the command queue
	//printf("trying to enqueue kernel with size: %d\n",gpu->local_size[0]*gpu->global_size[0]*gpu->global_size[1]);
	cl_event event[1];

	size_t global_size[2] = {30720,68};
	size_t local_size[2] = {256,1};
	printf("global(%d %d) local(%d %d)\n",global_size[0],global_size[1],gpu->local_size[0],gpu->local_size[1]);
	
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_gather_kernel,
										2,NULL,global_size,local_size,0,NULL,NULL);

	CHECK_ENQUEUE_KERNEL(gpu->error);

	size_t global_size2[2] = {30720,68};
	size_t local_size2[2] = {256,1};
	cl_event event2[1];
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_select_best,
										2,NULL,global_size2,local_size2,0,NULL,NULL);

	CHECK_ENQUEUE_KERNEL(gpu->error);

	/*size_t global_size3[2] = {15360,68};
	size_t local_size3[2] = {256,1};
	cl_event event3[1];
	gpu->error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->me_select_best_mv,
										2,NULL,global_size3,local_size3,0,NULL,&event3[0]);

	CHECK_ENQUEUE_KERNEL(gpu->error);
	gpu->error = clWaitForEvents(1,event3);
	if(gpu->error != CL_SUCCESS) gpu_log("kernel event not occured");
	clGetEventProfilingInfo(event3[0],CL_PROFILING_COMMAND_START,
							sizeof(cl_ulong),&startTime,&size);
	clGetEventProfilingInfo(event3[0],CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong),&endTime,&size);
	
	printf("me_select_best_mv time needed: %ld ns\n",endTime-startTime);*/

}

const static const char compilerFlags[] = "-cl-mad-enable -w";

int x264_gpu_deinit(x264_gpu_t *gpu) {
	// release debug element
	if(gpu->debug != (cl_mem)0) clReleaseMemObject(gpu->debug);
	if(gpu->image != (cl_mem)0) clReleaseMemObject(gpu->image);
	if(gpu->reference != (cl_mem)0) clReleaseMemObject(gpu->reference);
	if(gpu->result != (cl_mem)0) clReleaseMemObject(gpu->result);
	if(gpu->device_result != (cl_mem)0) clReleaseMemObject(gpu->device_result);
	if(gpu->valid_grid != (cl_mem)0) clReleaseMemObject(gpu->valid_grid);
	
	// relese kernel
	if(gpu->me_gather_kernel != (cl_kernel)0) clReleaseKernel(gpu->me_gather_kernel);
	if(gpu->me_selection_kernel != (cl_kernel)0)clReleaseKernel(gpu->me_selection_kernel);
	if(gpu->me_refinement_kernel != (cl_kernel)0)clReleaseKernel(gpu->me_refinement_kernel);
	//if(gpu->me_reset_grid != (cl_kernel)0)clReleaseKernel(gpu->me_reset_grid);
	
	// release program
	clReleaseProgram(gpu->program);
	
	//
	clReleaseCommandQueue(gpu->cmd_queue);
	
	//release context
	clReleaseContext(gpu->context);
}

int x264_gpu_init(x264_gpu_t *gpu) {
	int mb_count = gpu->image_info.image_width * gpu->image_info.image_height;
	mb_count /= (gpu->image_info.mb_width * gpu->image_info.mb_height);

	gpu->result_data = (workItemResult *)malloc(sizeof(workItemResult)*mb_count);
	gpu->mg_gather_set = false;
	gpu->calc = 0;
	// local_size must be at least 4
	gpu->assistance_points = 4;
	gpu->iteration = 0;
	gpu->local_size[0] = 32;
	gpu->local_size[1] = 1;

	// get platform
	cl_platform_id platformID = NULL;
	gpu->error = oclGetPlatformID(&platformID);
	if (gpu->error != CL_SUCCESS) {
		printf("Error getting the Platform ID");
		return EXIT_FAILURE;
	}

	// create computing device id
	gpu->error = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &(gpu->device_id), NULL);
	if (gpu->error != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}
	printf("Got device ID: %i\n",gpu->device_id);

	// create computing context
	cl_int errcode;
	gpu->context = clCreateContext(0,1,&(gpu->device_id),NULL,NULL,NULL);
	if(gpu->context == (cl_context)0) {
		gpu_log("wasn't able to create context");
		CHECK_CREATE_CONTEXT(errcode);
		return 0;
	}
	printf("created context\n");

	// query device informations
	size_t max_size[] = {0,0,0};
	size_t max_size_size[3];
	gpu->error = clGetDeviceInfo(gpu->device_id,CL_DEVICE_MAX_WORK_ITEM_SIZES,
								sizeof(size_t)*3,&max_size,&max_size_size);
	
	printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS is: %d | %d | %d\n",max_size[0],max_size[1],max_size[2]);
	
	long max_mem_size=0;
	gpu->error = clGetDeviceInfo(gpu->device_id,CL_DEVICE_MAX_MEM_ALLOC_SIZE,
								 sizeof(cl_long),&max_mem_size,&max_size_size);
	
	printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE is: %d\n",max_mem_size);
	
	gpu->error = clGetDeviceInfo(gpu->device_id,CL_DEVICE_LOCAL_MEM_SIZE,
								 sizeof(cl_long),&max_mem_size,&max_size_size);
	
	printf("CL_DEVICE_LOCAL_MEM_SIZE is: %d\n",max_mem_size);
	
	gpu->error = clGetDeviceInfo(gpu->device_id,CL_DEVICE_GLOBAL_MEM_SIZE,
								 sizeof(cl_long),&max_mem_size,&max_size_size);
	
	printf("CL_DEVICE_GLOBAL_MEM_SIZE is: %d\n",max_mem_size);
	
	size_t max_work_group_size=0;
	gpu->error = clGetDeviceInfo(gpu->device_id,CL_DEVICE_MAX_WORK_GROUP_SIZE,
								 sizeof(size_t),&max_work_group_size,&max_size_size);
	
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE is: %d\n",max_work_group_size);
	
	
	// create computing command queue
	printf("creating command queue\n");
	// CL_QUEUE_PROFILING_ENABLE
	gpu->cmd_queue = clCreateCommandQueue(gpu->context,gpu->device_id,CL_QUEUE_PROFILING_ENABLE,NULL);
	if(gpu->cmd_queue == (cl_command_queue)0) {
		gpu_log("error creating command queue for device");
		return 0;
	}
	printf("creating command queue for dct\n");
	int index;
	for(index=0;index<4;++index) {
		gpu->queue[index] = clCreateCommandQueue(gpu->context,gpu->device_id,CL_QUEUE_PROFILING_ENABLE,NULL);
		if(gpu->queue[index] == (cl_command_queue)0) {
		gpu_log("error creating command queue for device");
		return 0;
	} else {
	   printf("created command queue %d\n",index);
	   }

		}
	/*if(gpu->cmd_queue_dct == (cl_command_queue)0) {
		gpu_log("error creating command queue for device");
		return 0;
	}*/
	printf("opening computation file\n");
	// create computing device 
	FILE * fp = NULL;
	char *pData=NULL;
	char *file[1];
	long lRet = 0, filesize = 0;;
	fp = fopen(FILENAME,"rb");
	if (fp == NULL)
	{
		gpu_log("kernel file cannot be opened");
		return 0;
	}
	else
	{
		fseek(fp, 0, SEEK_END);
		filesize = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		pData = (char *)malloc(sizeof(char)*(filesize + 2));
		lRet = fread(pData, sizeof(char), filesize, fp);
		pData[filesize] = '\0';
		fclose(fp);
	}
	
	if(pData == NULL) {
		gpu_log("wasn't able to read file");
		return 0;
	}
	
	printf("create programm from file...");
	gpu->program = clCreateProgramWithSource(gpu->context,1,(const char **)&pData,NULL,NULL);
	if(gpu->program == (cl_program)0) {
		gpu_log("wasn't able to create program");
		return 0;
	}
	printf("successfull\n");
	
	//build program
	printf("build programm from file\n");
	gpu->error = clBuildProgram(gpu->program,0,NULL,NULL,NULL,NULL);
	if(gpu->error != CL_SUCCESS) {
		gpu_log("can't build program");
		char *result = (char *)malloc(1<<16);
		int returnSize=0;
		gpu->error = clGetProgramBuildInfo(gpu->program,gpu->device_id,CL_PROGRAM_BUILD_LOG,1<<16,result,&returnSize);
		if(returnSize > 0) {
			printf("compilation log: %s %d(%d)\n",result,returnSize,1<<16);
		}
		return 0;
	}
	printf("compilation successfull\n");
	
	// create first kernel
	printf("create gatherMotionKernel from file\n");
	gpu->me_gather_kernel = clCreateKernel(gpu->program,"GatherMotionKernel",&gpu->error);
	if(gpu->me_gather_kernel == (cl_kernel)0) {
		gpu_log("wasn't able to create me_gather_kernel");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}

	gpu->dct16x16_kernel = clCreateKernel(gpu->program,"calcDCT16x16",&gpu->error);
	if(gpu->dct16x16_kernel == (cl_kernel)0) {
		gpu_log("wasn't able to create dct16x16_kernel");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}

	/*printf("create me_reset_grid from file\n");
	gpu->me_reset_grid = clCreateKernel(gpu->program,"resetValidGridKernel",&gpu->error);
	if(gpu->me_reset_grid == (cl_kernel)0) {
		gpu_log("wasn't able to create me_reset_grid");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}*/
	gpu->me_select_best = clCreateKernel(gpu->program,"selectBestMatch",&gpu->error);
	if(gpu->me_select_best == (cl_kernel)0) {
		gpu_log("wasn't able to create me_select_best");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}
	/*gpu->me_select_best_mv = clCreateKernel(gpu->program,"selectBestMatchMV",&gpu->error);
	if(gpu->me_select_best_mv == (cl_kernel)0) {
		gpu_log("wasn't able to create me_select_best_mv");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}
	gpu->me_predict_nieghbors = clCreateKernel(gpu->program,"predictNeighbors",&gpu->error);
	if(gpu->me_predict_nieghbors == (cl_kernel)0) {
		gpu_log("wasn't able to create me_predict_nieghbors");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}*/
	
	gpu->me_selection_kernel = clCreateKernel(gpu->program,"SelectionBestMatchKernel",&gpu->error);
	if(gpu->me_selection_kernel == (cl_kernel)0) {
		gpu_log("wasn't able to create me_gather_kernel");
		switch(gpu->error) {
			case CL_INVALID_PROGRAM: gpu_log("CL_INVALID_PROGRAM occured"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: gpu_log("CL_INVALID_PROGRAM_EXECUTABLE occured"); break;
			case CL_INVALID_KERNEL_NAME: gpu_log("CL_INVALID_KERNEL_NAME occured"); break;
			case CL_INVALID_KERNEL_DEFINITION: gpu_log("CL_INVALID_KERNEL_DEFINITION occured"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE occured"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY occured"); break;
		}
		return 0;
	}
	
	// create second kernel
	// create third kernel
	
	// create cl_mem objects
	// 64??? see frame.c/h for more informations
	size_t size = (gpu->image_info.image_width+64) * (gpu->image_info.image_height+gpu->image_info.image_height%gpu->image_info.mb_height);
	size_t size2 = (gpu->image_info.image_width) * ((gpu->image_info.image_height+gpu->image_info.image_height%gpu->image_info.mb_height));
	cl_image_format format;
    	format.image_channel_order = CL_R;
    	format.image_channel_data_type = CL_UNSIGNED_INT8;

	int temp_height = gpu->image_info.image_height+gpu->image_info.image_height%gpu->image_info.mb_height;

	int temp_width = gpu->image_info.image_width;

	int count = (gpu->image_info.image_width / gpu->image_info.mb_width) *  (gpu->image_info.image_height+gpu->image_info.image_height%gpu->image_info.mb_height)/gpu->image_info.mb_height;

	printf("w%d h%d pitch%d\n",temp_width,temp_height,gpu->image_info.image_width);

	gpu->cost_result = clCreateBuffer(gpu->context,CL_MEM_WRITE_ONLY,sizeof(unsigned int)*289*8160,NULL,&gpu->error);
	if(gpu->cost_result == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating image cl_mem object");
		return -2;
	}

	gpu->residual_result = clCreateBuffer(gpu->context,CL_MEM_READ_WRITE,sizeof(int)*1984*1088,NULL,&gpu->error);
	if(gpu->residual_result == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating image cl_mem object");
		return -2;
	}

	gpu->image2D = clCreateImage2D(gpu->context,CL_MEM_READ_ONLY,&format,
				       temp_width,temp_height,
				       0,NULL,&gpu->error);
	if(gpu->image2D == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: gpu_log("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"); break;
			case CL_IMAGE_FORMAT_NOT_SUPPORTED: gpu_log("CL_IMAGE_FORMAT_NOT_SUPPORTED"); break;
			case CL_INVALID_OPERATION: gpu_log("CL_INVALID_OPERATION"); break;
			case CL_INVALID_IMAGE_SIZE: gpu_log("CL_INVALID_IMAGE_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating image2d cl_mem object");
		return -2;
	}
	gpu->reference2D = clCreateImage2D(gpu->context,CL_MEM_READ_ONLY,&format,
				       temp_width,temp_height,
				       0,NULL,&gpu->error);
	if(gpu->reference2D == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: gpu_log("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"); break;
			case CL_IMAGE_FORMAT_NOT_SUPPORTED: gpu_log("CL_IMAGE_FORMAT_NOT_SUPPORTED"); break;
			case CL_INVALID_OPERATION: gpu_log("CL_INVALID_OPERATION"); break;
			case CL_INVALID_IMAGE_SIZE: gpu_log("CL_INVALID_IMAGE_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating reference2D cl_mem object");
		return -2;
	}
	// create buffer for reference image
	printf("creating image buffer CL_MEM object with size %d\n",size);
	gpu->image = clCreateBuffer(gpu->context,CL_MEM_READ_ONLY,size,NULL,&gpu->error);
	if(gpu->image == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating image cl_mem object");
		return -2;
	}
	printf("creating device_result CL_MEM object with size %d\n",sizeof(workItemResult)*mb_count);
	gpu->device_result = clCreateBuffer(gpu->context,CL_MEM_READ_WRITE,sizeof(workItemResult)*mb_count,NULL,&gpu->error);
	if(gpu->device_result == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating device_result cl_mem object with size %d\n",sizeof(workItemResult)*mb_count);
		return -2;
	}
	// create cl_mem object for reference frame
	printf("creating reference buffer CL_MEM object with size %d\n",size);
	gpu->reference = clCreateBuffer(gpu->context,CL_MEM_READ_ONLY,sizeof(char)*size,NULL,&gpu->error);
	if(gpu->reference == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating reference cl_mem object\n");
		return -2;
	}
	// create cl_mem object for debug 
	printf("creating debug buffer CL_MEM object with size %d\n",size);
	gpu->debug = clCreateBuffer(gpu->context,CL_MEM_READ_WRITE,sizeof(cl_char)*size,NULL,&gpu->error);
	if(gpu->debug == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating reference cl_mem object");
		return -2;
	}

	// create cl_mem object for result
	int image_mb_width = gpu->image_info.image_width / gpu->image_info.mb_width;
	int image_mb_height = (gpu->image_info.image_height+gpu->image_info.mb_height-1)/ gpu->image_info.mb_height;
	size = (image_mb_width+2)*(image_mb_height+1);

	gpu->valid_grid = clCreateBuffer(gpu->context,CL_MEM_READ_WRITE,sizeof(char)*size,NULL,&gpu->error);
	if(gpu->valid_grid == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating reference cl_mem object");
		return -2;
	}
	// set global_size 
	gpu->global_size[0] = image_mb_width*gpu->local_size[0];
	gpu->global_size[1] = image_mb_height;
	printf("image_width in mb: %d image_height in mb: %d\n",image_mb_width,image_mb_height);
	printf("local_size %d | %d\n",gpu->local_size[0],gpu->local_size[1]);
	printf("global_size %d | %d\n",gpu->global_size[0],gpu->global_size[1]);
	printf("size of workItemResult %d\n",sizeof( workItemResult));
	size = sizeof( workItemResult) * (gpu->image_info.image_width*gpu->image_info.image_height)/(gpu->image_info.mb_width*gpu->image_info.mb_height) * gpu->local_size[0];
	
	printf("creating workItemResult buffer CL_MEM object with size %d\n",size/sizeof( workItemResult));
	gpu->result = clCreateBuffer(gpu->context,CL_MEM_READ_WRITE,size,NULL,&gpu->error);
	if(gpu->result == (cl_mem)0) {
		switch(gpu->error) {
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 
		}
		printf("error creating reference cl_mem object");
		return -2;
	}
	
	gpu->result_read = 0;
	
	gpu->sel_global_size[0] = image_mb_width;
	gpu->sel_global_size[1] = image_mb_height;
	
	gpu->sel_local_size[0] = image_mb_width;
	gpu->sel_local_size[1] = 1;
	/*gpu->sel_global_size[0] = 120;
	gpu->sel_global_size[1] = 68;

	gpu->sel_local_size[0] = 120;
	gpu->sel_local_size[1] = 1;*/
	
    printf("finished with gpu_init\n");
}

typedef struct {
    FILE *fh;
    int width, height;
    int next_frame;
} yuv_input_t;

yuv_input_t *open_yuv(char *filename, int width, int height) {
	yuv_input_t *h = malloc(sizeof(yuv_input_t));
    h->width = width;
    h->height = height;
    h->next_frame = 0;
	

	h->fh = fopen(filename, "rb");
    if( h->fh == NULL )
        return 0;
	
    return h;
}

int get_total_frame_yuv( yuv_input_t *h )
{
    int i_frame_total = 0;
	
    if( !fseek( h->fh, 0, SEEK_END ) )
    {
        uint64_t i_size = ftell( h->fh );
        fseek( h->fh, 0, SEEK_SET );
        i_frame_total = (int)(i_size / ( h->width * h->height * 3 / 2 ));
    }
	
    return i_frame_total;
}

typedef struct yuv_frame_t_ {
	char *y_plane;
	char *u_plane;
	char *v_plane;
} yuv_frame_t;

void delete_frame(yuv_frame_t *frame) {
	free(frame->y_plane);
	free(frame->u_plane);
	free(frame->v_plane);
	free(frame);
}

yuv_frame_t* init_yuv_frame(int width, int height) {
	yuv_frame_t *frame = (yuv_frame_t *)malloc(sizeof(yuv_frame_t));
	if(frame == 0) return 0;
	
	// create planes
	frame->y_plane = (char *)malloc(sizeof(char)*width*height);
	frame->u_plane = (char *)malloc(sizeof(char)*width*height/4);
	frame->v_plane = (char *)malloc(sizeof(char)*width*height/4);
	
	return frame;
}

yuv_frame_t* read_yuv(yuv_input_t *h) {
	if( fseek( h->fh,  0, SEEK_SET ) )
		return -1;
	yuv_frame_t *frame = init_yuv_frame(h->width, h->height);
	if(frame == 0) return 0;
	
    if( fread( frame->y_plane, 1, h->width * h->height, h->fh ) <= 0
	   || fread( frame->u_plane, 1, h->width * h->height / 4, h->fh ) <= 0
	   || fread( frame->v_plane, 1, h->width * h->height / 4, h->fh ) <= 0 ) {
		gpu_log("wasn't able to read file");
        return -1;
	}
	
    h->next_frame++;
	
    return frame;
}

int close_yuv(yuv_input_t *h)
{
    if( !h || !h->fh )
        return 0;
    fclose( h->fh );
    free( h );
    return 0;
}

#pragma mark GPU FUNCTIONS
// do a simple test on two different images
void test_x264_gpu_calc_cost_p16x16() {
	// try to open file
	/*yuv_input_t *fst_frame = open_yuv("image_0.yuv",1920,1080);
	if(fst_frame == 0) gpu_log("can't open image_0.yuv");
	yuv_input_t *snd_frame = open_yuv("image_1.yuv",1920,1080);
	if(snd_frame == 0) gpu_log("can't open image_1.yuv");
	yuv_frame_t *frame0 = read_yuv(fst_frame);
	if(frame0 == 0) gpu_log("can't read image_1.yuv");
	yuv_frame_t *frame1 = read_yuv(snd_frame);
	if(frame1 == 0) gpu_log("can't read image_1.yuv");
	close_yuv(fst_frame);
	close_yuv(snd_frame);
	
	
	// run the first kernel test
	x264_gpu_t *gpu_t = (x264_gpu_t *)malloc(sizeof(x264_gpu_t));
	if(gpu_t != 0) {
		gpu_t->image_info.image_width = 1920;
		gpu_t->image_info.image_height = 1080;
		gpu_t->image_info.search_region = 16;
		gpu_t->image_info.mb_width = 16;
		gpu_t->image_info.mb_height = 16;
		gpu_t->image_info.refIdx = 0;
		
		x264_gpu_init(gpu_t);
		x264_init_me_reset_grid(gpu_t);
		x264_init_me_selection_kernel(gpu_t);
		
		// measure time here
		x264_gpu_processImage_char(gpu_t,frame1->y_plane,frame0->y_plane,0);
		printf("doing processing\n");
		size_t size = gpu_t->image_info.image_width * gpu_t->image_info.image_height;
		size_t size2 = size;
		char *debug = (char *)malloc(size);
		
		size /= (gpu_t->image_info.mb_width * gpu_t->image_info.mb_height);
		size *= gpu_t->local_size[0];
		workItemResult *result = (workItemResult *)malloc(sizeof(workItemResult)*size);
		unsigned int *cost_buffer = (unsigned int *)malloc(sizeof(unsigned int)*289*8160);
		printf("number of elements in result: %d\n",size);
		memset(result,0,size);
		int error = clEnqueueReadBuffer(gpu_t->cmd_queue,gpu_t->debug,CL_TRUE,0,size2,debug,0,NULL,NULL);
		if(error != CL_SUCCESS) {
			gpu_log("there was an error in reading debug result");
			switch(error) {
				case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;
				case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
				case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;
				case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
				case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
				case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;
			}
			return 0;
		}
		error = clEnqueueReadBuffer(gpu_t->cmd_queue,gpu_t->cost_result,CL_TRUE,0,sizeof(unsigned int)*289*8160,cost_buffer,0,NULL,NULL);
		if(error != CL_SUCCESS) {
			gpu_log("there was an error in reading debug result");
			switch(error) {
				case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;
				case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
				case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;
				case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
				case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
				case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;
			}
			return 0;
		}
		error = clEnqueueReadBuffer(gpu_t->cmd_queue,gpu_t->result,CL_TRUE,0,sizeof(workItemResult)*size,result,0,NULL,NULL);
		if(error != CL_SUCCESS) {
			gpu_log("there was an error in reading result");
			switch(error) {
				case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;
				case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
				case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;
				case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
				case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
				case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;
			}
			return 0;
		}
		int i;
		int x;
		int index=256;
		int minIndex;
		int min = 1<<30;
		for(x=0;x<256;x++) {
			if(cost_buffer[index] < min) {
				min = cost_buffer[index];
				minIndex = index;
			}
			index++;
		}
		int count;
		index = 256;
		for(x=0;x<256;x++) {
			if(cost_buffer[index] == min) {
				printf("same cost: %d\n",x);
			}
			index++;
		}
		printf("MINIMUM: %d (%d)\n",min,minIndex&255);
		printf("\n");
		index= 0;
		for(x=0;x<16;x++) {
			printf("(%d) ",(unsigned char)debug[index++]);
		}
		/*for(i=0;i<16;i++) {
			for(x=0;x<16;x++) {
				printf("(%d) ",debug[index++]);
			}
			printf("\n");
		}*/
		/*		   printf("\n");
		/*for(i=0;i<32;i++) {
			//if(result[i].mv_diff[0] != 0 || result[i].mv_diff[1] != 0)
			printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d\n",
					i,result[i].mv_predicted[0],result[i].mv_predicted[1],
					result[i].mv_diff[0],result[i].mv_diff[1],
					result[i].costs,result[i].refIdx);
		}*/
	/*	printf("start wit selection\n");
		int image_mb_height = (gpu_t->image_info.image_height+gpu_t->image_info.mb_height-1) / gpu_t->image_info.mb_height;
		int image_mb_width = gpu_t->image_info.image_width / gpu_t->image_info.mb_width;
		x264_gpu_doSelection(gpu_t,0);
		free (result);

		char *testBuffer = (char *)malloc(sizeof(char)*122*69);
		if(testBuffer == 0) exit(-1);
		memset(testBuffer,0,122*69);
		error = clEnqueueReadBuffer(gpu_t->cmd_queue,gpu_t->valid_grid,CL_TRUE,0,sizeof(char)*122*69,testBuffer,0,NULL,NULL);
		if(error != CL_SUCCESS) {
			gpu_log("there was an error in reading result");
			switch(error) {
				case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;
				case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
				case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;
				case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
				case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
				case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;
			}
			//return 0;
		}
		
		char *resultBuffer = (char *)malloc(sizeof(char)*1920*1080);
		if(resultBuffer == 0) {
			printf("wasn't able to create buffer\n");
			exit(0);
		}
		int y;
		for(y=0;y<2;y++) {
			for(x=0;x<122;x++) {
				printf("%d ",testBuffer[x+y*122]);
			}
			printf("\n");
		}
		free(testBuffer);
		
		int mb_count = image_mb_height * image_mb_width;
		
		workItemResult *workItemBuffer = (workItemResult *) malloc(sizeof(workItemResult)*mb_count);
		memset(resultBuffer,0,1920*1080);
		memset(workItemBuffer,0,sizeof(workItemResult)*mb_count);
		
		printf("read debug buffer from device with size:%d\n",sizeof(workItemResult)*mb_count);
		error = clEnqueueReadBuffer(gpu_t->cmd_queue,gpu_t->device_result,CL_TRUE,0,sizeof(workItemResult)*256,workItemBuffer,0,NULL,NULL);
		if(error != CL_SUCCESS) {
			gpu_log("there was an error in reading result");
			switch(error) {
				case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;
				case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
				case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;
				case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
				case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
				case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;
			}
			//return 0;
		}
		
		/*for(i=0;i<mb_count;i++) {
			if(workItemBuffer[i].mv_predicted[0]+workItemBuffer[i].mv_diff[0] > 10 ||
			   workItemBuffer[i].mv_predicted[1]+workItemBuffer[i].mv_diff[1] > 10) {
			      printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d VEC(%d | %d)\n",
					i,workItemBuffer[i].mv_predicted[0],workItemBuffer[i].mv_predicted[1],
					workItemBuffer[i].mv_diff[0],workItemBuffer[i].mv_diff[1],
					workItemBuffer[i].costs,workItemBuffer[i].refIdx,
					workItemBuffer[i].mv_predicted[0]+workItemBuffer[i].mv_diff[0],
					workItemBuffer[i].mv_predicted[1]+workItemBuffer[i].mv_diff[1]);
			}
			
			if(workItemBuffer[i].mv_predicted[0]+workItemBuffer[i].mv_diff[0] < -10 ||
			   workItemBuffer[i].mv_predicted[1]+workItemBuffer[i].mv_diff[1] < -10) {
			      printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d VEC(%d | %d)\n",
					i,workItemBuffer[i].mv_predicted[0],workItemBuffer[i].mv_predicted[1],
					workItemBuffer[i].mv_diff[0],workItemBuffer[i].mv_diff[1],
					workItemBuffer[i].costs,workItemBuffer[i].refIdx,
					workItemBuffer[i].mv_predicted[0]+workItemBuffer[i].mv_diff[0],
					workItemBuffer[i].mv_predicted[1]+workItemBuffer[i].mv_diff[1]);
			}
		}*/
		
		/*for(i=0;i<16;i++) {
			printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d VEC(%d | %d)\n",
					i,workItemBuffer[i].mv_predicted[0],workItemBuffer[i].mv_predicted[1],
					workItemBuffer[i].mv_diff[0],workItemBuffer[i].mv_diff[1],
					workItemBuffer[i].costs,workItemBuffer[i].refIdx,
					workItemBuffer[i].mv_predicted[0]+workItemBuffer[i].mv_diff[0],
					workItemBuffer[i].mv_predicted[1]+workItemBuffer[i].mv_diff[1]);
		}
		printf("\n");
		for(i=118;i<136;i++) {
			printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d VEC(%d | %d)\n",
					i,workItemBuffer[i].mv_predicted[0],workItemBuffer[i].mv_predicted[1],
					workItemBuffer[i].mv_diff[0],workItemBuffer[i].mv_diff[1],
					workItemBuffer[i].costs,workItemBuffer[i].refIdx,
					workItemBuffer[i].mv_predicted[0]+workItemBuffer[i].mv_diff[0],
					workItemBuffer[i].mv_predicted[1]+workItemBuffer[i].mv_diff[1]);
		}/*
		printf("\n");
		for(i=240;i<240+16;i++) {
			printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d\n",
					i,workItemBuffer[i].mv_predicted[0],workItemBuffer[i].mv_predicted[1],
					workItemBuffer[i].mv_diff[0],workItemBuffer[i].mv_diff[1],
					workItemBuffer[i].costs,workItemBuffer[i].refIdx);
		}*/
	/*	printf("\n");
		
		free(resultBuffer);
		free(workItemBuffer);
	}
	
	exit(0);
	
	delete_frame(frame0);
	delete_frame(frame1);
	x264_gpu_deinit(gpu_t);*/
	
}

void x264_read_result(x264_gpu_t *gpu) {
		//x264_timer_start();
		int mb_count = gpu->image_info.image_width * gpu->image_info.image_height;
		mb_count = mb_count / (gpu->image_info.mb_width * gpu->image_info.mb_height);
		int error = clEnqueueReadBuffer(gpu->cmd_queue,gpu->device_result,CL_FALSE,0,sizeof(workItemResult)*mb_count,gpu->result_data,0,NULL,NULL);
		if(gpu->error != CL_SUCCESS) {
			gpu_log("there was an error in reading result");
			switch(error) {
				case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;
				case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;
				case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;
				case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;
				case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
				case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;
			}
			printf("GPU ERROR: wasn't able to read result_data!!!\n");
		}
		clFinish(gpu->cmd_queue);
		gpu->result_read = 1;
}

#pragma mark COST FUNCTIONS

void x264_gpu_read_result(x264_gpu_t *gpu) {
	if(gpu->result_read == 1) return;
	printf("read result\n");
	x264_read_result(gpu);
	
	/// read first 10 items
	#if 0
	int i;
	workItemResult *result = gpu->result_data;
	for(i=0;i<20;i++) {
		printf("(%d) mvp: (%d | %d) diff: (%d | %d) (costs: %d) refIdx: %d\n",
					i,result[i].mv_predicted[0],result[i].mv_predicted[1],
					result[i].mv_diff[0],result[i].mv_diff[1],
					result[i].costs,result[i].refIdx);
	}
	#endif
	//exit(0);
}

static int index_count=0;

void Plain2MV(cl_int2 *vector, int mv) {
	vector->x = mv >> 16;
	vector->x -= 64;
	vector->y = mv & 0xFFFF;
	vector->y -= 64;
}

void x264_gpu_me_search_ref2( x264_t *h, x264_gpu_thread_t *gpu, x264_me_t *m, int i_mvc, int *p_halfpel_thresh ) {
	int mbs_per_row = gpu->config->image_info.image_width / gpu->config->image_info.mb_width;
	int mb_x = h->mb.i_mb_x;
	int mb_y = h->mb.i_mb_y;
	// m->i_ref
	
	workItemResult item = gpu->result[mb_x+mb_y*mbs_per_row];
	m->cost = (item.costs);
	cl_int2 vector, test;
	Plain2MV(&test,item.mv_diff);
	vector = test;
//	vector[0] = test[0];
//	vector[1] = test[1];
	Plain2MV(&test,item.mv_predicted);
	vector.x += test.x;
	vector.y += test.y;
	m->mv[0] = vector.x << 2;
	m->mv[1] = vector.y << 2;

	const int16_t *p_cost_mvx = m->p_cost_mv - m->mvp[0];
	const int16_t *p_cost_mvy = m->p_cost_mv - m->mvp[1];
	m->cost_mv = p_cost_mvx[ m->mv[0] ] + p_cost_mvy[ m->mv[1] ];
	m->cost += m->cost_mv;

	//printf("gather object: %d (%d %d) mvp(%3d %3d) mvp_diff(%3d %3d) sum(%3d %3d)\n",item.costs,mb_x,mb_y,item.mv_predicted[0],item.mv_predicted[1],item.mv_diff[0],item.mv_diff[1],m->mv[0]>>2,m->mv[1]>>2);
	
	/*if(index_count < 10) index_count++;
	else exit(0);
	printf("gather object: %d (%d %d) mvp(%3d) mvp_diff(%d) sum(%3d %3d)\n",item.costs,mb_x,mb_y,item.mv_predicted,item.mv_diff,m->mv[0]>>2,m->mv[1]>>2);*/
}
void x264_gpu_mb_predict_mv_ref16x162( x264_t *h, x264_gpu_thread_t *gpu, int i_list, int i_ref, int16_t mvc[9][2], int *i_mvc ) {
}
void x264_gpu_mb_predict_mv_16x162( x264_t *h, x264_gpu_thread_t *gpu, int i_list, int i_ref, int16_t mvp[2] ) {
	int mbs_per_row = gpu->config->image_info.image_width / gpu->config->image_info.mb_width;
	int mb_x = h->mb.i_mb_x;
	int mb_y = h->mb.i_mb_y;
	
	workItemResult item = gpu->result[mb_x+mb_y*mbs_per_row];
	cl_int2 test;
	Plain2MV(&test,item.mv_predicted);
	mvp[0] = (test.x) << 2;
	mvp[1] = (test.y) << 2;
	
	/*mvp[0] <<= 2;
	mvp[1] <<= 2;*/
}

// gather mvp for the current macroblock from the result of the device
void x264_gpu_mb_predict_mv_16x16( x264_t *h, x264_gpu_t *gpu, int i_list, int i_ref, int16_t mvp[2] )
{
	// get current mb position
	// mvp must be in halfpel positions
	int mbs_per_row = gpu->image_info.image_width / gpu->image_info.mb_width;
	int mb_x = h->mb.i_mb_x;
	int mb_y = h->mb.i_mb_y;
	
	workItemResult item = gpu->result_data[mb_x+mb_y*mbs_per_row];
	cl_int2 test;
	Plain2MV(&test, item.mv_predicted);
	mvp[0] = (test.x) << 2;
	mvp[1] = (test.y) << 2;
	
	mvp[0] <<= 2;
	mvp[1] <<= 2;
//	printf("estimated mvp(%f | %f)\n",(float)mvp[0]/4.0f,(float)mvp[1]/4.0f);
}

// not specified in the standard ... looking what they war doing because we don't really need this function in our case
void x264_gpu_mb_predict_mv_ref16x16( x264_t *h, x264_gpu_t *gpu, int i_list, int i_ref, int16_t mvc[9][2], int *i_mvc ) {
}

static int test=0;

// can be modifiedL h, gpu, m, mvc, p_halfpel_thresh
void x264_gpu_me_search_ref( x264_t *h, x264_gpu_t *gpu, x264_me_t *m, int i_mvc, int *p_halfpel_thresh ) {
	int mbs_per_row = gpu->image_info.image_width / gpu->image_info.mb_width;
	int mb_x = h->mb.i_mb_x;
	int mb_y = h->mb.i_mb_y;
	
	workItemResult item = gpu->result_data[mb_x+mb_y*mbs_per_row];
	//m->cost = (item.costs);
	cl_int2 vector, test;
	Plain2MV(&test,item.mv_diff);
	vector = test;
//	vector[0] = test[0];
//	vector[1] = test[1];
	Plain2MV(&test,item.mv_predicted);
	vector.x += test.x;
	vector.y += test.y;
	m->mv[0] = vector.x << 2;
	m->mv[1] = vector.y << 2;
	
	// compute the real cost
	const int16_t *p_cost_mvx = m->p_cost_mv - m->mvp[0];
	const int16_t *p_cost_mvy = m->p_cost_mv - m->mvp[1];
	m->cost_mv = p_cost_mvx[ m->mv[0] ] + p_cost_mvy[ m->mv[1] ];
	m->cost += m->cost_mv;
}

int x264_gpu_inter_sub16x16_dct( int16_t dct[16][4][4], x264_t *h, x264_gpu_thread_t *thread, int16_t nz[16]) {
	x264_gpu_t *gpu = thread->config;
	int mbs_per_row = gpu->image_info.image_width / gpu->image_info.mb_width;
	int mb_x = h->mb.i_mb_x;
	int mb_y = h->mb.i_mb_y;

	int mb_id = mb_x + mb_y*mbs_per_row;

	int readIdx = thread->dctRdIdx;

	int i=0, x, y;
	int16_t *ptr = dct;

	int index=0;
	int process = thread->dct_delta;
	index += (mbs_per_row*(process) <= mb_id && mb_id < mbs_per_row*(process*2))*1;
	index += (mbs_per_row*(process*2) <= mb_id && mb_id < mbs_per_row*(process*3))*2;
	index += (mbs_per_row*(process*3) <= mb_id && mb_id < mbs_per_row*(process*4))*3;

	int out = 0;

	//printf("readIdx: %d index: %d ptrIdx: %d\n",readIdx,index,thread->element->ptr_index);
	if(thread->element->ptr_index < index) {
		cl_int execution_status = -1;
		size_t retSize;
		int end = 1;
		int ret;
		do {
			pthread_yield();
			ret = clGetEventInfo(thread->g_dct_event[readIdx][thread->element->ptr_index+1],
				       CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(cl_int),&execution_status,&retSize);
			if(thread->element->active[thread->element->ptr_index+1] == 0)  execution_status = 0x1;

			/*if(out == 0) {
				++out;
				printf("SUB16x16_DCT access event %d %d active:%d %d\n",readIdx,thread->element->ptr_index+1,thread->element->active[thread->element->ptr_index+1],execution_status);
			}*/
			if(retSize == 4) {
				end -= (execution_status == CL_COMPLETE);
				/*if(end && out == 0) {
					out++;
					printf("x264_gpu_inter_sub16x16_dct will have to wait for element %d %d\n",readIdx,thread->element->ptr_index+1);
				} else {
					if(!end) printf("x264_gpu_inter_sub16x16_dct ready for element %d %d\n",readIdx,thread->element->ptr_index+1);
				}*/
			}
		} while(end);
        //exit(0);
		//printf("readIdx: %d index: %d ptrIdx: %d\n",readIdx,index,thread->element->ptr_index);
		//printf("read: %d %d [%d]\n",thread->dctRdIdx,index,thread->pushIdx[thread->dctRdIdx][index]);
		if(index == (thread->dct_count-1)) thread->dctRdIdx = (thread->dctRdIdx+1) & THREAD_BUFFER_DEPTH_MASK;
		thread->element->ptr_index++;
		thread->element->active[thread->element->ptr_index] = 0;
	}

	mb_id <<= 8;
	int mb_id2 = mb_id >> 4;	

	memcpy(dct,&(thread->dct16x16result[mb_id]),sizeof(int16_t)*256);
	memcpy(nz,&(thread->nz[mb_id2]),sizeof(int)*16);
}

#define CHECK_CL_BUFFER(buffer,error,where)\
	if(buffer == (cl_mem)0) {\
		switch(error) {\
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;\
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;\
			case CL_INVALID_BUFFER_SIZE: gpu_log("CL_INVALID_BUFFER_SIZE"); break;\
			case CL_INVALID_HOST_PTR: gpu_log("CL_INVALID_HOST_PTR"); break;\
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;\
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;	 \
		}\
		printf("error creating %s cl_mem object",where);\
		return -2;\
	}

#define CHECK_READ_BUFFER(error)\
	if(error != CL_SUCCESS) {\
		gpu_log("there was an error in reading result");\
		switch(error) {\
			case CL_INVALID_COMMAND_QUEUE: gpu_log("CL_INVALID_COMMAND_QUEUE"); break;\
			case CL_INVALID_CONTEXT: gpu_log("CL_INVALID_CONTEXT"); break;\
			case CL_INVALID_MEM_OBJECT: gpu_log("CL_INVALID_MEM_OBJECT"); break;\
			case CL_INVALID_VALUE: gpu_log("CL_INVALID_VALUE"); break;\
			case CL_INVALID_EVENT_WAIT_LIST: gpu_log("CL_INVALID_EVENT_WAIT_LIST"); break;\
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: gpu_log("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;\
			case CL_OUT_OF_HOST_MEMORY: gpu_log("CL_OUT_OF_HOST_MEMORY"); break;\
		}\
		printf("GPU ERROR: wasn't able to read result_data!!!\n");\
	}

void x264_init_buffer_element(x264_gpu_thread_t *gpu, x264_gpu_buffer_t *element, int idx) {
	size_t size;
	element->status = RUNNING;

	cl_image_format format;
    	format.image_channel_order = CL_R;
    	format.image_channel_data_type = CL_UNSIGNED_INT8;

	element->image = 0;
	element->reference = 0;
	element->ptr_index = -1;
	element->active[0] = 0;
	element->active[1] = 0;
	element->active[2] = 0;
	element->active[3] = 0;
	
	int error;
	int mb_count = gpu->config->image_info.image_width * gpu->config->image_info.image_height;
	mb_count /= (gpu->config->image_info.mb_width * gpu->config->image_info.mb_height);

	int temp_height = gpu->config->image_info.image_height+gpu->config->image_info.image_height%gpu->config->image_info.mb_height;

	int temp_width = gpu->config->image_info.image_width+64;
	
	int img_mb_width = (temp_width-64) >> 4;
	int img_mb_height = (temp_height) >> 4;

	size = (gpu->config->image_info.image_width+64);
	size *= (gpu->config->image_info.image_height + (gpu->config->image_info.image_height)%(gpu->config->image_info.mb_height));
	element->g_image = clCreateImage2D(gpu->config->context,CL_MEM_READ_ONLY,&format,
				       temp_width,temp_height,
				       0,NULL,&error);
	CHECK_CL_BUFFER(element->g_image,error,"g_image in buffer element");
		
	element->g_ref = clCreateImage2D(gpu->config->context,CL_MEM_READ_ONLY,&format,
				       temp_width,temp_height,
				       0,NULL,&error);
	CHECK_CL_BUFFER(element->g_ref,error,"g_image in buffer element");

	size = (img_mb_width*img_mb_height)*sizeof(workItemResult);
	element->g_device_result = clCreateBuffer(gpu->config->context,CL_MEM_READ_WRITE,size,NULL,&error);
	CHECK_CL_BUFFER(element->g_device_result,error,"g_device_result in buffer element");

	element->g_dct_result = clCreateBuffer(gpu->config->context,CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,sizeof(cl_short)*temp_width*temp_height,NULL,&error);
	CHECK_CL_BUFFER(element->g_dct_result,error,"g_dct_result in buffer element");

	element->g_nz_result = clCreateBuffer(gpu->config->context,CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,sizeof(cl_int)*img_mb_width*img_mb_height*16,NULL,&error);
	CHECK_CL_BUFFER(element->g_nz_result,error,"g_dct_result in buffer element");

	element->nz = (int *)clEnqueueMapBuffer(gpu->config->queue[idx],element->g_nz_result, CL_TRUE,CL_MAP_READ, 0, sizeof(cl_int)*img_mb_width*img_mb_height*16, 0, NULL, NULL, NULL);
	//printf("finished: %d\n",clFinish(gpu->config->queue[idx]));
	element->dct16x16result = (int16_t *)clEnqueueMapBuffer(gpu->config->queue[idx],element->g_dct_result, CL_TRUE,CL_MAP_READ, 0, sizeof(cl_short)*img_mb_width*img_mb_height*256, 0, NULL, NULL, NULL);
	//printf("finished: %d\n",clFinish(gpu->config->queue[idx]));

	//element->dct16x16result = (int16_t *)malloc(sizeof(int16_t)*1984*1088);
	int i;
	int delta = img_mb_width*img_mb_height*256;
	delta /= gpu->dct_count;

	for(i=0;i<gpu->dct_count;i++) {
		element->dctPtr[i] = &element->dct16x16result[delta*i]; 
	}

	/*element->dctPtr[0] = &element->dct16x16result[delta*0];
	element->dctPtr[1] = &element->dct16x16result[delta*1];
	element->dctPtr[2] = &element->dct16x16result[delta*2];
	element->dctPtr[3] = &element->dct16x16result[delta*3];*/

	//element->nz = (int16_t *)malloc(sizeof(int)*120*68*16);
	//delta = img_mb_width*img_mb_height*(16>>2);
	delta = img_mb_width*img_mb_height*(16);
	delta /= gpu->dct_count;
	
	for(i=0;i<gpu->dct_count;i++) {
		element->nz_ptr[i] = &element->nz[delta*i];
	}

	/*element->nz_ptr[0] = &element->nz[delta*0];
	element->nz_ptr[1] = &element->nz[delta*1];
	element->nz_ptr[2] = &element->nz[delta*2];
	element->nz_ptr[3] = &element->nz[delta*3];*/
	
	element->g_cost_result = clCreateBuffer(gpu->config->context,CL_MEM_READ_WRITE,sizeof(unsigned int)*256*8160,NULL,&error);
	CHECK_CL_BUFFER(element->g_cost_result,error,"g_image in buffer element");

	/*element->dct16x16result = (int16_t *)malloc(sizeof(int16_t)*1984*1088);
	if(element->dct16x16result == 0) {
		printf("error in allocating element->dct16x16result buffer!!!!!\n");
		 ;
	}*/

	element->result = malloc(size);
	if(element->result == 0) {
		printf("error in creating buffer element\n");
		exit(0);
	}

	size <<= 2;
	element->g_result = clCreateBuffer(gpu->config->context,CL_MEM_READ_WRITE,size,NULL,&error);
	CHECK_CL_BUFFER(element->g_result,error,"g_image in buffer element");

	/*int image_mb_width = gpu->config->image_info.image_width / gpu->config->image_info.mb_width;
	int image_mb_height = (gpu->config->image_info.image_height+gpu->config->image_info.mb_height-1)/ gpu->config->image_info.mb_height;
	size = (image_mb_width+2)*(image_mb_height+1);
	element->g_grid = clCreateBuffer(gpu->config->context,CL_MEM_READ_ONLY,size,NULL,&error);
	CHECK_CL_BUFFER(element->g_image,error,"g_image in buffer element");*/
}

void x264_deinit_buffer_element(x264_gpu_buffer_t *element) {
	if(element->g_image != (cl_mem)0) clReleaseMemObject(element->g_image);
	if(element->g_ref != (cl_mem)0) clReleaseMemObject(element->g_ref);
	if(element->g_result != (cl_mem)0) clReleaseMemObject(element->g_result);
	if(element->g_device_result != (cl_mem)0) clReleaseMemObject(element->g_device_result);
	if(element->g_grid != (cl_mem)0) clReleaseMemObject(element->g_grid);
	if(element->g_cost_result != (cl_mem)0) clReleaseMemObject(element->g_cost_result);
	if(element->g_dct_result != (cl_mem)0) clReleaseMemObject(element->g_dct_result);
}

/// TODO must be mutexed to garantuee access only to one single item per time slot
inline void x264_update_buffer_element(x264_gpu_thread_t *thread, x264_gpu_buffer_t *element, x264_frame_t *img, x264_frame_t *ref) {
	x264_gpu_t *gpu = thread->config;
	cl_command_queue *queue = &(gpu->queue[thread->writeIdx]);
//	printf("update (buffer) in queue %d [thread->writeIdx]\n",thread->writeIdx);
	//printf("update element %p\n",element);
	int error;
	int mb_count = gpu->image_info.image_width * gpu->image_info.image_height;
	mb_count /= (gpu->image_info.mb_width * gpu->image_info.mb_height);

	int image_stride = img->i_stride[0];
	int image_height = img->i_lines[0];

	int image_mb_width2 = (img->i_stride[0]-64) >> 4;
	int image_mb_height2 = (img->i_lines[0]) >> 4;

	size_t size = (img->i_stride[0]) * (img->i_lines[0]);
	
	element->image = img;
	element->reference = ref;

	mlock(element->reference->plane[0],size);
	mlock(element->image->plane[0],size);

	if(img == NULL || ref == NULL) {
		printf("null pointer found\n");
		exit(0);
	}
	// reset arguments

	/// reset selection settings
	error = clSetKernelArg(gpu->me_selection_kernel,0,sizeof(cl_mem),&element->g_result);
	CHECK_SET_KERNEL_ARGMSG(error,"me_selection_kernel arg: 0");
	error = clSetKernelArg(gpu->me_selection_kernel,6,sizeof(cl_mem),&element->g_device_result);
	CHECK_SET_KERNEL_ARGMSG(error,"me_selection_kernel arg: 6");

	/// reset gathering settings
	error = clSetKernelArg(gpu->me_gather_kernel,8,sizeof(cl_mem),&element->g_ref);
	CHECK_SET_KERNEL_ARGMSG(error,"me_gather_kernel arg: 8");
	error = clSetKernelArg(gpu->me_gather_kernel,7,sizeof(cl_mem),&element->g_image);
	CHECK_SET_KERNEL_ARGMSG(error,"me_gather_kernel arg: 7");
	error = clSetKernelArg(gpu->me_gather_kernel,6,sizeof(cl_mem),&element->g_cost_result);
	CHECK_SET_KERNEL_ARGMSG(error,"me_gather_kernel arg: 6");

	error = clSetKernelArg(gpu->me_select_best,0,sizeof(cl_mem),&element->g_cost_result);
	CHECK_SET_KERNEL_ARGMSG(error,"me_select_best arg: 0");
	error = clSetKernelArg(gpu->me_select_best,1,sizeof(cl_mem),&element->g_device_result);
	CHECK_SET_KERNEL_ARGMSG(error,"me_select_best arg: 1");

	char *ptr = element->image->plane[0];
	// do computation
	// transfer data to device
	//printf("address of ptr(image): %p\n",ptr);
	size_t origin[3] = {0,0,0};
	size_t region[3] = {image_stride,image_height,1};
	size_t inRowPitch = image_stride;

	/*unsigned char *img_ref = (unsigned char *)(element->image->plane[0]);
	//printf("qp: %d f_value: %d qbits: %d\n",qp,f_value,qbits);
	char temp[10];
	char *testString = "255\n";
	char *testW = "P2\n";
	char testH[40];
	char fileName[50];
	sprintf(testH,"%d %d\n",region[0],region[1]);
	sprintf(fileName,"source_image%d%d.pgm",thread->curr_frame_nr,1);
	FILE * pFile;
	char hex[10];
	pFile = fopen ( fileName , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
	int i;
  	for(i=0;i<(gpu->image_info.image_width*gpu->image_info.image_height);i++){
		//unsigned char pixel = ((unsigned int)img[i]+(unsigned int)img[i+1])>>1;
  		sprintf(hex,"%3d", (unsigned char)img_ref[i]);
		fwrite(hex,1,strlen(hex),pFile);
  		//fputc((int)',',pFile);
  		//fputc((unsigned char)img[i],pFile);
  		if(i % (gpu->image_info.image_width-1) == 0 && i > 0) fputc('\n',pFile);
  		else fputc(' ',pFile);
  		
  	}
  	fclose (pFile);
	img_ref = (unsigned char *)(element->reference->plane[0]);
	sprintf(testH,"%d %d\n",region[0],region[1]);
	sprintf(fileName,"source_image%d%d.pgm",thread->curr_frame_nr,0);
	pFile = fopen ( fileName , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
  	for(i=0;i<(gpu->image_info.image_width*gpu->image_info.image_height);i++){
		//unsigned char pixel = ((unsigned int)img[i]+(unsigned int)img[i+1])>>1;
  		sprintf(hex,"%3d", (unsigned char)img_ref[i]);
		fwrite(hex,1,strlen(hex),pFile);
  		//fputc((int)',',pFile);
  		//fputc((unsigned char)img[i],pFile);
  		if(i % (gpu->image_info.image_width-1) == 0 && i > 0) fputc('\n',pFile);
  		else fputc(' ',pFile);
  		
  	}
  	fclose (pFile);
	if(thread->curr_frame_nr == 25) exit(0);*/

	error = clEnqueueWriteImage(*queue, element->g_image, CL_FALSE,origin,region,
									inRowPitch, 0, (void *)element->image->plane[0],
									  0, NULL, NULL);

	ptr = element->reference->plane[0];
	CHECK_ENQUEUE_KERNELMSG(error,"write image");
	//printf("finished: %d\n",clFinish(*queue));
	error = clEnqueueWriteImage(*queue, element->g_ref, CL_FALSE,origin,region,
									 inRowPitch, 0, (void *)element->reference->plane[0],
									  0, NULL, NULL);
    CHECK_ENQUEUE_KERNELMSG(error,"write referece");
	//printf("finished: %d\n",clFinish(*queue));
	size_t global_size1[2] = {image_mb_width2*256,image_mb_height2};
	size_t local_size1[2] = {256,1};
	
	error = clEnqueueNDRangeKernel(*queue,gpu->me_gather_kernel,
					2,NULL,global_size1,local_size1,0,NULL,NULL);
					//printf("gather\n");
	//printf("finished: %d\n",clFinish(*queue));
	CHECK_ENQUEUE_KERNELMSG(error,"me_gather_kernel");

	gpu->error = clEnqueueNDRangeKernel(*queue,gpu->me_select_best,
					2,NULL,global_size1,local_size1,0,NULL,NULL);
	//printf("select best match\n");
	//printf("finished: %d\n",clFinish(*queue));
    CHECK_ENQUEUE_KERNELMSG(error,"select best");

	size = 0;
	int image_mb_width = gpu->image_info.image_width / gpu->image_info.mb_width;
	int image_mb_height = (gpu->image_info.image_height+gpu->image_info.mb_height-1)/ gpu->image_info.mb_height;

	// do remove first
    size_t global_size_sel[2] = {image_mb_width,image_mb_height>>1};
	size_t local_size_sel[2] = {image_mb_width,1};	
	
	//printf("dimension: %d %d\n",image_mb_width,image_mb_height>>1);

	int iteration=0, iteration2=1;
	error = clSetKernelArg(gpu->me_selection_kernel,1,sizeof(int),&iteration);
	CHECK_SET_KERNEL_ARG(error);
	error = clEnqueueNDRangeKernel(*queue,gpu->me_selection_kernel,2,NULL,global_size_sel,local_size_sel,0,NULL,NULL);
	CHECK_ENQUEUE_KERNELMSG(error,"me_selection_kernel");
	//printf("selection\n");
    //printf("finished: %d\n",clFinish(*queue));
	error = clSetKernelArg(gpu->me_selection_kernel,1,sizeof(int),&iteration2);
	error = clEnqueueNDRangeKernel(*queue,gpu->me_selection_kernel,2,NULL,global_size_sel,local_size_sel,0,NULL,NULL);
	//printf("selection\n");
	//printf("finished: %d\n",clFinish(*queue));
	//clReleaseEvent(element->g_event);
	error = clEnqueueReadBuffer(*queue,element->g_device_result,CL_FALSE,0,sizeof(workItemResult)*(image_mb_width*image_mb_height),element->result,0,NULL,&(element->g_event));
	element->status = RUNNING;
		//printf("finished: %d\n",clFinish(*queue));
}

void x264_gpu_prepush_sub16x16_dct(x264_t *h, x264_gpu_thread_t *thread, int idx) {
// 	int a = thread->pushIdx > -1;
// 	int b = thread->init == 0;
// 	x264_timer_start();
	//printf("trying to push: %d %d [%d]\n",thread->dctWrIdx2,idx,thread->pushIdx[thread->dctWrIdx2][idx]);
	int out = 0;
	while(thread->pushIdx[thread->dctWrIdx2][idx] > -1 || thread->init == 0) {
		pthread_yield();
	} 
// 	x264_timer_stop();
// 	x264_print_time();
	pthread_mutex_lock(&(thread->pushtex));
	thread->pushIdx[thread->dctWrIdx2][idx] = idx;
	if(idx == 0) thread->buffer[thread->dctWrIdx2].fdec = h->fdec;
	if(idx == (thread->dct_count-1)) {
		//printf("prePush: %p frame\n",h->fdec);
		thread->dctWrIdx2 = (thread->dctWrIdx2+1) & THREAD_BUFFER_DEPTH_MASK;
	}
	pthread_mutex_unlock(&(thread->pushtex));
}

void x264_gpu_udpate_dct(x264_t *h, x264_gpu_thread_t *thread, int idx) {
	//printf("called update dct\n");
	pthread_mutex_lock(&(thread->mutex));
	x264_gpu_buffer_t *element = &(thread->buffer[thread->dctWrIdx]); 	// because of pre computation
	//printf("update(DCT) element %p\n",element);
	x264_frame_t *frame = element->fdec;
	pthread_mutex_unlock(&(thread->mutex));
	x264_gpu_t *gpu = thread->config;
	//printf("thread->dctWrIdx: %d %d %p\n",thread->dctWrIdx,idx,frame);
	cl_command_queue *queue = &(gpu->queue[thread->dctWrIdx]);
	//if(idx == 3) exit(0);
	
	int y_orig = (thread->dct_delta)<<4;
	size_t origin[3] = {0,y_orig*idx,0};
	int stride = frame->i_stride[0];
	int image_mb_width = (stride-64)>>4;
	int image_mb_height = (frame->i_lines[0]) >> 4;
	size_t region[3] = {stride,y_orig+32,1};
	if(idx == (thread->dct_count -1)) {
		region[1] = y_orig;
	}
	size_t inRowPitch = stride;
	int ptr_offset = (origin[1]*stride);

	#define qp (thread->qp)
	int process = thread->dct_delta;
	int error;
	/*unsigned char *img = (unsigned char *)(frame->plane[0]);
	//printf("qp: %d f_value: %d qbits: %d\n",qp,f_value,qbits);
	char temp[10];
	char *testString = "255\n";
	char *testW = "P2\n";
	char testH[40];
	char fileName[50];
	sprintf(testH,"%d %d\n",stride,region[1]);
	sprintf(fileName,"source_image%d%d.pgm",thread->curr_frame_nr,idx);
	FILE * pFile;
	char hex[10];
	pFile = fopen ( fileName , "wb" );
  	fwrite(testW,1,strlen(testW),pFile);
	fwrite(testH,1,strlen(testH),pFile);
	fwrite(testString,1,strlen(testString),pFile);
	int i;
  	for(i=ptr_offset;i<(gpu->image_info.image_width*gpu->image_info.image_height);i++){
		//unsigned char pixel = ((unsigned int)img[i]+(unsigned int)img[i+1])>>1;
  		sprintf(hex,"%3d", (unsigned char)img[i]);
		fwrite(hex,1,strlen(hex),pFile);
  		//fputc((int)',',pFile);
  		//fputc((unsigned char)img[i],pFile);
  		if(i % (gpu->image_info.image_width-1) == 0 && i > 0) fputc('\n',pFile);
  		else fputc(' ',pFile);
  		
  	}
  	fclose (pFile);
	//if(thread->dctWrIdx == 3 && idx == 3) exit(0);*/

	if(idx == 0) {
		//printf("UDATE_DCT %p\n",&(element->g_ref));
		error = clSetKernelArg(gpu->dct16x16_kernel,0,sizeof(cl_mem),&element->g_image);
		error = clSetKernelArg(gpu->dct16x16_kernel,1,sizeof(cl_mem),&element->g_ref);
		error = clSetKernelArg(gpu->dct16x16_kernel,2,sizeof(cl_mem),&element->g_device_result);
		error = clSetKernelArg(gpu->dct16x16_kernel,3,sizeof(cl_mem),&element->g_dct_result);
		error = clSetKernelArg(gpu->dct16x16_kernel,4,sizeof(cl_mem),&element->g_nz_result);
		error = clSetKernelArg(gpu->dct16x16_kernel,6,sizeof(cl_int),&qp);
	}
	//size_t global_size1[2] = {30720,68};
	size_t local_size1[2] = {256,1};
	size_t global_size2[2] = {image_mb_width*256,thread->dct_delta};
	size_t size_dct = sizeof(int16_t)*image_mb_width*image_mb_height*256/(thread->dct_count);
	size_t size_nz = sizeof(int)*image_mb_width*image_mb_height*16/(thread->dct_count);
	size_t size1 = sizeof(int16_t)*image_mb_width*image_mb_height*256/(thread->dct_count);
	size_t size2 = sizeof(int)*image_mb_width*image_mb_height*16/(thread->dct_count);

	int delta = process*idx;
	//printf("update (DCT buffer) in queue %d [thread->readIdx] with idx %d delta: %d\n",thread->dctWrIdx,idx,delta);
	error = clSetKernelArg(gpu->dct16x16_kernel,5,sizeof(cl_int),&delta);
	//x264_timer_start();
	error = clEnqueueWriteImage(*queue, element->g_ref, CL_FALSE,origin,region,
				    inRowPitch, 0, ((void *)frame->plane[0])+ptr_offset,
				    0, NULL, NULL);
				    clFinish(*queue);
	error = clEnqueueNDRangeKernel(*queue,gpu->dct16x16_kernel,2,NULL,global_size2,local_size1,0,NULL,NULL);
	WAIT();//clFinish(*queue);
	error = clEnqueueReadBuffer(*queue,element->g_dct_result,CL_FALSE,size_dct*idx,size1,element->dctPtr[idx],0,NULL,NULL);
	WAIT();//clFinish(*queue);
	//printf("UDPATE: resetting event for queue %d %d\n",thread->dctWrIdx,idx);
	error = clEnqueueReadBuffer(*queue,element->g_nz_result,CL_FALSE,size_nz*idx,size2,element->nz_ptr[idx],0,NULL,&(thread->g_dct_event[thread->dctWrIdx][idx]));
    WAIT();//	clFinish(*queue);
	/*x264_timer_stop();
	x264_print_time();*/
	pthread_mutex_lock(&(thread->pushtex));
	thread->pushIdx[thread->dctWrIdx][idx] = -1;
	//printf("reset: %d %d [%d]\n",thread->dctWrIdx,idx,thread->pushIdx[thread->dctWrIdx][idx]);
	thread->dctWrIdx = (thread->dctWrIdx+(idx == (thread->dct_count-1))) & THREAD_BUFFER_DEPTH_MASK;
	element->active[idx] = 1;
	pthread_mutex_unlock(&(thread->pushtex));
	#undef qp
}

void x264_block(x264_t *h, x264_gpu_thread_t *thread, int idx) {
	x264_gpu_buffer_t *element = &(thread->buffer[thread->readIdx]); 	// because of pre computation
	x264_gpu_t *gpu = thread->config;
	cl_command_queue *queue = &(gpu->queue[thread->readIdx]);
	clFinish(*queue);
}

void x264_gpu_push_x16_dct(x264_t *h, x264_gpu_thread_t *thread, x264_gpu_buffer_t *element) {
/*	x264_frame_t *frame = h->fref0[0];
	x264_gpu_t *gpu = thread->config;
	size_t origin[3] = {0,0,0};
	size_t region[3] = {1984,1088,1};
	size_t inRowPitch = 1984;

	int qp = h->mb.i_qp;
	int qbits = 15 + qp/6;
	int f_value = 1<<qbits;
	f_value /= 6;

	printf("qp: %d f_value: %d qbits: %d\n",qp,f_value,qbits);

	int error = clSetKernelArg(gpu->dct16x16_kernel,0,sizeof(cl_mem),&element->g_image);
	error = clSetKernelArg(gpu->dct16x16_kernel,1,sizeof(cl_mem),&element->g_ref);
	error = clSetKernelArg(gpu->dct16x16_kernel,2,sizeof(cl_mem),&element->g_device_result);
	error = clSetKernelArg(gpu->dct16x16_kernel,3,sizeof(cl_mem),&element->g_dct_result);
	error = clSetKernelArg(gpu->dct16x16_kernel,4,sizeof(cl_mem),&element->g_nz_result);
	error = clSetKernelArg(gpu->dct16x16_kernel,6,sizeof(cl_int),&qp);
	error = clEnqueueWriteImage(gpu->cmd_queue, element->g_ref, CL_FALSE,origin,region,
									inRowPitch, 0, (void *)frame->plane[0],
									  0, NULL, NULL);
	//size_t global_size1[2] = {30720,68};
	size_t local_size1[2] = {256,1};
	size_t global_size2[2] = {30720,17};
	size_t size_dct = sizeof(int16_t)*120*68*256/4;
	size_t size_nz = sizeof(int)*120*68*16/4;
	size_t size1 = sizeof(int16_t)*120*68*256/4;
	size_t size2 = sizeof(int)*120*68*16/4;

	int delta = 0;
	error = clSetKernelArg(gpu->dct16x16_kernel,5,sizeof(cl_int),&delta);
	error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->dct16x16_kernel,2,NULL,global_size2,local_size1,0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_dct_result,CL_FALSE,0,size1,element->dctPtr[0],0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_nz_result,CL_FALSE,0,size2,element->nz_ptr[0],0,NULL,&(element->g_dct_event[0]));

	int delta1 = 17;
	error = clSetKernelArg(gpu->dct16x16_kernel,5,sizeof(cl_int),&delta1);
	error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->dct16x16_kernel,2,NULL,global_size2,local_size1,0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_dct_result,CL_FALSE,size_dct,size1,element->dctPtr[1],0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_nz_result,CL_FALSE,size_nz,size2,element->nz_ptr[1],0,NULL,&(element->g_dct_event[1]));

	int delta2 = 34;
	error = clSetKernelArg(gpu->dct16x16_kernel,5,sizeof(cl_int),&delta2);
	error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->dct16x16_kernel,2,NULL,global_size2,local_size1,0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_dct_result,CL_FALSE,size_dct*2,size1,element->dctPtr[2],0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_nz_result,CL_FALSE,size_nz*2,size2,element->nz_ptr[2],0,NULL,&(element->g_dct_event[2]));

	int delta3 = 51;
	error = clSetKernelArg(gpu->dct16x16_kernel,5,sizeof(cl_int),&delta3);
	error = clEnqueueNDRangeKernel(gpu->cmd_queue,gpu->dct16x16_kernel,2,NULL,global_size2,local_size1,0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_dct_result,CL_FALSE,size_dct*3,size1,element->dctPtr[3],0,NULL,NULL);
	error = clEnqueueReadBuffer(gpu->cmd_queue,element->g_nz_result,CL_FALSE,size_nz*3,size2,element->nz_ptr[3],0,NULL,&(element->g_dct_event[3]));

	element->ptr_index=-1;
	//x264_print_time();*/
}

#define Y4M_MAGIC "YUV4MPEG2"
#define MAX_YUV4_HEADER 80
#define Y4M_FRAME_MAGIC "FRAME"
#define MAX_FRAME_HEADER 80

void x264_init_gpu_thread(x264_t *h264, x264_gpu_t *gpu, x264_gpu_thread_t *d2, char *handle) {
	d2->x264_handle = h264;
	d2->config = gpu;
	d2->hit_count = 0;
	d2->miss_count = 0;
	d2->frame_count = 0;
	d2->running = 0;
	d2->curr_frame_nr = 0;
	d2->readIdx = 0;
	d2->writeIdx = 0;
	d2->result = 0;
	d2->dctRdIdx = 0;
	d2->element = 0;
	d2->dctWrIdx = 0;
	d2->dctWrIdx2 = 0;
	d2->init = 0;

	d2->p_read_frame = x264_gpu_read_frame;
	d2->p_get_frame_total = x264_gpu_get_frame_total;

	// init file
	int  i, n, d;
	for(n=0;n<4;n++)
		for(i=0;i<4;i++)
			d2->pushIdx[n][i] = -1;

	char header[MAX_YUV4_HEADER+10];
	char *tokstart, *tokend, *header_end;
	d2->filehandle = malloc(sizeof(y4m_file_handle_t));
	y4m_file_handle_t *h = d2->filehandle;
	
	h->next_frame = 0;
	
	if( !strcmp(handle, "-") )
		h->fh = stdin;
	else
		h->fh = fopen(handle, "rb");
	if( h->fh == NULL )
		return -1;
	
	h->frame_header_len = strlen(Y4M_FRAME_MAGIC)+1;
	
	/* Read header */
	for( i=0; i<MAX_YUV4_HEADER; i++ )
	{
		header[i] = fgetc(h->fh);
		if( header[i] == '\n' )
		{
		/* Add a space after last option. Makes parsing "444" vs
		"444alpha" easier. */
		header[i+1] = 0x20;
		header[i+2] = 0;
		break;
		}
	}
	if( i == MAX_YUV4_HEADER || strncmp(header, Y4M_MAGIC, strlen(Y4M_MAGIC)) )
		return -1;
	
	/* Scan properties */
	header_end = &header[i+1]; /* Include space */
	h->seq_header_len = i+1;
	for( tokstart = &header[strlen(Y4M_MAGIC)+1]; tokstart < header_end; tokstart++ )
	{
		if(*tokstart==0x20) continue;
		switch(*tokstart++)
		{
		case 'W': /* Width. Required. */
		h->width = strtol(tokstart, &tokend, 10);
		tokstart=tokend;
		break;
		case 'H': /* Height. Required. */
		h->height = strtol(tokstart, &tokend, 10);
		tokstart=tokend;
		break;
		case 'C': /* Color space */
		if( strncmp("420", tokstart, 3) )
		{
			fprintf(stderr, "Colorspace unhandled\n");
			return -1;
		}
		tokstart = strchr(tokstart, 0x20);
		break;
		case 'I': /* Interlace type */
		switch(*tokstart++)
		{
		case 'p': break;
		case '?':
		case 't':
		case 'b':
		case 'm':
		default:
			fprintf(stderr, "Warning, this sequence might be interlaced\n");
		}
		break;
		case 'F': /* Frame rate - 0:0 if unknown */
		if( sscanf(tokstart, "%d:%d", &n, &d) == 2 && n && d )
		{
			//x264_reduce_fraction( &n, &d );
			//p_param->i_fps_num = n;
			//p_param->i_fps_den = d;
		}
		tokstart = strchr(tokstart, 0x20);
		break;
		case 'A': /* Pixel aspect - 0:0 if unknown */
		/* Don't override the aspect ratio if sar has been explicitly set on the commandline. */
		if( sscanf(tokstart, "%d:%d", &n, &d) == 2 && n && d)
		{
//  			x264_reduce_fraction( &n, &d );
// 			p_param->vui.i_sar_width = n;
// 			p_param->vui.i_sar_height = d;
		}
		tokstart = strchr(tokstart, 0x20);
		break;
		case 'X': /* Vendor extensions */
		if( !strncmp("YSCSS=",tokstart,6) )
		{
			/* Older nonstandard pixel format representation */
			tokstart += 6;
			if( strncmp("420JPEG",tokstart,7) &&
			strncmp("420MPEG2",tokstart,8) &&
			strncmp("420PALDV",tokstart,8) )
			{
			fprintf(stderr, "Unsupported extended colorspace\n");
			return -1;
			}
		}
		tokstart = strchr(tokstart, 0x20);
		break;
		}
	}

	// init mutex
	pthread_mutex_init(&(d2->mutex),NULL);
	pthread_mutex_init(&(d2->pushtex),NULL);
	pthread_cond_init(&(d2->cond),NULL);

	for(i=0;i<FRAME_BUFFER_SIZE;i++) {
		d2->frame_buffer[i] = x264_frame_new(d2->x264_handle);
	}
	d2->frame_buffer_read = 0;
	d2->dct16x16result = 0;// gpu->dct16x16result;

	// count
	int stride = d2->frame_buffer[0]->i_stride[0];

	int lines = d2->frame_buffer[0]->i_lines[0];
	int image_mb_width = (stride-64) >> 4;
	int image_mb_height = lines >> 4;

	d2->img_mb_width = image_mb_width;
	d2->img_mb_height = image_mb_height;

	d2->dct_count = 1;
	d2->dct_delta = image_mb_height;

	for(i=4;i>0;--i) {
		if((image_mb_height % i) == 0) {
			d2->dct_delta = image_mb_height / i;
			d2->dct_count = i;
			break;
		}
	}

	for(i=0;i<THREAD_BUFFER_DEPTH;i++)
		x264_init_buffer_element(d2,&(d2->buffer[i]),i);

	printf("SET: dct_count to %d and dct_delta to %d\n",d2->dct_count,d2->dct_delta);
}

void x264_gpu_thread_start(x264_gpu_thread_t *d) {
	pthread_mutex_lock(&(d->mutex));
	d->running = 1;
	d->init = 0;
	pthread_create(&(d->thread),NULL,x264_gpu_thread_run,(void *)d);
	pthread_mutex_unlock(&(d->mutex));
}

void x264_gpu_thread_stop(x264_gpu_thread_t *d) {
	pthread_mutex_lock(&(d->mutex));
	d->running = 0;
	pthread_mutex_unlock(&(d->mutex));
}


void x264_gpu_thread_pop_result(x264_t *h, x264_gpu_thread_t *gpu_thread) {
	int idx = gpu_thread->readIdx;
	//printf("want to pop ... x264_gpu_thread_pop_result\n");
	if(gpu_thread->number >= gpu_thread->curr_frame_nr-1 || gpu_thread->buffer[idx].status == RUNNING) gpu_thread->miss_count++;
	while(gpu_thread->number >= gpu_thread->curr_frame_nr-1 || gpu_thread->buffer[idx].status == RUNNING) {
		pthread_yield();
	}
	//printf("popped ... x264_gpu_thread_pop_result\n");
	pthread_mutex_lock(&(gpu_thread->mutex));
	gpu_thread->readIdx = (gpu_thread->readIdx+1)%THREAD_BUFFER_DEPTH;
	gpu_thread->number++;
	gpu_thread->hit_count++;
	gpu_thread->result = gpu_thread->buffer[idx].result;
	//x264_gpu_push_x16_dct(h,gpu_thread,&(gpu_thread->buffer[idx]));
	gpu_thread->dct16x16result = gpu_thread->buffer[gpu_thread->dctRdIdx].dct16x16result;
	gpu_thread->nz = gpu_thread->buffer[gpu_thread->dctRdIdx].nz;
	gpu_thread->element = &(gpu_thread->buffer[idx]);
	gpu_thread->element->ptr_index = -1;
	//printf("POP (%d buffer element)\n",idx);
	pthread_mutex_unlock(&(gpu_thread->mutex));
}

x264_frame_t* getUnusedFrame(x264_gpu_thread_t *thread) {
	thread->frame_buffer_read++;
	thread->frame_buffer_read &= FRAME_BUFFER_MASK;
	//printf("use frame: %p at index: %d\n",thread->frame_buffer[thread->frame_buffer_read],thread->frame_buffer_read);
	return thread->frame_buffer[thread->frame_buffer_read];
}

void *x264_gpu_thread_run(void *attr) {
	/// init phase
	x264_gpu_thread_t *data = (x264_gpu_thread_t *)attr;
	x264_t *h = data->x264_handle;

	int i;
	data->frame_count = get_frame_total_y4m(data->filehandle);
	//printf("number of frames: %d\n",data->frame_count);
    	x264_picture_t pic;
	x264_frame_t *image;// = x264_frame_new(data->x264_handle);
	x264_frame_t *ref = getUnusedFrame(data);//x264_frame_new(data->x264_handle);
	x264_frame_t *temp;

	//printf("width: %d height: %d\n",data->config->image_info.image_width,data->config->image_info.image_height);
	x264_picture_alloc(&pic,X264_CSP_I420,data->config->image_info.image_width,data->config->image_info.image_height);
	
	data->p_read_frame(&pic,data->filehandle,data->curr_frame_nr);
	data->curr_frame_nr++;
	x264_frame_copy_picture(data->x264_handle,ref,&pic);
	if( h->param.i_width != 16 * h->sps->i_mb_width ||
            h->param.i_height != 16 * h->sps->i_mb_height )
            x264_frame_expand_border_mod16( h, ref );

	for(i=0;i<THREAD_BUFFER_DEPTH-1;i++) {
		/// read in first image
		data->p_read_frame(&pic,data->filehandle,data->curr_frame_nr);
		image = getUnusedFrame(data);//x264_frame_new(data->x264_handle);
		x264_frame_copy_picture(data->x264_handle,image,&pic);
		if( h->param.i_width != 16 * h->sps->i_mb_width ||
            	    h->param.i_height != 16 * h->sps->i_mb_height )
            		x264_frame_expand_border_mod16( h, image );
		
		//printf("update buffer element %d\n",i);
		x264_update_buffer_element(data,&(data->buffer[i]),image,ref);
		ref = image;
		data->writeIdx++;
		data->curr_frame_nr++;
	}

	data->readIdx = 0;
	data->init = 1;
 	printf("init phase completed %d\n",data->running);

	x264_gpu_buffer_t *element;
	cl_int execution_status;
	size_t retSize;
	int diff;
	
	int status[THREAD_BUFFER_DEPTH];

	while(data->running) {
		// check for all actual threads
		for(i=0;i<THREAD_BUFFER_DEPTH;i++) {
			element = &(data->buffer[i]);
			if(element->status == RUNNING) {
				clGetEventInfo(element->g_event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(cl_int),&execution_status,&retSize);
				if(retSize == 4) {
				   /* if(status[i] != execution_status) {
				    printf("status[%d] switched: %d\n",i,execution_status);
				   status[i] = execution_status;
				}*/
					if(execution_status == CL_COMPLETE) {
						element->status = FINISHED;
						//printf("element %p has finished\n",&element->g_ref);
					} else {
					   element->status = RUNNING;
				    }
				} else {
				}
			}
		}

		for(i=0;i<PTR_MAX;i++) {
			if(data->pushIdx[data->dctWrIdx][i] > -1) {
				x264_gpu_udpate_dct(h,data,data->pushIdx[data->dctWrIdx][i]);
			}
		}

		pthread_mutex_lock(&(data->mutex));
		diff = (data->readIdx+THREAD_BUFFER_DEPTH)-data->writeIdx;
		diff &= THREAD_BUFFER_DEPTH_MASK;
		if(diff > 2 && data->curr_frame_nr < data->frame_count) {
			//printf("WRITE: write frame to %d (%d of %d)\n",data->writeIdx,data->curr_frame_nr,data->frame_count);
			data->p_read_frame(&pic,data->filehandle,data->curr_frame_nr+data->i_seek);
			//printf("pushing frame %d to gpu pipeline data->i_seek: %d\n",data->curr_frame_nr,data->i_seek);
			data->curr_frame_nr++;
			//if(data->curr_frame_nr == 25) exit(0);
			//if(element->image != 0) image = element->image;
			/*else */image = getUnusedFrame(data);//x264_frame_new(data->x264_handle);
			x264_frame_copy_picture(data->x264_handle,image,&pic);
			if( h->param.i_width != 16 * h->sps->i_mb_width ||
        	    	    	h->param.i_height != 16 * h->sps->i_mb_height ) {
	            		x264_frame_expand_border_mod16( h, image );
			}
			x264_update_buffer_element(data,&(data->buffer[data->writeIdx]),image,ref);
			ref = image;
			data->writeIdx = (data->writeIdx+1) % THREAD_BUFFER_DEPTH;
		}

		pthread_mutex_unlock(&(data->mutex));
		pthread_yield();
	}
	pthread_exit(0);
	// clear everything
}

int x264_gpu_read_frame( x264_picture_t *p_pic, y4m_file_handle_t *h, int i_frame) {
    int          slen = strlen(Y4M_FRAME_MAGIC);
    int          i    = 0;
    char         header[16];

    if( i_frame != h->next_frame )
    {
        if (fseek(h->fh, (uint64_t)i_frame*(3*(h->width*h->height)/2+h->frame_header_len)
                  + h->seq_header_len, SEEK_SET))
            return -1;
    }

    /* Read frame header - without terminating '\n' */
    if (fread(header, 1, slen, h->fh) != slen)
        return -1;

    header[slen] = 0;
    if (strncmp(header, Y4M_FRAME_MAGIC, slen))
    {
        fprintf(stderr, "Bad header magic (%"PRIx32" <=> %s)\n",
                *((uint32_t*)header), header);
        return -1;
    }

    /* Skip most of it */
    while (i<MAX_FRAME_HEADER && fgetc(h->fh) != '\n')
        i++;
    if (i == MAX_FRAME_HEADER)
    {
        fprintf(stderr, "Bad frame header!\n");
        return -1;
    }
    h->frame_header_len = i+slen+1;

    if( fread(p_pic->img.plane[0], 1, h->width*h->height, h->fh) <= 0
        || fread(p_pic->img.plane[1], 1, h->width * h->height / 4, h->fh) <= 0
        || fread(p_pic->img.plane[2], 1, h->width * h->height / 4, h->fh) <= 0)
        return -1;

    h->next_frame = i_frame+1;

    return 0;
}

int x264_gpu_get_frame_total( y4m_file_handle_t *handle )
{
    int          i_frame_total = 0;
    uint64_t     init_pos      = ftell(handle->fh);

    if( !fseek( handle->fh, 0, SEEK_END ) )
    {
        uint64_t i_size = ftell( handle->fh );
        fseek( handle->fh, init_pos, SEEK_SET );
        i_frame_total = (int)((i_size - handle->seq_header_len) /
                              (3*(handle->width*handle->height)/2+handle->frame_header_len));
    }

    return i_frame_total;
}

#undef Y4M_MAGIC
#undef MAX_YUV4_HEADER
#undef Y4M_FRAME_MAGIC
#undef MAX_FRAME_HEADER

#endif
