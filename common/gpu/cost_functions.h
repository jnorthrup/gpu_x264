/*
 * cost_functions.h
 *
 *  Created on: Aug 20, 2009
 *      Author: destinaton
 */
 
#ifdef GPU_OPTIMIZE

#ifndef COST_FUNCTIONS_H_
#define COST_FUNCTIONS_H_

#include "common/common.h"
#include "common/pixel.h"
#include "encoder/me.h"
#include "sys/mman.h"
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <pthread.h>

// include OpenCL
#ifdef MAC_DARWIN
#include <OpenCL/opencl.h>
#else 
#include <CL/cl.h>
#endif

#define MAX_REFERENCE_SIZE	16

typedef int bool;
#define false 0
#define true (-1)
#define FALSE false
#define TRUE true

// new data structures
typedef struct workItemResult_t {
	int mv_predicted;			// predicted motion vector
	int mv_diff;			// best match diff vector
	unsigned int costs;			// best match cost
	int 	refIdx;				// best match reference index
} workItemResult __attribute__((aligned));

static void gpu_log(char *str) {
	printf("ERROR in gpu code: %s\n",str);
}

typedef struct x264_gpu_image_t_ {
	/* width of the image */
	int image_width;
	/* height of the image */
	int image_height;
	/* serach region for each macroblock */
	int search_region;
	
	/* macroblock width */
	int mb_width;
	/* macroblock height */
	int mb_height;
	int image_mb_width;
	int image_mb_height;
	/* */
	int refIdx;
} x264_gpu_image_t;

typedef struct x264_gpu_t_ {
	/* computing device */
	cl_device_id device_id;
	/* computing queue */
	cl_command_queue cmd_queue;
	cl_command_queue queue[4];
	/* computing program */
	cl_program program;
	/* computing kernel's */
	cl_kernel me_gather_kernel;
	int mg_gather_set;
	cl_kernel me_selection_kernel;
	cl_kernel me_refinement_kernel;
	cl_kernel me_reset_grid;
	cl_kernel me_predict_nieghbors;
	cl_kernel me_select_best;
	cl_kernel me_select_best_mv;
	cl_kernel dct16x16_kernel;
	/* computing context */
	cl_context context;

	int *dct16x16result;
	
	/* work group size in 2d */
	size_t local_size[2];
	size_t sel_local_size[2];
	/* work group grid size in 2d */
	size_t global_size[2];
	size_t sel_global_size[2];
	/* default error */
	cl_int error;

	int calc;
	
	/* image memory object CL_MEM_READ_ONLY*/ 
	cl_mem image;
	cl_mem image2D;
	/* resulting memory object  CL_MEM_READ_WRITE */
	cl_mem result;
	cl_mem residual_result;
	/* reference image object CL_MEM_READ_ONLY */
	cl_mem reference;
	cl_mem reference2D;
	cl_mem nz_result;
	/* debug object CL_MEM_READ_WRITE */
	cl_mem debug;
	/* device_result CL_MEM_READ_WRITE */
	cl_mem device_result;
	cl_mem cost_result;
	cl_mem valid_grid;
	
	int iteration;
	int assistance_points;
	
	int result_read;
	
	/* image informations */
	x264_gpu_image_t image_info;
	
	workItemResult *result_data;
} x264_gpu_t;

void x264_gpu_processImage(x264_gpu_t *gpu, x264_frame_t *ref, x264_frame_t *image, int refIdx);
void x264_gpu_processImage_char(x264_gpu_t *gpu, char *ref, char *image, int refIdx);
void x264_read_result(x264_gpu_t *gpu);

void x264_gpu_reset(x264_gpu_t *gpu);

void x264_gpu_doSelection(x264_gpu_t *gpu, int iteration);

void x264_gpu_me_search_ref( x264_t *h, x264_gpu_t *gpu, x264_me_t *m, int i_mvc, int *p_halfpel_thresh );
void x264_gpu_mb_predict_mv_ref16x16( x264_t *h, x264_gpu_t *gpu, int i_list, int i_ref, int16_t mvc[9][2], int *i_mvc );
void x264_gpu_mb_predict_mv_16x16( x264_t *h, x264_gpu_t *gpu, int i_list, int i_ref, int16_t mvp[2] );
void x264_gpu_read_result(x264_gpu_t *gpu);

// ################ GPU THREAD MODULE ##################

#define THREAD_BUFFER_DEPTH 4
#define THREAD_BUFFER_DEPTH_MASK THREAD_BUFFER_DEPTH-1

typedef enum BUFFER_STATUS_t_ {
	RUNNING = 0,
	FINISHED
} BUFFER_STATUS;

#define PTR_MAX 4
#define PTR_MAX_MASK PTR_MAX-1

typedef struct _x264_gpu_buffer_t_ {
	volatile workItemResult *result;
	volatile int16_t *dct16x16result;
	volatile int *nz;
	volatile BUFFER_STATUS status;
	x264_frame_t *image;
	x264_frame_t *reference;
	x264_frame_t *fdec;

	int active[4];

	int *nz_ptr[4];
	int16_t *dctPtr[4];
	int ptr_index;
	
	cl_mem g_image;
	cl_mem g_ref;
	cl_mem g_result;
	cl_mem g_device_result;
	cl_mem g_dct_result;
	cl_mem g_nz_result;
	cl_mem g_grid;
	cl_mem g_cost_result;
	cl_event g_event;
} x264_gpu_buffer_t;

typedef struct {
    FILE *fh;
    int width, height;
    int next_frame;
    int seq_header_len, frame_header_len;
    int frame_size;
} y4m_file_handle_t;

#define FRAME_BUFFER_SIZE THREAD_BUFFER_DEPTH*4
#define FRAME_BUFFER_MASK FRAME_BUFFER_SIZE-1

typedef struct _x264_gpu_thread_t_ {
	x264_gpu_t *config;
	x264_t *x264_handle;
	
	workItemResult *result;
	int16_t *dct16x16result;
	int *nz;
	int *index;

	int qp;

	x264_gpu_buffer_t *element;
	
	// main thread
	pthread_t thread;
	pthread_mutex_t mutex, pushtex;
	pthread_cond_t cond;

	x264_gpu_buffer_t buffer[THREAD_BUFFER_DEPTH];

	int readIdx, writeIdx;
	int running, init;

	int dctRdIdx, dctWrIdx, dctWrIdx2;
	int pushIdx[PTR_MAX][4];

	int dct_count;
	int dct_delta;
	int img_mb_width, img_mb_height;

	cl_event g_dct_event[4][4];

	// x264 file pointer
	y4m_file_handle_t *filehandle;
	x264_frame_t *frame_buffer[FRAME_BUFFER_SIZE];
	int frame_buffer_read;

	volatile int curr_frame_nr, number;
	// x264 file information
	int frame_count;
	int hit_count;
	int miss_count;

	int i_seek;

	int (*p_read_frame)( x264_picture_t *p_pic, y4m_file_handle_t *handle, int i_frame );
	int (*p_get_frame_total)( y4m_file_handle_t *handle );
} x264_gpu_thread_t;

int x264_gpu_get_frame_total(y4m_file_handle_t *handle);
int x264_gpu_read_frame( x264_picture_t *p_pic, y4m_file_handle_t *handle, int i_frame );

//void x264_gpu_thread_run(x264_gpu_thread_t *d);
void *x264_gpu_thread_run(void *attr);

void x264_init_buffer_element(x264_gpu_thread_t *gpu, x264_gpu_buffer_t *element, int idx);
void x264_deinit_buffer_element(x264_gpu_buffer_t *element);
void x264_update_buffer_element(x264_gpu_thread_t *thread, x264_gpu_buffer_t *element, x264_frame_t *img, x264_frame_t *ref);

void x264_init_gpu_thread(x264_t *h, x264_gpu_t *gpu, x264_gpu_thread_t *d, char *filename);
void x264_block(x264_t *h, x264_gpu_thread_t *thread, int idx);
void x264_gpu_prepush_sub16x16_dct(x264_t *h, x264_gpu_thread_t *thread, int idx);
void x264_gpu_udpate_dct(x264_t *h, x264_gpu_thread_t *thread, int idx);
void x264_gpu_push_sub16x16_dct(x264_t *h, x264_gpu_thread_t *gpu, x264_gpu_buffer_t *element);
void x264_gpu_thread_pop_result(x264_t *h, x264_gpu_thread_t *gpu_thread);

void x264_gpu_thread_stop(x264_gpu_thread_t *d);
void x264_gpu_thread_start(x264_gpu_thread_t *d);

void x264_gpu_me_search_ref2( x264_t *h, x264_gpu_thread_t *gpu, x264_me_t *m, int i_mvc, int *p_halfpel_thresh );
void x264_gpu_mb_predict_mv_ref16x162( x264_t *h, x264_gpu_thread_t *gpu, int i_list, int i_ref, int16_t mvc[9][2], int *i_mvc );
void x264_gpu_mb_predict_mv_16x162( x264_t *h, x264_gpu_thread_t *gpu, int i_list, int i_ref, int16_t mvp[2] );

int x264_gpu_inter_sub16x16_dct( int16_t dct[16][4][4], x264_t *h, x264_gpu_thread_t *thread, int16_t nz[16]);

// ################ GPU THREAD MODULE ##################

// create easier access functions
/*
 */
int x264_gpu_init(x264_gpu_t *gpu);
int x264_gpu_deinit(x264_gpu_t *gpu);

/*
	set all needed kernel args
	and start kernels as seen in flow chart of the gpu estimation
 */

#endif /* COST_FUNCTIONS_H_ */

#endif // GPU_OPTIMIZE
