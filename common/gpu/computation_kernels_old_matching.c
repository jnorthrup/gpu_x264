	/*
 * Kernel functions for x264 accerlation through OpenCL
 */

#define MAX_REFERENCE_SIZE	4
#define COST_MAX 65535
#define GROUP_WRAP 2
#define GROUP_MID 4
#define STEP_START_SIZE 4
#define NUMBER_OF_NEIGHBOURS 8

typedef struct workItemResult_t {
	int2 	mv_predicted;			// predicted motion vector
	int2 	mv_diff;			// best match diff vector
	unsigned int costs;			// best match cost
	char 	refIdx;				// best match reference index
} workItemResult __attribute__((aligned));


// get reference costs in exp golomb coding bits
int createRefCost(int ref) {
	if(ref == 0) return 0;
	int bins;
	bins = 16 - clz(ref);
	return ((bins<<1) - 1);
}
/*
 create costs based on the mv difference       
 */
int createMVDCost(int i_lambda, int x, int y) {
	int costs = 0;
	x = abs(x);
	y = abs(y);
	// created costs by mvd
	
	// copy of the source inside x.264 to keep compatibility
	costs = i_lambda * (log2((float)x+1)*2 + 0.718f + !!(x)) + .5f;
	costs += i_lambda * (log2((float)y+1)*2 + 0.718f + !!(y)) + .5f;
	return costs;
}

#define A 0
#define B 1
#define C 2

int2 median(int2 mvp_a, int2 mvp_b, int2 mvp_c) {
	int2 diff;
	diff = mvp_a + mvp_b + mvp_c;
	diff -= min(mvp_a,min(mvp_b,mvp_c));
	diff -= max(mvp_a,max(mvp_b,mvp_c));
	return mvp_a;
}

#define IS_POSITIVE(x) {\
	if(x > 0) {\
		x = 1;\
	} else {\
		x = 0;\
	}};

#define COPY_STRUCT(dst,src) \
	dst.costs = src.costs;\
	dst.mv_diff.x = src.mv_diff.x;\
	dst.mv_diff.y = src.mv_diff.y;\
	dst.mv_predicted.x = src.mv_predicted.x;\
	dst.mv_predicted.y = src.mv_predicted.y;\
	dst.refIdx = src.refIdx;


// selection kernel
/*
 @param	data				data to process must be read and write able
 @param	iteration			current iteration process
 iteration goes from 0 ... (image_mb_height - 1)*2 + image_mb_width
 @param	image_mb_width		image width in macroblocks
 @param image_mb_height		image height in macroblocks
 @param assitance_number	number assistance points per sample
 @param result				structure to write to
 */
__kernel void SelectionBestMatchKernel(	__global workItemResult *data,
					const int iteration, const int image_mb_widht, const int image_mb_height,
					const int assistance_number,
					__global char *debug,
					__global workItemResult *result,
					__local workItemResult *neighbour,
					__local workItemResult *buffer) {
	unsigned int i, it;
	int local_size, delta, delta2;
	int validA, validB, validC;

	validA = validB = validC = 0;
	
	int global_id_x = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int global_id_y = get_group_id(1)*get_local_size(1);
	
	int global_id = get_group_id(1)*get_local_size(0)*get_num_groups(0)+global_id_x;
	
	int global_id_load = global_id * assistance_number;
	int maxIterations = ((image_mb_height-1)<<1 + image_mb_widht);
	int x_block_pos;
	
		x_block_pos = iteration - (get_group_id(1)<<1);
		
		// VERIFIRED TILL HERE .. 
		
		if(x_block_pos >= 0 && x_block_pos < image_mb_widht && global_id_x == x_block_pos) {
			workItemResult temp;
			// verified till here... does the right positioning
			
			// buffer assistance points
			local_size = get_local_size(0);
			delta = get_local_id(0)*assistance_number;
			delta2 = get_local_id(0)*3;
			// chache in local memory
			for(i=0;i<assistance_number;i++) {
				COPY_STRUCT(buffer[i+delta],data[global_id_load+i]);
			}

			
			// temp element used if a is not available
			temp.refIdx = -1;
			temp.mv_predicted.x = 0;
			temp.mv_predicted.y = 0;
			
			// here is a new excpetion that must be considered
			// what if the macroblock is the first in the current row to be processes... no A exists
			validA = (global_id-1)>0 && ((global_id) % (local_size*get_num_groups(0)) != 0);
			validB = global_id-local_size*get_num_groups(0);
			validC = global_id-local_size*get_num_groups(0)+1;
			IS_POSITIVE(validB);			
			IS_POSITIVE(validC);
		
			// take care of processing the first macroblock where no A block exists
			if(validA == 1) {
				COPY_STRUCT(neighbour[A+delta2],result[global_id-1]);
			} else { 
				COPY_STRUCT(neighbour[A+delta2],temp);
			}

			barrier(CLK_LOCAL_MEM_FENCE);
//			return;
			
			// check for availability
			if((validB == 0) && (validC == 0)) {
				COPY_STRUCT(neighbour[B+delta2],neighbour[A+delta2]);
				COPY_STRUCT(neighbour[C+delta2],neighbour[A+delta2]);
			}

			if(validB == 1) {
				COPY_STRUCT(neighbour[B+delta2],result[global_id-local_size*get_num_groups(0)]);
			} else {
				COPY_STRUCT(neighbour[B+delta2],temp);
			}
			if(validC == 1) {
				COPY_STRUCT(neighbour[C+delta2],result[global_id-local_size*get_num_groups(0)+1]);
			} else {
				COPY_STRUCT(neighbour[C+delta2],temp);
			}
			
			int ref_match;
			
			uint2 diff[4];
			int2 diff_signed[4];
			unsigned int curDiff_x;
			unsigned int curDiff_y;
			unsigned int stored_index=assistance_number;
			
			int2 mv_p = temp.mv_predicted;
			
			curDiff_x = 65535;
			curDiff_y = 65535;
			
			unsigned int temp_diff_x = curDiff_x;
			unsigned int temp_diff_y = curDiff_y;
			
			for(i=0;i<assistance_number;i++) {
				ref_match = 0;
				ref_match += (neighbour[A+delta2].refIdx == buffer[i+delta].refIdx);
				ref_match += (neighbour[B+delta2].refIdx == buffer[i+delta].refIdx) << 1;
				ref_match += (neighbour[C+delta2].refIdx == buffer[i+delta].refIdx) << 2;
				
				mv_p = temp.mv_predicted;
				
				mv_p = (ref_match == 1) ? neighbour[A+delta2].mv_predicted+neighbour[A+delta2].mv_diff : mv_p;
				mv_p = (ref_match == 2) ? neighbour[B+delta2].mv_predicted+neighbour[B+delta2].mv_diff : mv_p;
				mv_p = (ref_match == 4) ? neighbour[C+delta2].mv_predicted+neighbour[C+delta2].mv_diff : mv_p;
				
				
				if(mv_p.x == temp.mv_predicted.x && mv_p.y == temp.mv_predicted.y) {
					mv_p = median(neighbour[A+delta2].mv_predicted,
								  neighbour[B+delta2].mv_predicted,
								  neighbour[C+delta2].mv_predicted);
				}
				
				// create new diff
				
				//return;
				diff[i] = abs_diff(buffer[i+delta].mv_predicted,mv_p);
				diff_signed[i] = buffer[i+delta].mv_predicted - mv_p;
				
				temp_diff_x = diff[i].x;
				temp_diff_y = diff[i].y;
				
//3return;
				// rememeber best match motion vector difference and index
				if(temp_diff_x <= curDiff_x && temp_diff_y <= curDiff_y) {
					curDiff_x = temp_diff_x;
					curDiff_y = temp_diff_y;
					stored_index = i;
				}
			}
			if(stored_index == assistance_number) return;
			// write best result back to result
			buffer[stored_index+delta].mv_diff += diff_signed[stored_index];
			buffer[stored_index+delta].mv_predicted = mv_p;			/// wrong interpretation
			COPY_STRUCT(result[global_id],buffer[stored_index+delta]);
		}
}

/*
 (mb_width*mb_height) > local_size
 @param ref				input reference image
 @param refIdx			current refIdx
 @param image_width		width of ref image and the current image
 @param image_height		height of ref image and the current image
 @param image			current image to do motion estimation on
 @param debug_output		for debugging purpose
 @param mb_width			macroblock width
 @param mb_height		macroblock height
 @param serach_region	serach region ... number of pixels from the current mb to the left and right
 @param current_mb		local store cache for the current mb per work group
 @param region_buffer	local store cache for the current search region per work group
 @param result			resulting mvp vectors SIZE: number of work items
 */

#define assistance_points 4

__kernel void
GatherMotionKernel(__global const unsigned char *ref,
								 const int refIdx,
								 const int image_width,
								 const int image_height,
								 __global const unsigned char *image,
								 __global char *debug_output,
								 int mb_width,
								 int mb_height,
								 int search_region,
								 __local workItemResult *local_result,
								 __local unsigned char *current_mb,
								 __local unsigned char *region_buffer,
								 __local int *neighbour,
								 __local unsigned int *curCost,
								 __local unsigned int *tempCost,
								 __local short *temp_x,
								 __local short *temp_y,
								 __global workItemResult *result) {
	
	// neighbor [assistance_points * 3 * NUMBER_OF_NEIGHBORS]
	
	int x, y, i, w;
	__global workItemResult* outPtr;
	__global unsigned char *ptr;
	
	/* get position of the current macroblock and the current virt. assistance point */
	int mb_x = get_group_id(0);
	int mb_y = get_group_id(1);
	
	int local_size = get_local_size(0);
	
	int ass_number = get_local_id(0) / NUMBER_OF_NEIGHBOURS;
	int local_id = get_local_id(0) % NUMBER_OF_NEIGHBOURS;
	int local_id_load = get_local_id(0);
	
	// offset in the current image to gather macroblock from
	int offset = mb_y*mb_height*image_width+mb_x*mb_width;
	
	// init local buffer
	if(local_id < assistance_points) curCost[local_id] = COST_MAX;
	
	/* load current_mb local cache */
	
	ptr = image + offset;
	prefetch(ptr,mb_width);
	i = local_id_load;
	for(y=0;y<mb_height;y++) {
		prefetch(ptr+image_width,mb_width);
		for(x=local_id_load;x<mb_width;x+=local_size) {
			current_mb[i] = *(ptr + x);
			i+=mb_width;
		}
		ptr += image_width;
	}
	
	/// needs 18ms till here
	
	/* estimate search region */
	
#define MAX(value,diff,max_value) \
((value + diff) > max_value) ? (max_value - value) : diff
	
	offset = offset-(min(mb_x*mb_width,search_region));
	offset -= (min(mb_y*mb_height*image_width,search_region*image_width));
	
	int temp_width = MAX(mb_x*mb_width,search_region+mb_width,image_width);
	temp_width += min(mb_x*mb_width,search_region);
	int temp_height = MAX(mb_y*mb_height,search_region+mb_height,image_height);
	temp_height += min(mb_y*mb_height,search_region);
	
	/* load search region into local cache */
	ptr = ref+offset;
	prefetch(ptr,temp_width);
	i=local_id_load;
	for(y=0;y<temp_height;y++) {
		prefetch(ptr+image_width,temp_width);
		for(x=local_id_load;x<temp_width;x+=local_size) {
			region_buffer[i] = *(ptr + x);
			i+=temp_width;
		}
		ptr += image_width;
	}
	// --- 73 ms till here (time for 16x16 load 29ms ... 48x48 ... us be about a factor of 9 higher)
	// do wait for all threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int step_size = STEP_START_SIZE;
	
	int nei_x[8] = {-1, 0, 1,-1,1,-1,0,1};
	int nei_y[8] = {-1,-1,-1, 0,0, 1,1,1};
	
	int ax	= temp_width/GROUP_WRAP*(ass_number%GROUP_WRAP);// + delta_region_width>>1; 
	int ay	= temp_height/GROUP_WRAP*(ass_number/GROUP_WRAP);// + delta_region_height>>1;
	
	// calc SAD
	offset = ay*temp_width+ax;
	local_result[ass_number].mv_predicted.x = ax;
	local_result[ass_number].mv_predicted.y = ay;
	local_result[ass_number].refIdx = refIdx;
	local_result[ass_number].mv_diff.x = 0;
	local_result[ass_number].mv_diff.y = 0;
	
	int mvd_x_n, mvd_y_n;

	w = (sizeof(int)<<3) - clz(mb_width-1);

#define GET_COORD(coord) \
	 ((coord>>w) + coord&(mb_width-1))
	
	if(local_id == 0) {
		
		mvd_x_n = (mb_width * mb_height)>>2;
		mvd_y_n = 0;
		for(y=0;y<4;y++) {
			i = y;
			for(x=0;x<mvd_x_n;x+=4) {
				mvd_y_n += abs_diff(current_mb[i],region_buffer[offset+GET_COORD(i)]);
				mvd_y_n += abs_diff(current_mb[i+4],region_buffer[offset+GET_COORD(i+4)]);
				mvd_y_n += abs_diff(current_mb[i+8],region_buffer[offset+GET_COORD(i+8)]);
				mvd_y_n += abs_diff(current_mb[i+12],region_buffer[offset+GET_COORD(i+12)]);
				i += 16;
			}
		}
		/*for(y=0;y<mb_height;y++) {
			for(x=0;x<4;x++) {
				mvd_y_n += abs_diff(current_mb[mvd_x_n+x+0],region_buffer[offset+x+0]);
				mvd_y_n += abs_diff(current_mb[mvd_x_n+x+4],region_buffer[offset+x+4]);
				mvd_y_n += abs_diff(current_mb[mvd_x_n+x+8],region_buffer[offset+x+8]);
				mvd_y_n += abs_diff(current_mb[mvd_x_n+x+12],region_buffer[offset+x+12]);
			}
			mvd_x_n+=mb_width;
			offset+=temp_width;
		}*/
		mvd_y_n += createRefCost(refIdx);
		mvd_y_n += createMVDCost(25,0,0);
		local_result[ass_number].costs = mvd_y_n;
		curCost[ass_number] = mvd_y_n;
		temp_x[ass_number] = ax;
		temp_y[ass_number] = ay;
	}

	step_size = STEP_START_SIZE;
	barrier(CLK_LOCAL_MEM_FENCE);

	/* do computation as long as there was no change in the curCost value or min step size has been reached */
	do {
		
		tempCost[ass_number] = curCost[ass_number];
		
		i = ((ass_number << 3) + local_id)*3;
		
#define X_INDEX 0
#define Y_INDEX 1
#define COST_INDEX 2
		
		neighbour[i+X_INDEX] = (step_size*nei_x[local_id])+temp_x[ass_number];
		if(neighbour[i+X_INDEX] < 0 || neighbour[i+X_INDEX] > temp_width) neighbour[i+X_INDEX] = -1;
		//neighbour[i+X_INDEX] = select(neighbour[i+X_INDEX],-1,neighbour[i+X_INDEX] < 0);
		//neighbour[i+X_INDEX] = select(neighbour[i+X_INDEX],-1,neighbour[i+X_INDEX] > temp_width);
		
		neighbour[i+Y_INDEX] = (step_size*nei_y[local_id])+temp_y[ass_number];
		if(neighbour[i+Y_INDEX] < 0 || neighbour[i+Y_INDEX] > temp_height) neighbour[i+Y_INDEX] = -1;
		//neighbour[i+Y_INDEX] = select(neighbour[i+Y_INDEX],-1,neighbour[i+Y_INDEX] < 0);
		//neighbour[i+Y_INDEX] = select(neighbour[i+Y_INDEX],-1,neighbour[i+Y_INDEX] > temp_height);
		
		neighbour[i+COST_INDEX] = COST_MAX;
		
		if((neighbour[i+Y_INDEX] > -1) && (neighbour[i+X_INDEX] > -1)) {
			
			// calc SAD
			// do optimization here... think about a methode how to do a shared memory
			// read with less conflicts
			offset = neighbour[i+Y_INDEX]*temp_width+neighbour[i+X_INDEX];
			mvd_x_n = (mb_width * mb_height)>>2;
			mvd_y_n = 0;
			for(y=0;y<4;y++) {
				i = y;
				for(x=0;x<mvd_x_n;x+=4) {
					mvd_y_n += abs_diff(current_mb[i],region_buffer[offset+GET_COORD(i)]);
					mvd_y_n += abs_diff(current_mb[i+4],region_buffer[offset+GET_COORD(i+4)]);
					mvd_y_n += abs_diff(current_mb[i+8],region_buffer[offset+GET_COORD(i+8)]);
					mvd_y_n += abs_diff(current_mb[i+12],region_buffer[offset+GET_COORD(i+12)]);
					i += 16;
				}
			}
			/*mvd_x_n = 0;
			mvd_y_n = 0;
			for(y=0;y<mb_height;++y) {
				for(x=0;x<4;x++) {
					mvd_y_n += abs_diff(current_mb[mvd_x_n+0+x],region_buffer[offset+x+0]);
					mvd_y_n += abs_diff(current_mb[mvd_x_n+4+x],region_buffer[offset+x+4]);
					mvd_y_n += abs_diff(current_mb[mvd_x_n+8+x],region_buffer[offset+x+8]);
					mvd_y_n += abs_diff(current_mb[mvd_x_n+12+x],region_buffer[offset+x+12]);
				}
				mvd_x_n += mb_width;
				offset += temp_width;
			}*/
			neighbour[i+COST_INDEX] = mvd_x_n;
			mvd_x_n = neighbour[i+X_INDEX]-ax;
			mvd_y_n = neighbour[i+Y_INDEX]-ay;
			neighbour[i+COST_INDEX] += createRefCost(refIdx);
			neighbour[i+COST_INDEX] += createMVDCost(25,mvd_x_n,mvd_y_n);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		// step_size = 0;
		
		if(local_id == 0) {
			y=COST_MAX;
			for(x=0;x<NUMBER_OF_NEIGHBOURS;x++) {
				i = ((ass_number << 3) + x)*3;
				if(neighbour[i+COST_INDEX] < curCost[ass_number]) {
					curCost[ass_number] = neighbour[i+COST_INDEX];
					y = i;
				}
			}
			if(y < COST_MAX) {
				local_result[ass_number].costs = neighbour[y+COST_INDEX];
				local_result[ass_number].mv_diff.x = neighbour[y+X_INDEX]-ax;
				local_result[ass_number].mv_diff.y = neighbour[y+Y_INDEX]-ay;
				temp_x[ass_number] = neighbour[y+X_INDEX];
				temp_y[ass_number] = neighbour[y+Y_INDEX];
			}
			
		}
		step_size = step_size >> 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		
	} while(step_size > 1 || tempCost[ass_number] != curCost[ass_number]);
	// write result back
	if(local_id == 0) {
		i = image_width / mb_width;
		x = mad24(mb_y,i,mb_x);
		x = x << 2;
		COPY_STRUCT(result[x + ass_number],local_result[ass_number]);
	}
}


