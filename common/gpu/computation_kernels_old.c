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
	int2 	mv_diff;				// best match diff vector
	unsigned int costs;					// best match cost
	char 	refIdx;					// best match reference index
} workItemResult __attribute__((aligned));


// get reference costs in exp golomb coding bits
inline int createRefCost(ushort ref) {
	if(ref == 0) return 0;
	int bins = 16 - clz(ref);
	return (bins<<1 - 1);
	//return 1;
}

/*
 create costs based on the mv difference       
 */
inline int createMVDCost(int i_lambda, int x, int y) {
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
__kernel void SelectionBestMatchKernel(__global workItemResult *data,
									   int iteration, int image_mb_widht, int image_mb_height,
									   int assistance_number,
									   __global char *debug,
									   __global workItemResult *result,
									   __local workItemResult *neighbour,
									   __local workItemResult *buffer) {
	
	unsigned int i, it;
	
	int global_id_x = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int global_id_y = get_group_id(1)*get_local_size(1);
	
	int global_id = get_group_id(1)*get_local_size(0)*get_num_groups(0)+global_id_x;
	
	int global_id_load = global_id * assistance_number;
	int maxIterations = ((image_mb_height-1)*2 + image_mb_widht);
	
	for(it=0;it<maxIterations;++it) {
		int x_block_pos = it - (get_group_id(1)<<1);
		
		// VERIFIRED TILL HERE .. 
		
		if(x_block_pos >= 0 && x_block_pos < image_mb_widht && global_id_x == x_block_pos) {
			// verified till here... does the right positioning
			
			// buffer assistance points
			int local_size = get_local_size(0);
			int delta = get_local_id(0)*assistance_number;
			int delta2 = get_local_id(0)*3;
			// chache in local memory
			for(i=0;i<assistance_number;i++) buffer[i+delta] = data[global_id_load+i];
			
			// temp element used if a is not available
			workItemResult temp;
			temp.refIdx = -1;
			temp.mv_predicted.x = 0;
			temp.mv_predicted.y = 0;
			
			// here is a new excpetion that must be considered
			// what if the macroblock is the first in the current row to be processes... no A exists
			int validA = (global_id-1)>0 && ((global_id) % (local_size*get_num_groups(0)) != 0);
			int validB = (global_id-local_size*get_num_groups(0));
			validB = validB > 0;
			int validC = (global_id-local_size*get_num_groups(0)+1);
			validC = validC>0;
			
			// take care of processing the first macroblock where no a exists
			neighbour[A+delta2] = (validA) ? result[global_id-1] : temp;
			
			// check for availability
			if(!validB && !validC) {
				neighbour[B+delta2] = neighbour[A+delta2];
				neighbour[C+delta2] = neighbour[A+delta2];
			} else {
				neighbour[B+delta2] = (validB) ? result[global_id-local_size*get_num_groups(0)] : temp;
				neighbour[C+delta2] = (validC) ? result[global_id-local_size*get_num_groups(0)+1] : temp;
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
			result[global_id] = buffer[stored_index+delta];
		}
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

#define MB_COUNT 8100
#define THREADS_PER_VIRT_GROUP	16
#define NUMBER_OF_VIRT_GROUP	4

__kernel void GatherMotionKernel(__global unsigned char *ref,
								 int refIdx,
								 int image_width,
								 int image_height,
								 __global unsigned char *image,
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
	
#define assistance_points 4
	
	unsigned int x, y, i;
	
	/* get position of the current macroblock and the current virt. assistance point */
	int mb_x = get_group_id(0);
	int mb_y = get_group_id(1);
	
	int local_size = get_local_size(0);
	
	int ass_number = get_local_id(0) / NUMBER_OF_NEIGHBOURS;
	int local_id = get_local_id(0) % NUMBER_OF_NEIGHBOURS;
	int local_id_load = get_local_id(0);
	
	//int global_id = mad24(get_global_id(1),get_global_size(0),get_global_id(0));

	
	// offset in the current image to gather macroblock from
	int offset = mb_y*mb_height*image_width+mb_x*mb_width;
	
	// init local buffer
	if(local_id < assistance_points) curCost[local_id] = COST_MAX;
	
	/* load current_mb local cache */
	__global unsigned char *ptr;
	
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
	
	__local short step_size = STEP_START_SIZE;
	
	__local short nei_x[8] = {-1, 0, 1,-1,1,-1,0,1};
	__local short nei_y[8] = {-1,-1,-1, 0,0, 1,1,1};
	
	int ax	= temp_width/GROUP_WRAP*(ass_number%GROUP_WRAP);// + delta_region_width>>1; 
	int ay	= temp_height/GROUP_WRAP*(ass_number/GROUP_WRAP);// + delta_region_height>>1;
	
	// calc SAD
	offset = mad24(ay,temp_width,ax);
	local_result[ass_number].mv_predicted.x = ax;
	local_result[ass_number].mv_predicted.y = ay;
	local_result[ass_number].refIdx = refIdx;
	local_result[ass_number].mv_diff.x = 0;
	local_result[ass_number].mv_diff.y = 0;
	
	int mvd_x_n, mvd_y_n;
	
	if(local_id == 0) {
		
		mvd_x_n = 0;
		mvd_y_n = 0;
		for(y=0;y<mb_height;y++) {
			for(x=0;x<mb_width;x+=4) {
				mvd_y_n += abs_diff(current_mb[mvd_x_n],region_buffer[offset+x]);
				mvd_y_n += abs_diff(current_mb[mvd_x_n+1],region_buffer[offset+x+1]);
				mvd_y_n += abs_diff(current_mb[mvd_x_n+2],region_buffer[offset+x+2]);
				mvd_y_n += abs_diff(current_mb[mvd_x_n+3],region_buffer[offset+x+4]);
				mvd_x_n+=4;
			}
			offset+=temp_width;
		}
		mvd_y_n += createRefCost(refIdx);
		mvd_y_n += createMVDCost(25,0,0);
		local_result[ass_number].costs = mvd_y_n;
		curCost[ass_number] = mvd_y_n;
	}
	
	temp_x[ass_number] = ax;
	temp_y[ass_number] = ay;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	/* do computation as long as there was no change in the curCost value or min step size has been reached */
	do {
		
		tempCost[ass_number] = curCost[ass_number];
		
		i = ((ass_number << 3) + local_id)*3;
		
#define X_INDEX 0
#define Y_INDEX 1
#define COST_INDEX 2
		
		neighbour[i+X_INDEX] = mad24(step_size,nei_x[local_id],temp_x[ass_number]);
		neighbour[i+X_INDEX] = select(neighbour[i+X_INDEX],-1,neighbour[i+X_INDEX] < 0);
		neighbour[i+X_INDEX] = select(neighbour[i+X_INDEX],-1,neighbour[i+X_INDEX] > temp_width);
		
		neighbour[i+Y_INDEX] = mad24(step_size,nei_y[local_id],temp_y[ass_number]);
		neighbour[i+Y_INDEX] = select(neighbour[i+Y_INDEX],-1,neighbour[i+Y_INDEX] < 0);
		neighbour[i+Y_INDEX] = select(neighbour[i+Y_INDEX],-1,neighbour[i+Y_INDEX] > temp_height);
		
		neighbour[i+COST_INDEX] = COST_MAX;
		
		if(neighbour[i+Y_INDEX] > -1 && neighbour[i+X_INDEX] > -1) {
			
			// calc SAD
			offset = mad24(neighbour[i+Y_INDEX],temp_width,neighbour[i+X_INDEX]);
			mvd_x_n = 0;
			mvd_y_n =0;
			for(y=0;y<mb_height;++y) {
				for(x=0;x<mb_width;x+=4) {
					mvd_y_n += abs_diff(current_mb[mvd_x_n+0],region_buffer[offset+x+0]);
					mvd_y_n += abs_diff(current_mb[mvd_x_n+1],region_buffer[offset+x+1]);
					mvd_y_n += abs_diff(current_mb[mvd_x_n+2],region_buffer[offset+x+2]);
					mvd_y_n += abs_diff(current_mb[mvd_x_n+3],region_buffer[offset+x+3]);
					mvd_x_n+=4;
				}
				offset += temp_width;
			}
			neighbour[i+COST_INDEX] = mvd_y_n;
			mvd_x_n =	neighbour[i+X_INDEX]-ax;
			mvd_y_n =	neighbour[i+Y_INDEX]-ay;
			neighbour[i+COST_INDEX] += createRefCost(refIdx);
			neighbour[i+COST_INDEX] += createMVDCost(25,mvd_x_n,mvd_y_n);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
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
			step_size >>= 1;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
	} while(step_size > 1 || tempCost[ass_number] != curCost[ass_number]);
	
	/* write result back */
	if(local_id == 0) {
		int img_mb_w = image_width / mb_width;
		int out_index = mad24(mb_y,img_mb_w,mb_x);
		out_index *= 4;
		result[out_index+ass_number] = local_result[ass_number];
	}
	
}

__kernel void GatherMotionKernelold(__global char *ref,
									int refIdx,
									int image_width,
									int image_height,
									__global char *image,
									__global char *debug_output,
									int mb_width,
									int mb_height,
									int search_region,
									__local workItemResult *local_result,
									__local char *current_mb,
									__local char *region_buffer,
									__global workItemResult *result) {
	int i;
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int group_id = mad24(get_group_id(1),get_num_groups(0),get_group_id(0));//get_group_id(0) + get_group_id(1)*get_num_groups(0);
	
	// 1 check positions each group is one macroblock (VERIFIED)
	int image_mb_height =	image_height / mb_height;
	int image_mb_width =	image_width / mb_width;
	
	int block_x_pos = get_group_id(0);
	int block_y_pos = get_group_id(1);
	
	int work_group_size = local_size;
	
	int mvp_x, mvd_x_n;
	int mvp_y, mvd_y_n;
	
	int cost, x, y;
	__global char *ptr;
	
	// 1.1 cache current mb block (VERIFIED!!!)
	int current_mb_start = block_x_pos*mb_width + block_y_pos*mb_height*image_width;
	
	int divTemp = mb_height / local_size;
	int start = divTemp * local_id;
	int end = select(start+divTemp,mb_height,local_id == local_size-1);
	
	ptr = image+current_mb_start;//image+mad24(start,image_width,current_mb_start);//+start*image_width;
	
	//i=start*mb_width;
	//for(y=start;y<end;y++) {
	i=local_id;
	for(y=0;y<mb_height;y++) {
		for(x=local_id;x<mb_width;x+=local_size) {
			current_mb[i] = *(ptr + x);
			i+=local_size;
		}
		ptr += mb_width;
	}
	
	//barrier(CLK_LOCAL_MEM_FENCE);
	//return;*/
	
	//}
	
	// 1.2
	// create temp result for current local id
	
	// 2 choose assitance point depending on local id
	int region_width = search_region*mb_width;
	region_width <<= 1;
	
	//char region_buffer[search_region*2+mb_width][search_region*2+mb_height];
	// current_mb_start = block_x_pos*mb_width + block_y_pos*mb_height*image_width
	/// LOAD SEARCH REGION INTO LOCAL MEMORY
	
	int curCost = COST_MAX;
	
	//((value - diff) < 0) ? value : diff
	
#define MAX(value,diff,max_value) \
((value + diff) > max_value) ? (max_value - value) : diff
	
	int startRegion = current_mb_start-(min(block_x_pos*mb_width,search_region));
	startRegion -= (min(block_y_pos*mb_height*image_width,search_region*image_width));
	
	int temp_width = MAX(block_x_pos*mb_width,search_region+mb_width,image_width);
	temp_width += min(block_x_pos*mb_width,search_region);
	int temp_height = MAX(block_y_pos*mb_height,search_region+mb_height,image_height);
	temp_height += min(block_y_pos*mb_height,search_region);
	
	int neighbour[NUMBER_OF_NEIGHBOURS][3];
	
	int temp_y = temp_height/local_size;
	start = temp_y * local_id;
	end = select(start+temp_y,temp_height,local_id == local_size-1);
	
	//ptr = ref+startRegion+start*image_width;
	//i=start*temp_width;
	ptr = ref+startRegion;
	
	i=local_id;
	for(y=0;y<temp_height;y++) {
		for(x=local_id;x<temp_width;x+=local_size) {
			region_buffer[i] = *(ptr + x);
			i+=local_size;
		}
		ptr += image_width;
	}
	
	/*for(y=start;y<end;y++) {
	 for(x=0;x<temp_width;x++) {
	 region_buffer[i] = *(ptr+x);
	 ++i;
	 }
	 ptr += image_width;
	 }*/
	//}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_id < 4) {
		
		// VERIFIED TILL HERE !!!!!!!!
		
		
		// think about this part one more... you gotta have create constraints
		// on the max local size... so that the best locations can be estimated
		
		// setup assistance points
		int region_wh = temp_width;
		int region_hh = temp_height;
		int delta_region_width = temp_width/GROUP_WRAP;
		int delta_region_height = temp_height/GROUP_WRAP;
		
		int ax	= delta_region_width*(local_id%GROUP_WRAP);// + delta_region_width>>1; 
		int ay	= delta_region_height*(local_id/GROUP_WRAP);// + delta_region_height>>1; 
		
		mvp_x = ax;
		mvp_y = ay;
		local_result[local_id].mv_predicted.x = mvp_x;
		local_result[local_id].mv_predicted.y = mvp_y;
		local_result[local_id].refIdx = refIdx;
		
		cost = 0;
		// calc SAD
		int start_region = ax + ay*temp_width;
		mvd_x_n = 0;
		for(y=0;y<mb_height;y++) {
			for(x=0;x<mb_width;x++) {
				cost += abs_diff(current_mb[mvd_x_n],region_buffer[start_region+x]);
				mvd_x_n++;
			}
			start_region+=temp_width;
		}
		
		cost += createRefCost(refIdx);
		cost += createMVDCost(25,0,0);
		
		/*	if(group_id == 0 && local_id == 0) result[0].costs = cost;
		 return;*/
		
		// calc mvd cost
		mvd_x_n = 0; mvd_y_n = 0;
		//cost += COST_MAX;
		local_result[local_id].costs = cost;
		local_result[local_id].mv_diff.x = 0;
		local_result[local_id].mv_diff.y = 0;
		curCost = cost;
		
		int stepSize = STEP_START_SIZE;
		
		__local char nei_x[8] = {-1, 0, 1,-1,1,-1,0,1};
		__local char nei_y[8] = {-1,-1,-1, 0,0, 1,1,1};
		__local char* lPtr;
		
		// start motion estimation
		// compiler works till here
		// reset neigbour
		for(i=0;i<NUMBER_OF_NEIGHBOURS;i++) {
			neighbour[i][0] = ax;
			neighbour[i][1] = ay;
		}
		
		int tempCost;
		
		int temp_x = ax;
		int temp_index=0;
		temp_y = ay;
		int cost_cache;
		
		do {
			tempCost = curCost;
			// check costs for all sourronding 4+5
			for(i=0;i<NUMBER_OF_NEIGHBOURS;i++) {
				neighbour[i][0] = mad24(stepSize,nei_x[i],temp_x);
				neighbour[i][0] = select(neighbour[i][0],-1,neighbour[i][0] > -1);
				neighbour[i][0] = select(neighbour[i][0],-1,neighbour[i][0] > region_wh);
				
				neighbour[i][1] = mad24(stepSize,nei_y[i],temp_y);
				neighbour[i][1] = select(neighbour[i][1],-1,neighbour[i][1] > -1);
				neighbour[i][1] = select(neighbour[i][1],-1,neighbour[i][1] > region_wh);
				
				if(neighbour[i][0] == -1 || neighbour[i][1] == -1) continue;
				
				// calc SAD
				start_region = mad24(neighbour[i][1],temp_width,neighbour[i][0]);//neighbour[i][0]+neighbour[i][1]*(temp_width);
				neighbour[i][3] = 0;
				lPtr = current_mb;
				for(y=0;y<mb_height;++y) {
					for(x=0;x<mb_width;x+=4) {
						neighbour[i][3] += abs_diff(*lPtr,region_buffer[start_region+x]);
						++lPtr;
						neighbour[i][3] += abs_diff(*lPtr,region_buffer[start_region+x+1]);
						++lPtr;
						neighbour[i][3] += abs_diff(*lPtr,region_buffer[start_region+x+2]);
						++lPtr;
						neighbour[i][3] += abs_diff(*lPtr,region_buffer[start_region+x+3]);
						++lPtr;
					}
					start_region += temp_width;
				}
				mvd_x_n =	neighbour[i][0]-ax;
				mvd_y_n =	neighbour[i][1]-ay;
				neighbour[i][3] += createRefCost(refIdx);
				neighbour[i][3] += createMVDCost(25,mvd_x_n,mvd_y_n);
				temp_index = select(temp_index,i,neighbour[i][3]<curCost);
				curCost = min(neighbour[i][3],curCost);
			}
			
			if(tempCost == curCost) break;
			
			mvd_x_n =	neighbour[temp_index][0]-ax;
			mvd_y_n =	neighbour[temp_index][1]-ay;
			local_result[local_id].costs = curCost;
			local_result[local_id].mv_diff.x = mvd_x_n;
			local_result[local_id].mv_diff.y = mvd_y_n;
			temp_x = neighbour[temp_index][0];
			temp_y = neighbour[temp_index][1];
			
			stepSize >>= 1;
			
		} while(curCost < tempCost && stepSize > 0);
		
		i = mad24(group_id,local_size,local_id);
		result[i] = local_result[local_id];
		//	result[group_id+local_id].refIdx = 27;
	}
}
