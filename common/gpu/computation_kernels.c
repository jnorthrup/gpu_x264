/*
* Kernel functions for x264 accerlation through OpenCL
*/

#define MAX_REFERENCE_SIZE 4
#define COST_MAX 655356
#define GROUP_WRAP 2
#define GROUP_MID 4
#define STEP_START_SIZE 4
#define NUMBER_OF_NEIGHBOURS 8
#define SEARCH_REGION 16

/*typedef struct workItemResult_t {
	int2 	mv_predicted;			// predicted motion vector
	int2 	mv_diff;			// best match diff vector
	unsigned int costs;			// best match cost
	int 	refIdx;				// best match reference index
} workItemResult __attribute__((aligned));*/

typedef struct workItemResult_t {
	int mv_predicted;			// predicted motion vector
	int mv_diff;			// best match diff vector
	unsigned int costs;			// best match cost
	int 	refIdx;				// best match reference index
} workItemResult __attribute__((aligned));


// get reference costs in exp golomb coding bits
int createRefCost(int ref) {
	if(ref == 0) return 0;
	ref = 16 - clz(ref);
	return ((ref<<1) - 1);
}
/*
create costs based on the mv difference       
*/
inline int createMVDCost(int i_lambda, int x, int y) {
	if(x == 0 && y == 0) return 0;
	x = abs(x);
	y = abs(y);
	// created costs by mvd
	
	// copy of the source inside x.264 to keep compatibility
	x = i_lambda * x;//(native_log2((float)x+1)*2 + !!(x));
	x += i_lambda * y;//(native_log2((float)y+1)*2 + !!(y));
	return x;
}

#define A 0
#define B 1
#define C 2

int2 median(int2 mvp_a, int2 mvp_b, int2 mvp_c) {
	int2 diff;
	diff.x = mvp_a.x + mvp_b.x + mvp_c.x;
	diff.x -= min(mvp_a.x,min(mvp_b.x,mvp_c.x));
	diff.x -= max(mvp_a.x,max(mvp_b.x,mvp_c.x));
	diff.y = mvp_a.y + mvp_b.y + mvp_c.y;
	diff.y -= min(mvp_a.y,min(mvp_b.y,mvp_c.y));
	diff.y -= max(mvp_a.y,max(mvp_b.y,mvp_c.y));
	return diff;
}

inline int2 median2(int2 mvp_a, int2 mvp_b) {
	int2 diff;
	diff = mvp_a + mvp_b;
	diff.x >>= 1;
	diff.y >>= 1;
	return diff;
}

void get_cost(unsigned char *ref, int *c) {
*c += ref[0];
*c += ref[1];
*c += ref[2];
*c += ref[3];
}

void get_cost_char4(uchar4 *ref, int *c) {
*c += (*ref).x;			
*c += (*ref).y;
*c += (*ref).z;
*c += (*ref).w;
}

void get_cost2(unsigned char *ref, unsigned char *ref2, int *c) {
*c += abs_diff(ref[0],ref2[0]);
*c += abs_diff(ref[1],ref2[1]);
*c += abs_diff(ref[2],ref2[2]);
*c += abs_diff(ref[3],ref2[3]);
}

#define INIT_VALUE ((64 << 16) | 64)
#define ADD_VALUE 64

#define IS_POSITIVE(x) {\
	if(x > 0) {\
		x = 1;\
	} else {\
		x = 0;\
	}};

#define COPY_STRUCT(dst,src) \
	dst.costs = src.costs;\
	dst.mv_diff = src.mv_diff;\
	dst.mv_predicted = src.mv_predicted;\
	dst.refIdx = src.refIdx;

#define COPY_RESET(dst) \
	dst.costs = 0;\
	dst.mv_diff = INIT_VALUE;\
	dst.mv_predicted = INIT_VALUE;\
	dst.refIdx = -1;

inline int MV2Plain(int2 mv) {
	mv.x += ADD_VALUE;
	mv.y += ADD_VALUE;
	mv.x &= 0xFFFF;
	mv.y &= 0xFFFF;
	return ((mv.x<<16) | mv.y);
}

inline int createPlain(int x, int y) {
	x += ADD_VALUE;
	y += ADD_VALUE;
	x &= 0xFFFF;
	y &= 0xFFFF;
	return ((x << 16) | (y));
}

inline int2 Plain2MV(int mv) {
	int2 vector;
	vector.x = mv >> 16;
	vector.x -= ADD_VALUE;
	vector.y = mv & 0xFFFF;
	vector.y -= ADD_VALUE;
	return vector;
}

// selection kernel
/*
@param	data			data to process must be read and write able
@param	iteration		current iteration process
				iteration goes from 0 ... (image_mb_height - 1)*2 + image_mb_width
@param	image_mb_width		image width in macroblocks
@param image_mb_height		image height in macroblocks
@param assitance_number	number assistance points per sample
@param result			structure to write to
*/
__kernel void SelectionBestMatchKernel( __global workItemResult *data,
					const int iteration, const int image_mb_widht, const int image_mb_height,
					const int assistance_number,
					__global char *debug,
					volatile __global workItemResult *result) {
					
    	__local int2 neighbour[360];

	unsigned int i;

	int row = (get_group_id(1)<<1) + iteration;
	int col = get_group_id(0)*get_local_size(0)+get_local_id(0);
	int stride = get_global_size(0);
	
	int global_id = row * get_global_size(0) + col;

	int local_size = get_local_size(0);
	int local_idx = get_local_id(0);
	int max_load = local_size * assistance_number;

	workItemResult element, temp;
	workItemResult zeroElement;
	COPY_RESET(zeroElement);

	/*int global_id_load = (row * get_global_size(0))*assistance_number;

	// buffer all items
	for(i=local_idx;i<max_load;i+=local_size) {
		temp = data[global_id_load+i];
		COPY_STRUCT(buffer[i],temp);
	}

	barrier(CLK_LOCAL_MEM_FENCE);*/

	int value = 1<<29;
	int minIdx;
	int offset;/* = local_idx * assistance_number;
	for(i=0;i<assistance_number;++i) {
		if(data[global_id_load+i].costs < value) {
			value = data[global_id_load+i].costs;
			minIdx = i;
		}
	}*/

	/*COPY_STRUCT(element,data[global_id_load+minIdx]);
	element.mv_predicted += element.mv_diff;
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	COPY_STRUCT(result[global_id],element);*/

	offset = local_idx * 3;

	int validA, validB, validC;

	if(col == 0) {
		neighbour[offset+A] = Plain2MV(zeroElement.mv_predicted);
		validA = 0;
	} else {
		neighbour[offset+A] = Plain2MV(result[global_id-1].mv_predicted) + Plain2MV(result[global_id-1].mv_diff);
		validA = 1;
	}

	if(row == 0) {
		neighbour[offset+B] = Plain2MV(zeroElement.mv_predicted);
		validB = 0;
	} else {
		neighbour[offset+B] = Plain2MV(result[global_id-stride].mv_predicted) + Plain2MV(result[global_id-stride].mv_diff);
		validB = 1;
	}

	if(col == stride-1) {
		// @BUG fixed choosing partition D instaed of C if only c is not available
		neighbour[offset+C] = Plain2MV(result[global_id-1-stride].mv_predicted) + Plain2MV(result[global_id-1-stride].mv_diff);
		validC = 0;
	} else if(row == 0) {
	   neighbour[offset+C] = Plain2MV(zeroElement.mv_predicted);
		validC = 0;
	}else {
		neighbour[offset+C] = Plain2MV(result[global_id+1-stride].mv_predicted) + Plain2MV(result[global_id+1-stride].mv_diff);
		validC = 1;
	}

	if(validC == 0 && validB == 0) {
		neighbour[offset+B]=neighbour[offset+A];
		neighbour[offset+C]=neighbour[offset+A];
	}

	int2 mv_p;

	if((validA && !validB && !validC) == 0) {
		mv_p = median(neighbour[A+offset],
			      neighbour[B+offset],
			      neighbour[C+offset]);
	} else {
		mv_p = neighbour[A+offset];
	}

	COPY_STRUCT(element,result[global_id]);	
	int2 diff = Plain2MV(element.mv_predicted)-mv_p;
	element.mv_predicted = MV2Plain(mv_p);
	element.mv_diff = MV2Plain(diff);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	COPY_STRUCT(result[global_id],element);
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

#define TAG_MASK 0xFFFFFC00;
#define TAG_SHIFT 10

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

int inRange(unsigned int origin, unsigned int other) {
	int2 coordOther, coordValue;
	coordOther.x = (other & 0x0F) << 1;
	coordOther.y = (other & 0xF0) >> 3;
	coordValue.x = (origin & 0x0F) << 1;
	coordValue.y = (origin & 0xF0) >> 3;

	int2 coordDiff = coordOther-coordValue;
	origin >>= 8;
	other >>= 8;
	int cost = createMVDCost(16,coordDiff.x,coordDiff.y);
	if(cost+other < origin) return 0;
	return 1;
}

__kernel void selectBestMatch(__global int *cost_result,
			      __global workItemResult *result) {

	// EXECUTED WITH IMG_MB_WIDTH/2 * IMG_MB_HEIGHT * 256
	__local unsigned int cost[256];
	int id = get_local_id(0);

	int mb_x = get_group_id(0);
	int mb_x_max = get_num_groups(0);
	int mb_y = get_group_id(1);
	int mb_y_max = get_num_groups(1);

	int i;
	int count;

	int id_input = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	id_input <<= 8;
	/*id_input = count  << 8;	//256
	id_input += count << 5;	//32
	id_input += count; // 1 -> x*(256+ 32+1)  = x*289*/

	int local_size = get_local_size(0);
	int step = 1;
	int step_inc = 1;
	int id_step;
	unsigned cost_src, cost_dst;
	int step_index[2];
	
	cost[id] = (cost_result[id_input + id] << 8) | id;
	barrier(CLK_LOCAL_MEM_FENCE);

	id_input = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	//id_input <<= 1;

	unsigned int cost_zeromv;
	if(mb_x == 0 && mb_y == 0) {
		cost_zeromv = cost[0];
	} else if(mb_x == mb_x_max-1 && mb_y == 0) {
		cost_zeromv = cost[15];
	} else if(mb_x == 0 && mb_y == mb_y_max-1) {
		cost_zeromv = cost[240];
	} else if(mb_x == mb_x_max-1 && mb_y == mb_y_max-1) {
		cost_zeromv = cost[255];
	} else if(mb_x == 0) {
		cost_zeromv = cost[128];
	} else if(mb_y == 0) {
		cost_zeromv = cost[8];
	} else if(mb_x == mb_x_max-1) {
		cost_zeromv = cost[143];
	} else if(mb_y == mb_y_max-1) {
		cost_zeromv = cost[248];
	} else cost_zeromv = cost[136];

	unsigned int decide[2] = {cost[id],cost_zeromv};
	cost[id] = decide[inRange(cost_zeromv,cost[id])];

	int limit = 128;
	int inc=0;
	int index1, index2;
	while(limit > 8 && id < limit) {
		if(id < limit) {
			index1 = id & (limit-1);
			index2 = index1 + limit;
			cost_src = cost[index1];
			cost_dst = cost[index2];
			step_index[0] = cost_src & 0xFF;
			step_index[1] = cost_dst & 0xFF;
			cost_src &= TAG_MASK;
			cost_dst &= TAG_MASK;
		
			cost[index1] = min(cost_src,cost_dst);
			cost[index1] |= step_index[(cost[index1] == cost_dst)];
			limit >>= 1;
		}
	}

	/*do {
		local_size >>= 1;
		id_step = id<<step_inc;

		if(id < local_size && id_step < 256 && id_step+step < 256) {
			cost_src = cost[id_step];
			cost_dst = cost[id_step+step];
			step_index[0] = cost_src & 0xFF;
			step_index[1] = cost_dst & 0xFF;
			cost_src &= TAG_MASK;
			cost_dst &= TAG_MASK;
			
			cost[id_step] = min(cost_src,cost_dst);
			cost[id_step] |= step_index[(cost[id_step] == cost_dst)];
		}
		step <<= 1;
		++step_inc;
		barrier(CLK_LOCAL_MEM_FENCE);
	} while(local_size > 0);*/

	if(id == 0) {
		cost_src = cost[0];
		step_index[0] = cost_src & 0xFF;
		cost_src &= TAG_MASK;
		for(inc=1;inc<16;inc++) {
			cost_dst = cost[inc];
			step_index[1] = cost_dst & 0xFF;
			cost_dst &= TAG_MASK;
			
			cost[0] = min(cost_src,cost_dst);
			cost[0] |= step_index[(cost[0] == cost_dst)];
		}

		int x = cost[0] & 0x0F;
		int y = cost[0] & 0xF0;
	
		y >>= 4;

		x += (mb_x == mb_x_max-1);
		y += (mb_y == mb_y_max-1);

		x *= 2;
		y *= 2;

		if(get_group_id(0) > 0) x -= 16;
		if(get_group_id(0) == get_num_groups(0)-1) x -= 16;

		if(get_group_id(1) > 0) y -= 16;
		if(get_group_id(1) == get_num_groups(1)-1) y -= 16;
	
		result[id_input].costs = (cost[0] >> 8);
		result[id_input].mv_predicted = createPlain(x,y);
		result[id_input].mv_diff = INIT_VALUE;
		result[id_input].refIdx = 0;
	}
}

__constant int Anorm[16] = {0,0,0,0, 1,0,0,1, 0,0,0,0, 0,1,1,0};
__constant int Atran[16] = {0,1,0,0, 0,0,0,1, 0,0,0,1, 0,1,0,0};
__constant int negative[16] = {0,0,0,0, 0,0,1,1, 0,1,1,0, 0,1,0,1};
__constant int dct_mask[16] = {0,16,32,48, 1,17,33,49, 2,18,34,50, 3,19,35,51};
__constant int x_dct[16] = {0,4,0,4, 8,12,8,12, 0,4,0,4, 8,12,8,12};
__constant int y_dct[16] = {0,0,4,4, 0,0,4,4, 8,8,12,12, 8,8,12,12};
__constant int MF_TABLE[18] = {13107,5243,8066,
			       11916,4660,7490,
			       10082,4194,6554,
			        9362,3647,5825,
				8192,3355,5243,
				7282,2893,4559};
__constant int MF_INDEX[16] = {0,2,0,2, 2,1,2,1, 0,2,0,2, 2,1,2,1};
__constant int f_TABLE[16] = {1,2,1,2, 2,3,2,3, 1,2,1,2, 2,3,2,3};

typedef union {
	int value;
	short svalue[2];
} intShortUnion;

#define SHIFT(x,s) ((s)<0 ? ((x)<<(1)) : ((s)==0 ? (x) : ((x)+(1<<((s)-1)))>>(s)))
#define DIV(n,d) (((n) + ((d)>>1)) / (d))

__kernel void
calcDCT16x16(__read_only image2d_t source,
	     __read_only image2d_t reference,
	     __global workItemResult *data,
	     __global short *residual_data,
	     __global int *non_zero,
	     const int start_y,
	     const int qp) {
	// @ EXECUTION  NOTE: 256 per macroblock for 16x16
	
	__local int residual[256];
	__local int table[16];
	__local int table_f[16];
	__local int2 mv_diff = 0;
	
	int qbits = 15 + (qp)/6;
	int f_value = (1<<qbits);
	f_value /= 6;

	uint4 pixel;
	int2 coord;

	int mb_x = get_group_id(0);
	int mb_y = start_y+get_group_id(1);
	int id = get_local_id(0);
	int j;

	if(id < 16) {
		j = SHIFT(MF_TABLE[((qp % 6)*3)+MF_INDEX[id]],(qp/6)-1);
		table[id] = j;

		table_f[id] = min((1<<15)/j,DIV(11<<10,j));
	}

	int mb_idx = mb_x + mb_y * get_num_groups(0);

	if(id == 0) mv_diff = Plain2MV(data[mb_idx].mv_predicted) + Plain2MV(data[mb_idx].mv_diff);

	int x_off = id & 15;
	int y_off = id >> 4;

	coord.x = (mb_x << 4) + x_off;
	coord.y = (mb_y << 4) + y_off;

	pixel = read_imageui(source,sampler,coord);
	residual[id] = -pixel.x;

	barrier(CLK_LOCAL_MEM_FENCE);
	coord.x += mv_diff.x;
	coord.y += mv_diff.y;

	pixel = read_imageui(reference,sampler,coord);
	residual[id] += pixel.x;

	barrier(CLK_LOCAL_MEM_FENCE);	

	int row = ((id >> 4) & 3) << 2;
	
	y_off = (id & 0xC0) + (x_off);

	int value;
	int choose[2];

	choose[0] = residual[y_off] << Anorm[row];
	choose[1] = ~(residual[y_off] << Anorm[row])+1;
	value = choose[negative[row]];

	choose[0] = residual[y_off+16] << Anorm[row+1];
	choose[1] = ~(residual[y_off+16] << Anorm[row+1])+1;
	value += choose[negative[row+1]];

	choose[0] = residual[y_off+32] << Anorm[row+2];
	choose[1] = ~(residual[y_off+32] << Anorm[row+2])+1;
	value += choose[negative[row+2]];

	choose[0] = residual[y_off+48] << Anorm[row+3];
	choose[1] = ~(residual[y_off+48] << Anorm[row+3])+1;
	value += choose[negative[row+3]];

	barrier(CLK_LOCAL_MEM_FENCE);
	residual[id] = value;
	barrier(CLK_LOCAL_MEM_FENCE);

	y_off = (id & 0xF0);
	y_off += (x_off & 0xC);

	row = (id & 3);

	choose[0] = residual[y_off] << Atran[row];
	choose[1] = ~(residual[y_off] << Atran[row])+1;
	value = choose[negative[row]];

	choose[0] = residual[y_off+1] << Atran[row+4];
	choose[1] = ~(residual[y_off+1] << Atran[row+4])+1;
	value += choose[negative[row+4]];

	choose[0] = residual[y_off+2] << Atran[row+8];
	choose[1] = ~(residual[y_off+2] << Atran[row+8])+1;
	value += choose[negative[row+8]];

	choose[0] = residual[y_off+3] << Atran[row+12];
	choose[1] = ~(residual[y_off+3] << Atran[row+12])+1;
	value += choose[negative[row+12]];

	barrier(CLK_LOCAL_MEM_FENCE);
	residual[id] = -value;
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset = (mb_idx << 8);

	// do quantiziation for
	int sign = (residual[id] < 0) ? -1 : 1;
	x_off = id & 0x3;
	y_off = (id >> 4) & 0x3;

	/*value = x_off + (y_off << 2);
	residual[id] = residual[id]*(sign);
	residual[id] = (table[value]*residual[id]+f_value);
	residual[id] >>= qbits;
	residual[id] = residual[id]*(sign);*/

	value = x_off + (y_off << 2);
	residual[id] = residual[id]*(sign);
	residual[id] = (table[value]*(residual[id]+table_f[value]));
	residual[id] >>= 16;
	residual[id] = residual[id]*(sign);


	value = dct_mask[id & 15];
	value += x_dct[id >> 4];
	value += y_dct[id >> 4] << 4;	
	barrier(CLK_LOCAL_MEM_FENCE);
	value = residual[value];
	barrier(CLK_LOCAL_MEM_FENCE);
	residual[id] = value;

	//residual_data[offset+id] = (short)value;

	if(id < 16) {
		int i=0;
		int index=id;
		int zero=0;
		#pragma unroll
		for(i=0;i<16;i++) {
			index = (id+i) & 15;
			zero += (residual[(id<<4)+index] != 0);
		}
		i = mb_idx << 4;
		non_zero[i + id] = (zero != 0);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(id < 128) {
		intShortUnion element;
		int id2 = id << 1;
		element.svalue[0] = (short)residual[id2];
		element.svalue[1] = (short)residual[id2+1];

		__global int *ptr = (__global int *)&residual_data[offset];
		*(ptr+id) = element.value;
	}
}

__kernel void
GatherMotionKernel(	const int refIdx,
			const int image_width,
			const int image_height,
			int mb_width,
			int mb_height,
			int search_region,
			__global int *cost_result,
			__read_only image2d_t source,
			__read_only image2d_t reference) {
	
	// neighbor [assistance_points * 3 * NUMBER_OF_NEIGHBORS]

	// must be executed with IMG_MB_WIDTH * IMG_MB_WIDTH * 256 THREADS

	__local unsigned int current_mb[256];
	
	int x, y, y2;
	
	/* get position of the current macroblock and the current virt. assistance point */
	// get chess pattern
	int mb_y = get_group_id(1);
	int mb_x = get_group_id(0);
	
	int curMb = get_group_id(0) + get_group_id(1)*get_num_groups(0);

	int mb_y_max = get_num_groups(1);
	int mb_x_max = get_num_groups(0);
	
	int local_size = get_local_size(0);
	int local_id = get_local_id(0);
	
	#define IMG_WIDTH (image_width+64)
	// offset in the current image to gather macroblock from
	int offset = mb_height*mb_y*IMG_WIDTH + mb_x*mb_width;
	
	// init local buffer
	int2 coord;
	coord.x = mb_x*mb_width + (local_id & 15);
	coord.y = mb_y*mb_height + (local_id / 16);

	uint4 pixel = read_imageui(source,sampler,coord);

	current_mb[local_id] = pixel.x;
	barrier(CLK_LOCAL_MEM_FENCE);
	
#define MAX(value,diff,max_value) \
((value + diff) > max_value) ? (max_value - value) : diff
	
	offset = offset-(min(mb_x*mb_width,search_region));
	offset -= (min(mb_y*mb_height*IMG_WIDTH,search_region*IMG_WIDTH));

	// MODIFIED SEARCH REGION FOR BETTER MATCHING ADAPTION LATER
	
	offset -= (mb_x == mb_x_max) * (mb_width);
	offset -= (mb_y == mb_y_max) * (mb_height*IMG_WIDTH);

#define temp_width 48
#define temp_height 48

	int move_x = (mb_x == mb_x_max-1)<<1;
	int move_y = (mb_y == mb_y_max-1)<<1;
		
	int x_off = (offset % IMG_WIDTH) + ((local_id & 15) << 1) + move_x;
	int y_off = (offset / IMG_WIDTH) + ((local_id / 16) << 1) + move_y;

	int cost = 0;
	y2 = 0;

	#pragma unroll
	coord.y = y_off;
	for(y=0;y<16;++y,++coord.y) {
		coord.x = x_off;
		#pragma unroll
		for(x=0;x<16;++x) {
			pixel = read_imageui(reference,sampler,coord);
			cost += abs_diff(current_mb[y2],pixel.x);
			++y2;
			coord.x++;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	local_id = (curMb<<8) + get_local_id(0);
	cost_result[local_id] = cost;
}



	