//
//  flow.metal
//  MIRFlow
//
//  Created by Mirror on 2022/4/16.
//

#include <metal_stdlib>

#import "MIRFlowShaderTypes.h"

using namespace metal;

#define EPS 0.001f
#define INF 1E+10F

constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);

kernel void MIRFlow_resizeUChar(texture2d<half, access::sample> texture,
                                device uchar *dst [[ buffer(0) ]],
                                constant int &width [[ buffer(1) ]],
                                constant int &height [[ buffer(2) ]],
                                uint2 gid [[ thread_position_in_grid ]]
                                ) {
    if (gid.x >= (uint)width || gid.y >= (uint)height) {
        return;
    }
    
    half a = texture.sample(textureSampler, float2(gid) / float2(width, height)).a;
    dst[gid.y * width + gid.x] = (uchar)(a * 255);
}

// BORDER_REPLICATE
kernel void MIRFlow_copyMakeBorder(texture2d<half, access::sample> texture,
                                   device uchar *dst [[ buffer(0) ]],
                                   constant int &width [[ buffer(1) ]],
                                   constant int &height [[ buffer(2) ]],
                                   constant int &border [[ buffer(3) ]],
                                   uint2 gid [[ thread_position_in_grid ]]
                                   ) {
    if (gid.x >= ((uint)width + 2*border) || gid.y >= ((uint)height + 2*border)) {
        return;
    }
    float2 pos = float2(gid - uint2(border, border)) / float2(width, height);
    pos = clamp(pos, float2(0.0), float2(1.0));
    half a = texture.sample(textureSampler, pos).a;
    dst[gid.y * (width + 2 * border) + gid.x] = (uchar)(a * 255);
}

kernel void MIRFlow_grayscale(texture2d<half, access::sample> input [[ texture(0) ]],
                              texture2d<half, access::write> output [[ texture(1) ]],
                              uint2 gid [[ thread_position_in_grid ]]
                              ) {
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    float2 pos = float2(gid) / float2(width, height);
    half4 color = input.sample(textureSampler, pos);
    half grayscale = dot(color.rgb, half3(0.299, 0.587, 0.114));
    output.write(half4(grayscale), gid);
}

kernel void MIRFlow_SpatialGradientKernel(constant uchar *src [[ buffer(MIRSpatialGradient_src) ]],
                                          device short *vx [[ buffer(MIRSpatialGradient_vx) ]],
                                          device short *vy [[ buffer(MIRSpatialGradient_vy) ]],
                                          constant MIRSpatialGradientOpt &opt [[ buffer(MIRSpatialGradient_opt) ]],
                                          uint2 gid [[ thread_position_in_grid ]]
                                          ) {
    int w = opt.w;
    int h = opt.h;
    if (gid.x >= (uint)w || gid.y >= (uint)h) {
        return;
    }
    int p_i, c_i, n_i, p_j, c_j, n_j;
    c_i = gid.x, c_j = gid.y;
    p_i = abs(c_i - 1), p_j = abs(c_j - 1);
    n_i = (w - 1) - abs((w - 1) - (c_i + 1)), n_j = (h - 1) - abs((h - 1) - (c_j + 1));
    
#define at(y, x) (src[(y) * opt.w + (x)])
    short v00 = at(p_j, p_i), v01 = at(p_j, c_i), v02 = at(p_j, n_i),
    v10 = at(c_j, p_i), v12 = at(c_j, n_i),
    v20 = at(n_j, p_i), v21 = at(n_j, c_i), v22 = at(n_j, n_i);
#undef at
    short tmp_add = v22 - v00, tmp_sub = v02 - v20, tmp_x = v12 - v10, tmp_y = v21 - v01;
    
    size_t offset = gid.y * w + gid.x;
    vx[offset] = tmp_add + tmp_sub + tmp_x + tmp_y;
    vy[offset] = tmp_add - tmp_sub + tmp_x + tmp_y;
}


kernel void MIRFlow_PrecomputeStructureTensor_hor(constant short *I0x [[ buffer(MIRPrecomputeStructureTensor_I0x) ]],
                                                  constant short *I0y [[ buffer(MIRPrecomputeStructureTensor_I0y) ]],
                                                  constant MIRPrecomputeStructureTensorOpt &opt [[ buffer(MIRPrecomputeStructureTensor_opt) ]],
                                                  device float *I0xx_aux [[ buffer(MIRPrecomputeStructureTensor_I0xx_aux) ]],
                                                  device float *I0yy_aux [[ buffer(MIRPrecomputeStructureTensor_I0yy_aux) ]],
                                                  device float *I0xy_aux [[ buffer(MIRPrecomputeStructureTensor_I0xy_aux) ]],
                                                  device float *I0x_aux [[ buffer(MIRPrecomputeStructureTensor_I0x_aux) ]],
                                                  device float *I0y_aux [[ buffer(MIRPrecomputeStructureTensor_I0y_aux) ]],
                                                  uint gid [[ thread_position_in_grid ]]
                                                  ) {
    if (gid >= (uint)opt.h) {
        return;
    }
    
    float sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0;
    int j = 0;
    int js = 0;
#define atI(ptr, row, col) *(ptr + (row) * opt.w + (col))
#define atAux(ptr, row, col) *(ptr + (row) * opt.ws + (col))
    for (; j < opt.patch_size; j++) {
        short x = atI(I0x, gid, j);
        short y = atI(I0y, gid, j);
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
        sum_x += x;
        sum_y += x;
    }
    atAux(I0xx_aux, gid, js) = sum_xx;
    atAux(I0yy_aux, gid, js) = sum_yy;
    atAux(I0xy_aux, gid, js) = sum_xy;
    atAux(I0x_aux, gid, js) = sum_x;
    atAux(I0y_aux, gid, js) = sum_y;
    js++;
    
    for (; j < opt.w; j++) {
        short xp = atI(I0x, gid, j - opt.patch_size);
        short yp = atI(I0y, gid, j - opt.patch_size);
        short x = atI(I0x, gid, j);
        short y = atI(I0y, gid, j);
        sum_xx += x * x - xp * xp;
        sum_yy += y * y - yp * yp;
        sum_xy += x * y - xp * yp;
        sum_x += x - xp;
        sum_y += x - xp;
        if ((j - opt.patch_size + 1) % opt.patch_stride == 0) {
            atAux(I0xx_aux, gid, js) = sum_xx;
            atAux(I0yy_aux, gid, js) = sum_yy;
            atAux(I0xy_aux, gid, js) = sum_xy;
            atAux(I0x_aux, gid, js) = sum_x;
            atAux(I0y_aux, gid, js) = sum_y;
            js++;
        }
    }
#undef atI
#undef atAux
}

kernel void MIRFlow_PrecomputeStructureTensor_ver(constant float *I0xx_aux [[ buffer(MIRPrecomputeStructureTensor_I0xx_aux) ]],
                                                  constant float *I0yy_aux [[ buffer(MIRPrecomputeStructureTensor_I0yy_aux) ]],
                                                  constant float *I0xy_aux [[ buffer(MIRPrecomputeStructureTensor_I0xy_aux) ]],
                                                  constant float *I0x_aux [[ buffer(MIRPrecomputeStructureTensor_I0x_aux) ]],
                                                  constant float *I0y_aux [[ buffer(MIRPrecomputeStructureTensor_I0y_aux) ]],
                                                  constant MIRPrecomputeStructureTensorOpt &opt [[ buffer(MIRPrecomputeStructureTensor_opt) ]],
                                                  device float *I0xx_buf [[ buffer(MIRPrecomputeStructureTensor_I0xx_buf) ]],
                                                  device float *I0yy_buf [[ buffer(MIRPrecomputeStructureTensor_I0yy_buf) ]],
                                                  device float *I0xy_buf [[ buffer(MIRPrecomputeStructureTensor_I0xy_buf) ]],
                                                  device float *I0x_buf [[ buffer(MIRPrecomputeStructureTensor_I0x_buf) ]],
                                                  device float *I0y_buf [[ buffer(MIRPrecomputeStructureTensor_I0y_buf) ]],
                                                  uint gid [[ thread_position_in_grid ]]
                                                  ) {
    if (gid >= (uint)opt.ws) {
        return;
    }
    
    float sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0;
    int i = 0;
    int is = 0;
#define at(ptr, row, col) *(ptr + (row) * opt.ws + (col))
    for (; i < opt.patch_size; i++) {
        sum_xx += at(I0xx_aux, i, gid);
        sum_yy += at(I0yy_aux, i, gid);
        sum_xy += at(I0xy_aux, i, gid);
        sum_x += at(I0x_aux, i, gid);
        sum_y += at(I0y_aux, i, gid);
    }
    at(I0xx_buf, is, gid) = sum_xx;
    at(I0yy_buf, is, gid) = sum_yy;
    at(I0xy_buf, is, gid) = sum_xy;
    at(I0x_buf, is, gid) = sum_x;
    at(I0y_buf, is, gid) = sum_y;
    is++;
    
    for (; i < opt.h; i++) {
        sum_xx += at(I0xx_aux, i, gid) - at(I0xx_aux, i - opt.patch_size, gid);
        sum_yy += at(I0yy_aux, i, gid) - at(I0yy_aux, i - opt.patch_size, gid);
        sum_xy += at(I0xy_aux, i, gid) - at(I0xy_aux, i - opt.patch_size, gid);
        sum_x += at(I0x_aux, i, gid) - at(I0x_aux, i - opt.patch_size, gid);
        sum_y += at(I0y_aux, i, gid) - at(I0y_aux, i - opt.patch_size, gid);
        if ((i - opt.patch_size + 1) % opt.patch_stride == 0) {
            at(I0xx_buf, is, gid) = sum_xx;
            at(I0yy_buf, is, gid) = sum_yy;
            at(I0xy_buf, is, gid) = sum_xy;
            at(I0x_buf, is, gid) = sum_x;
            at(I0y_buf, is, gid) = sum_y;
            is++;;
        }
    }
#undef at
}

#define INIT_BILINEAR_WEIGHTS(Ux, Uy) \
i_I1 = clamp(i + Uy + border_size, i_lower_limit, i_upper_limit); \
j_I1 = clamp(j + Ux + border_size, j_lower_limit, j_upper_limit); \
{ \
float di = i_I1 - floor(i_I1); \
float dj = j_I1 - floor(j_I1); \
w11 = di       * dj; \
w10 = di       * (1 - dj); \
w01 = (1 - di) * dj; \
w00 = (1 - di) * (1 - dj); \
}

float computeSSDMeanNorm(constant uchar *I0_ptr,
                         constant uchar *I1_ptr,
                         int I0_stride,
                         int I1_stride,
                         int patch_size,
                         float w00, float w01, float w10, float w11,
                         int i,
                         threadgroup vector_float2 *smem /*[8]*/
                         )
{
    int n = patch_size * patch_size;
    
    float4 vec1, vec2;
    {
        uchar4 I0_vec = *(I0_ptr + i * I0_stride);
        uchar4 I1_vec1 = *(I1_ptr + i * I1_stride);
        uchar4 I1_vec1_p = *(I1_ptr + i * I1_stride + 1);
        uchar4 I1_vec2 = *(I1_ptr + (i + 1) * I1_stride);
        uchar4 I1_vec2_p = *(I1_ptr + (i + 1) * I1_stride + 1);
        
        vec1 = w00 * float4(I1_vec1) + w01 * float4(I1_vec1_p)
        + w10 * float4(I1_vec2) + w11 * float4(I1_vec2_p)
        - float4(I0_vec);
    }
    {
        uchar4 I0_vec = *(I0_ptr + i * I0_stride + 4);
        uchar4 I1_vec1 = *(I1_ptr + i * I1_stride + 4);
        uchar4 I1_vec1_p = *(I1_ptr + i * I1_stride + 5);
        uchar4 I1_vec2 = *(I1_ptr + (i + 1) * I1_stride  + 4);
        uchar4 I1_vec2_p = *(I1_ptr + (i + 1) * I1_stride + 5);
        
        vec2 = w00 * float4(I1_vec1) + w01 * float4(I1_vec1_p)
        + w10 * float4(I1_vec2) + w11 * float4(I1_vec2_p)
        - float4(I0_vec);
    }
    float sum_diff = dot(vec1, float4(1.0)) + dot(vec1, float4(1.0));
    float sum_diff_sq = dot(vec1, vec1) + dot(vec1, vec1);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    smem[i] = (vector_float2){sum_diff, sum_diff_sq};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < 4)
        smem[i] += smem[i + 4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < 2)
        smem[i] += smem[i + 2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i == 0)
        smem[0] += smem[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 reduce_add_result = smem[0];
    sum_diff = reduce_add_result.x;
    sum_diff_sq = reduce_add_result.y;
    
    return sum_diff_sq - sum_diff * sum_diff / n;
}


kernel void MIRFlow_invertSearch_fwd1(constant vector_float2 *U_ptr [[ buffer(MIRInvertSearch_U) ]],
                                      constant uchar *I0_ptr [[ buffer(MIRInvertSearch_I0) ]],
                                      constant uchar *I1_ptr [[ buffer(MIRInvertSearch_I1) ]],
                                      constant MIRInvertSearchOpt &opt [[ buffer(MIRInvertSearch_opt) ]],
                                      device vector_float2 *S_ptr [[ buffer(MIRInvertSearch_S) ]],
                                      uint is [[ threadgroup_position_in_grid ]],
                                      uint sid [[ thread_position_in_threadgroup ]]
                                      ) {
    int w = opt.w, h = opt.h, patch_size = opt.patch_size, patch_stride = opt.patch_stride, border_size = opt.border_size, ws = opt.ws;
    int patchSizeHalf = patch_size / 2;
    
    int i = is * patch_stride;
    int j = 0;
    int w_ext = w + 2 * border_size;
    
    float i_lower_limit = border_size - patch_size + 1.0f;
    float i_upper_limit = border_size + h - 1.0f;
    float j_lower_limit = border_size - patch_size + 1.0f;
    float j_upper_limit = border_size + w - 1.0f;
    
    threadgroup vector_float2 smem[8];
    
    vector_float2 prev_U = U_ptr[(i + patchSizeHalf) * w + j + patchSizeHalf];
    S_ptr[is * ws] = prev_U;
    j += patch_stride;
    
    for (int js = 1; js < ws; js++, j += patch_stride)
    {
        float2 U = U_ptr[(i + patchSizeHalf) * w + j + patchSizeHalf];
        
        float i_I1, j_I1, w00, w01, w10, w11;
        
        INIT_BILINEAR_WEIGHTS(U.x, U.y);
        float min_SSD = computeSSDMeanNorm(I0_ptr + i * w + j,
                                           I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                           w, w_ext, patch_size, w00, w01, w10, w11, sid, smem);
        
        INIT_BILINEAR_WEIGHTS(prev_U.x, prev_U.y);
        float cur_SSD = computeSSDMeanNorm(I0_ptr + i * w + j,
                                           I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                           w, w_ext, patch_size, w00, w01, w10, w11, sid, smem);
        
        prev_U = (cur_SSD < min_SSD) ? prev_U : U;
        S_ptr[is * ws + js] = prev_U;
    }
    S_ptr[is * w + sid] = vector_float2(is, sid);
    
}


float4 processPatchMeanNorm(constant uchar *I0_ptr,
                            constant uchar *I1_ptr,
                            constant short *I0x_ptr,
                            constant short *I0y_ptr,
                            int patch_size,
                            int I0_stride, int I1_stride,
                            float w00, float w01, float w10, float w11,
                            float x_grad_sum, float y_grad_sum
                            ) {
    const float inv_n = 1.0f / (float)(patch_size * patch_size);

    float sum_diff = 0.0, sum_diff_sq = 0.0;
    float sum_I0x_mul = 0.0, sum_I0y_mul = 0.0;

    uchar4 I1_vec2_lo = *I1_ptr;
    uchar4 I1_vec2_p_lo = *(I1_ptr + 1);
    uchar4 I1_vec2_hi = *(I1_ptr + 4);
    uchar4 I1_vec2_p_hi = *(I1_ptr + 5);

    for (int i = 0; i < 8; i++)
    {
        float4 vec_lo, vec_hi;
        {
            
            uchar4 I0_vec_lo = *(I0_ptr + i * I0_stride);
            
            uchar4 I1_vec1_lo = I1_vec2_lo;
            uchar4 I1_vec1_p_lo = I1_vec2_p_lo;
            
            I1_vec2_lo = *(I1_ptr + (i + 1) * I1_stride);
            I1_vec2_p_lo = *(I1_ptr + (i + 1) * I1_stride + 1);
            
            vec_lo = w00 * float4(I1_vec1_lo) + w01 * float4(I1_vec1_p_lo) +
            w10 * float4(I1_vec2_lo) + w11 * float4(I1_vec2_p_lo) -
            float4(I0_vec_lo);
        }
        {
            uchar4 I0_vec_hi = *(I0_ptr + i * I0_stride + 4);
            
            uchar4 I1_vec1_hi = I1_vec2_hi;
            uchar4 I1_vec1_p_hi = I1_vec2_hi;
            
            I1_vec2_hi = *(I1_ptr + (i + 1) * I1_stride + 4);
            I1_vec2_p_hi = *(I1_ptr + (i + 1) * I1_stride + 5);
            
            vec_hi = w00 * float4(I1_vec1_hi) + w01 * float4(I1_vec1_p_hi) +
            w10 * float4(I1_vec2_hi) + w11 * float4(I1_vec2_p_hi) -
            float4(I0_vec_hi);
        }

        sum_diff += (dot(vec_lo, 1.0) + dot(vec_hi, 1.0));
        sum_diff_sq += (dot(vec_lo, vec_lo) + dot(vec_hi, vec_hi));

        short4 I0x_vec_lo = *(I0x_ptr + i * I0_stride);
        short4 I0x_vec_hi = *(I0x_ptr + i * I0_stride + 4);
        short4 I0y_vec_lo = *(I0y_ptr + i * I0_stride);
        short4 I0y_vec_hi = *(I0y_ptr + i * I0_stride + 4);

        sum_I0x_mul += dot(vec_lo, float4(I0x_vec_lo));
        sum_I0x_mul += dot(vec_hi, float4(I0x_vec_hi));
        sum_I0y_mul += dot(vec_lo, float4(I0y_vec_lo));
        sum_I0y_mul += dot(vec_hi, float4(I0y_vec_hi));
    }

    float dst_dUx = sum_I0x_mul - sum_diff * x_grad_sum * inv_n;
    float dst_dUy = sum_I0y_mul - sum_diff * y_grad_sum * inv_n;
    float SSD = sum_diff_sq - sum_diff * sum_diff * inv_n;
    
    float4 r = float4(SSD, dst_dUx, dst_dUy, 0);
    return r;
}


kernel void MIRFlow_invertSearch_fwd2(constant vector_float2 *U_ptr [[ buffer(MIRInvertSearch_U) ]],
                                      constant uchar *I0_ptr [[ buffer(MIRInvertSearch_I0) ]],
                                      constant uchar *I1_ptr [[ buffer(MIRInvertSearch_I1) ]],
                                      constant short *I0x_ptr [[ buffer(MIRInvertSearch_I0x) ]],
                                      constant short *I0y_ptr [[ buffer(MIRInvertSearch_I0y) ]],
                                      constant float *xx_ptr [[ buffer(MIRInvertSearch_I0xx_buf) ]],
                                      constant float *yy_ptr [[ buffer(MIRInvertSearch_I0yy_buf) ]],
                                      constant float *xy_ptr [[ buffer(MIRInvertSearch_I0xy_buf) ]],
                                      constant float *x_ptr [[ buffer(MIRInvertSearch_I0x_buf) ]],
                                      constant float *y_ptr [[ buffer(MIRInvertSearch_I0y_buf) ]],
                                      device vector_float2 *S_ptr [[ buffer(MIRInvertSearch_S) ]],
                                      constant MIRInvertSearchOpt &opt,
                                      uint2 gid [[ thread_position_in_grid ]])
{
    if (gid.x >= (uint)opt.ws || gid.y >= (uint)opt.hs) {
        return;
    }
    int w = opt.w, h = opt.h, ws = opt.ws, hs = opt.hs,
    patch_stride = opt.patch_stride, patch_size = opt.patch_size, border_size = opt.border_size,
    num_inner_iter = opt.num_inner_iter;
    
    int js = gid.y;
    int is = gid.x;
    int i = is * patch_stride;
    int j = js *  patch_stride;
//    const int psz =  patch_size;
    int w_ext = w + 2 * border_size;
    int index = is * ws + js;
    
    if (js >= ws || is >= hs) return;
    
    float2 U0 = S_ptr[index];
    float2 cur_U = U0;
    float cur_xx = xx_ptr[index];
    float cur_yy = yy_ptr[index];
    float cur_xy = xy_ptr[index];
    float detH = cur_xx * cur_yy - cur_xy * cur_xy;
    
    float inv_detH = (fabs(detH) < EPS) ? 1.0 / EPS : 1.0 / detH;
    float invH11 = cur_yy * inv_detH;
    float invH12 = -cur_xy * inv_detH;
    float invH22 = cur_xx * inv_detH;
    
    float prev_SSD = INF;
    float x_grad_sum = x_ptr[index];
    float y_grad_sum = y_ptr[index];
    
    const float i_lower_limit = border_size - patch_size + 1.0f;
    const float i_upper_limit = border_size + h - 1.0f;
    const float j_lower_limit = border_size - patch_size + 1.0f;
    const float j_upper_limit = border_size + w - 1.0f;
    
    for (int t = 0; t < num_inner_iter; t++)
    {
        float i_I1, j_I1, w00, w01, w10, w11;
        INIT_BILINEAR_WEIGHTS(cur_U.x, cur_U.y);
        float4 res = processPatchMeanNorm(I0_ptr  + i * w + j,
                                          I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                          I0x_ptr + i * w + j,
                                          I0y_ptr + i * w + j,
                                          patch_size,
                                          w, w_ext, w00, w01, w10, w11,
                                          x_grad_sum, y_grad_sum);
        
        float SSD = res.x;
        float dUx = res.y;
        float dUy = res.z;
        float dx = invH11 * dUx + invH12 * dUy;
        float dy = invH12 * dUx + invH22 * dUy;
        
        cur_U -= float2(dx, dy);
        
        if (SSD >= prev_SSD)
            break;
        prev_SSD = SSD;
    }
    
    float2 vec = cur_U - U0;
    S_ptr[index] = (dot(vec, vec) <= (float)(patch_size * patch_size)) ? cur_U : U0;
}
