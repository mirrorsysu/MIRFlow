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

struct RasterizerData {
    float4 position [[position]];
    float2 textureCoordinate;
};

kernel void MIRFlow_scaleInput(texture2d<half, access::sample> texture [[ texture(0) ]],
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

kernel void MIRFlow_resizeU(constant vector_float2 *src [[ buffer(0) ]],
                            device vector_float2 *dst [[ buffer(1) ]],
                            constant int &dstW [[ buffer(2) ]],
                            constant int &dstH [[ buffer(3) ]],
                            uint2 gid [[ thread_position_in_grid ]]
                            ) {
    if (gid.x >= (uint)dstW || gid.y >= (uint)dstH) {
        return;
    }
    int srcW = dstW / 2;
    int srcH = dstH / 2;
    
    float2 pos = float2(gid) / float2(dstW, dstH) * float2(srcW, srcH);
    float u = fract(pos.x);
    float v = fract(pos.y);
    int2 i00 = int2(floor(pos));
    int2 i01 = int2(min(int(floor(pos.x) + 1), srcW-1), floor(pos.y));
    int2 i10 = int2(floor(pos.x), min(int(floor(pos.y) + 1), srcH-1));
    int2 i11 = int2(min(int(floor(pos.x) + 1), srcW-1), min(int(floor(pos.y) + 1), srcH-1));
    
#define at(xy) *(src + (xy).y * srcW + (xy).x)
    vector_float2 v00 = at(i00);
    vector_float2 v01 = at(i01);
    vector_float2 v10 = at(i10);
    vector_float2 v11 = at(i11);
    vector_float2 r = v00*(1-u)*(1-v) + v01*u*(1-v) + v10*(1-u)*v + v11*u*v;
#undef at
    dst[gid.y * dstW + gid.x] = r * 2;
    
    //    if (gid.x >= (uint)srcW || gid.y >= (uint)srcH) {
    //        return;
    //    }
    //    dst[gid.y * dstW + gid.x] = src[gid.y * srcW + gid.x];
}

// BORDER_REPLICATE
kernel void MIRFlow_copyMakeBorder(texture2d<half, access::sample> texture [[ texture(0) ]],
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
