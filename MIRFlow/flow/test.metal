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

kernel void MIRFlow_testU(device vector_float2 *input [[ buffer(0) ]],
                          texture2d<half, access::write> output1 [[ texture(0) ]],
                          texture2d<half, access::write> output2 [[ texture(1) ]],
                          constant int &w [[ buffer(1) ]],
                          constant int &h [[ buffer(2) ]],
                          uint2 gid [[ thread_position_in_grid ]]
                          ) {
    if (gid.x >= (uint)w || gid.y >= (uint)h) {
        return;
    }
    vector_float2 r = input[gid.y * w + gid.x];
    half l1 = r.x / 20;
    half l2 = r.y / 20;
    output1.write(half4(l1, l1, l1, 1), gid);
    output2.write(half4(l2, l2, l2, 1), gid);
}

kernel void MIRFlow_testI(device uchar *input [[ buffer(0) ]],
                          texture2d<half, access::write> output [[ texture(0) ]],
                          constant int &w [[ buffer(1) ]],
                          constant int &h [[ buffer(2) ]],
                          uint2 gid [[ thread_position_in_grid ]]
                          ) {
    if (gid.x >= (uint)w || gid.y >= (uint)h) {
        return;
    }
    half r = input[gid.y * w + gid.x] / 255.0;
    output.write(half4(r, r, r, 1), gid);
}

