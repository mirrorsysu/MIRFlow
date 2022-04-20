//
//  a.metal
//  MIRFaceMorph
//
//  Created by MirrorMac on 2021/11/30.
//

#include <metal_stdlib>

#import "MIRMetalShaderTypes.h"

using namespace metal;

struct RasterizerData {
    float4 position [[position]];
    float2 textureCoordinate;
};
constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);

vertex RasterizerData MIRMetalVertexShader(uint vertexID [[ vertex_id ]],
                                           constant MIRMetalVertex *vertexData [[ buffer(0) ]]
                                           ) {
    RasterizerData out;
    out.position = float4(0.0, 0.0, 0.0, 1.0);
    out.position.xy = vertexData[vertexID].position.xy;
    out.textureCoordinate = vertexData[vertexID].textureCoordinate;
    return out;
}

fragment half4 MIRMetalFragmentShader(RasterizerData in [[stage_in]],
                                      texture2d<half> inputImageTexture [[ texture(0) ]]
                                      ) {
    half4 color = inputImageTexture.sample(textureSampler, in.textureCoordinate);
    return color;
}
