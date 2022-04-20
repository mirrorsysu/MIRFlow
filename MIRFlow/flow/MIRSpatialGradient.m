//
//  MIRSpatialGradient.m
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/20.
//

#import "MIRSpatialGradient.h"

#import "MIRMetalContext.h"

@implementation MIRSpatialGradient
+ (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer
                           src:(id<MTLBuffer>)src
                            vx:(id<MTLBuffer>)vx
                            vy:(id<MTLBuffer>)vy
                           opt:(MIRSpatialGradientOpt)opt {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRSpatialGradient"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_SpatialGradientKernel"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:src offset:0 atIndex:MIRSpatialGradient_src];
        [encoder setBuffer:vx offset:0 atIndex:MIRSpatialGradient_vx];
        [encoder setBuffer:vy offset:0 atIndex:MIRSpatialGradient_vy];
        [encoder setBytes:&opt length:sizeof(MIRSpatialGradientOpt) atIndex:MIRSpatialGradient_opt];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((opt.w + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (opt.h + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    return commandBuffer;
}
@end
