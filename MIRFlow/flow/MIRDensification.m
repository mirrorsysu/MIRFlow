//
//  MIRDensification.m
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/20.
//

#import "MIRDensification.h"

#import "MIRMetalContext.h"

@implementation MIRDensification
+ (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer
                             S:(id<MTLBuffer>)S
                            I0:(id<MTLBuffer>)I0
                            I1:(id<MTLBuffer>)I1
                             U:(id<MTLBuffer>)U
                           opt:(MIRDensificationOpt)opt {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRDensification"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_densification"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:U offset:0 atIndex:MIRDensification_U];
        [encoder setBuffer:I0 offset:0 atIndex:MIRDensification_I0];
        [encoder setBuffer:I1 offset:0 atIndex:MIRDensification_I1];
        [encoder setBytes:&opt length:sizeof(MIRDensificationOpt) atIndex:MIRDensification_opt];
        [encoder setBuffer:S offset:0 atIndex:MIRDensification_S];
        
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
