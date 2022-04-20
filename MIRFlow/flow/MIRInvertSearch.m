//
//  MIRInvertSearch.m
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/19.
//

#import "MIRInvertSearch.h"

#import "MIRMetalContext.h"

@implementation MIRInvertSearch
+ (id<MTLCommandBuffer>)encode_fw1:(id<MTLCommandBuffer>)commandBuffer
                                 U:(id<MTLBuffer>)U
                                I0:(id<MTLBuffer>)I0
                                I1:(id<MTLBuffer>)I1
                                 S:(id<MTLBuffer>)S
                               opt:(MIRInvertSearchOpt)opt {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_invertSearch_fwd1"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_invertSearch_fwd1"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:U offset:0 atIndex:MIRInvertSearch_U];
        [encoder setBuffer:I0 offset:0 atIndex:MIRInvertSearch_I0];
        [encoder setBuffer:I1 offset:0 atIndex:MIRInvertSearch_I1];
        [encoder setBuffer:S offset:0 atIndex:MIRInvertSearch_S];
        [encoder setBytes:&opt length:sizeof(MIRInvertSearchOpt) atIndex:MIRInvertSearch_opt];
        
        MTLSize threadgroupSize = MTLSizeMake(8, 1, 1);
        MTLSize threadgroupCount = MTLSizeMake(opt.hs * 8,
                                               1,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    return commandBuffer;
}
@end
