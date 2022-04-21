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
        MTLSize threadgroupCount = MTLSizeMake(opt.hs,
                                               1,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    return commandBuffer;
}

+ (id<MTLCommandBuffer>)encode_fw2:(id<MTLCommandBuffer>)commandBuffer
                                 U:(id<MTLBuffer>)U
                                I0:(id<MTLBuffer>)I0
                                I1:(id<MTLBuffer>)I1
                               I0x:(id<MTLBuffer>)I0x
                               I0y:(id<MTLBuffer>)I0y
                          I0xx_buf:(id<MTLBuffer>)I0xx_buf
                          I0yy_buf:(id<MTLBuffer>)I0yy_buf
                          I0xy_buf:(id<MTLBuffer>)I0xy_buf
                           I0x_buf:(id<MTLBuffer>)I0x_buf
                           I0y_buf:(id<MTLBuffer>)I0y_buf
                                 S:(id<MTLBuffer>)S
                               opt:(MIRInvertSearchOpt)opt {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_invertSearch_fwd2"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_invertSearch_fwd2"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:U offset:0 atIndex:MIRInvertSearch_U];
        [encoder setBuffer:I0 offset:0 atIndex:MIRInvertSearch_I0];
        [encoder setBuffer:I1 offset:0 atIndex:MIRInvertSearch_I1];
        
        [encoder setBuffer:I0x offset:0 atIndex:MIRInvertSearch_I0x];
        [encoder setBuffer:I0y offset:0 atIndex:MIRInvertSearch_I0y];
        [encoder setBuffer:I0xx_buf offset:0 atIndex:MIRInvertSearch_I0xx_buf];
        [encoder setBuffer:I0yy_buf offset:0 atIndex:MIRInvertSearch_I0yy_buf];
        [encoder setBuffer:I0xy_buf offset:0 atIndex:MIRInvertSearch_I0xy_buf];
        [encoder setBuffer:I0x_buf offset:0 atIndex:MIRInvertSearch_I0x_buf];
        [encoder setBuffer:I0y_buf offset:0 atIndex:MIRInvertSearch_I0y_buf];
        
        [encoder setBuffer:S offset:0 atIndex:MIRInvertSearch_S];
        [encoder setBytes:&opt length:sizeof(MIRInvertSearchOpt) atIndex:MIRInvertSearch_opt];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((opt.ws + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (opt.hs + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    return commandBuffer;
}

@end
