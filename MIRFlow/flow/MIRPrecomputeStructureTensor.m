//
//  MIRPrecomputeStructureTensor.m
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/19.
//

#import "MIRPrecomputeStructureTensor.h"

#import "MIRMetalContext.h"

@implementation MIRPrecomputeStructureTensor

+ (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer
                           I0x:(id<MTLBuffer>)I0x
                           I0y:(id<MTLBuffer>)I0y
                      I0xx_aux:(id<MTLBuffer>)I0xx_aux
                      I0yy_aux:(id<MTLBuffer>)I0yy_aux
                      I0xy_aux:(id<MTLBuffer>)I0xy_aux
                       I0x_aux:(id<MTLBuffer>)I0x_aux
                       I0y_aux:(id<MTLBuffer>)I0y_aux
                      I0xx_buf:(id<MTLBuffer>)I0xx_buf
                      I0yy_buf:(id<MTLBuffer>)I0yy_buf
                      I0xy_buf:(id<MTLBuffer>)I0xy_buf
                       I0x_buf:(id<MTLBuffer>)I0x_buf
                       I0y_buf:(id<MTLBuffer>)I0y_buf
                           opt:(MIRPrecomputeStructureTensorOpt)opt {
    
    // hor
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_PrecomputeStructureTensor_hor"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_PrecomputeStructureTensor_hor"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:I0x offset:0 atIndex:MIRPrecomputeStructureTensor_I0x];
        [encoder setBuffer:I0y offset:0 atIndex:MIRPrecomputeStructureTensor_I0y];
        [encoder setBytes:&opt length:sizeof(MIRPrecomputeStructureTensorOpt) atIndex:MIRPrecomputeStructureTensor_opt];
        [encoder setBuffer:I0xx_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0xx_aux];
        [encoder setBuffer:I0yy_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0yy_aux];
        [encoder setBuffer:I0xy_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0xy_aux];
        [encoder setBuffer:I0x_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0x_aux];
        [encoder setBuffer:I0y_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0y_aux];
        
        MTLSize threadgroupSize = MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1);
        MTLSize threadgroupCount = MTLSizeMake((opt.h + threadgroupSize.width - 1) / threadgroupSize.width,
                                               1,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // ver
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_PrecomputeStructureTensor_ver"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_PrecomputeStructureTensor_ver"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:I0xx_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0xx_aux];
        [encoder setBuffer:I0yy_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0yy_aux];
        [encoder setBuffer:I0xy_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0xy_aux];
        [encoder setBuffer:I0x_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0x_aux];
        [encoder setBuffer:I0y_aux offset:0 atIndex:MIRPrecomputeStructureTensor_I0y_aux];
        [encoder setBytes:&opt length:sizeof(MIRPrecomputeStructureTensorOpt) atIndex:MIRPrecomputeStructureTensor_opt];
        [encoder setBuffer:I0xx_buf offset:0 atIndex:MIRPrecomputeStructureTensor_I0xx_buf];
        [encoder setBuffer:I0yy_buf offset:0 atIndex:MIRPrecomputeStructureTensor_I0yy_buf];
        [encoder setBuffer:I0xy_buf offset:0 atIndex:MIRPrecomputeStructureTensor_I0xy_buf];
        [encoder setBuffer:I0x_buf offset:0 atIndex:MIRPrecomputeStructureTensor_I0x_buf];
        [encoder setBuffer:I0y_buf offset:0 atIndex:MIRPrecomputeStructureTensor_I0y_buf];
        
        MTLSize threadgroupSize = MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1);
        MTLSize threadgroupCount = MTLSizeMake((opt.ws + threadgroupSize.width - 1) / threadgroupSize.width,
                                               1,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    return commandBuffer;
}
@end
