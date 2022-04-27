//
//  MIRPrecomputeStructureTensor.h
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/19.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "MIRFlowShaderTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIRPrecomputeStructureTensor : NSObject
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
                           opt:(MIRPrecomputeStructureTensorOpt)opt;
@end

NS_ASSUME_NONNULL_END
