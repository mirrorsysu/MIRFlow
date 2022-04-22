//
//  MIRInvertSearch.h
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/19.
//

#import <Foundation/Foundation.h>
@import Metal;

#import "MIRFlowShaderTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIRInvertSearch : NSObject
+ (id<MTLCommandBuffer>)encode_fwd1:(id<MTLCommandBuffer>)commandBuffer
                                  U:(id<MTLBuffer>)U
                                 I0:(id<MTLBuffer>)I0
                                 I1:(id<MTLBuffer>)I1
                                  S:(id<MTLBuffer>)S
                                opt:(MIRInvertSearchOpt)opt;

+ (id<MTLCommandBuffer>)encode_fwd2:(id<MTLCommandBuffer>)commandBuffer
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
                                opt:(MIRInvertSearchOpt)opt;

+ (id<MTLCommandBuffer>)encode_bwd1:(id<MTLCommandBuffer>)commandBuffer
                                 I0:(id<MTLBuffer>)I0
                                 I1:(id<MTLBuffer>)I1
                                  S:(id<MTLBuffer>)S
                                opt:(MIRInvertSearchOpt)opt;

+ (id<MTLCommandBuffer>)encode_bwd2:(id<MTLCommandBuffer>)commandBuffer
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
                                opt:(MIRInvertSearchOpt)opt;

@end

NS_ASSUME_NONNULL_END
