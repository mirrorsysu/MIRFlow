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
+ (id<MTLCommandBuffer>)encode_fw1:(id<MTLCommandBuffer>)commandBuffer
                                 U:(id<MTLBuffer>)U
                                I0:(id<MTLBuffer>)I0
                                I1:(id<MTLBuffer>)I1
                                 S:(id<MTLBuffer>)S
                               opt:(MIRInvertSearchOpt)opt;
@end

NS_ASSUME_NONNULL_END
