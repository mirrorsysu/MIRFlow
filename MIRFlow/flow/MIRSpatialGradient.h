//
//  MIRSpatialGradient.h
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/20.
//

#import <Foundation/Foundation.h>
@import Metal;

#import "MIRFlowShaderTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIRSpatialGradient : NSObject
+ (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer
                           src:(id<MTLBuffer>)src
                            vx:(id<MTLBuffer>)vx
                            vy:(id<MTLBuffer>)vy
                           opt:(MIRSpatialGradientOpt)opt;
@end

NS_ASSUME_NONNULL_END
