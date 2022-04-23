//
//  MIRDensification.h
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/20.
//

#import <Foundation/Foundation.h>
@import Metal;

#import "MIRFlowShaderTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIRDensification : NSObject
+ (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer
                             S:(id<MTLBuffer>)S
                            I0:(id<MTLBuffer>)I0
                            I1:(id<MTLBuffer>)I1
                             U:(id<MTLBuffer>)U
                           opt:(MIRDensificationOpt)opt;
@end

NS_ASSUME_NONNULL_END
