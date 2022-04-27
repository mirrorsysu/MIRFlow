//
//  MIRFlow.h
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/19.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "MIRFlowShaderTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIRFlow : NSObject
+ (id<MTLCommandBuffer>)grayscale:(id<MTLCommandBuffer>)commandBuffer input:(id<MTLTexture>)input output:(id<MTLTexture>)output;
- (void)generateBufferWithWidth:(int)width height:(int)height;
- (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer I0:(id<MTLTexture>)I0 I1:(id<MTLTexture>)I1;
@end

NS_ASSUME_NONNULL_END
