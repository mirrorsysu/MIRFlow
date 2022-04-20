//
//  MIRMetalContext.h
//  MIRFaceMorph
//
//  Created by MirrorMac on 2021/11/26.
//

#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#import "MIRMetalShaderTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIRMetalContext : NSObject
+ (id<MTLDevice>)device;
+ (id<MTLCommandQueue>)commandQueue;
+ (id<MTLLibrary>)defaultLibrary;

+ (id<MTLCommandBuffer>)makeCommandBuffer:(id<MTLCommandQueue>)commandQueue;
+ (id<MTLCommandBuffer>)makeCommandBuffer;

+ (id<MTLFunction>)functionForFunctionName:(NSString *)name;
+ (MTLRenderPipelineDescriptor *)renderPipelineDescWithVertexName:(NSString *)vertexName
                                                     fragmentName:(NSString *)fragmentName;
+ (id<MTLRenderPipelineState>)renderPSOWithVertexName:(NSString *)vertexName
                                         fragmentName:(NSString *)fragmentName;
+ (MTLComputePipelineDescriptor *)computePipelineDescWithFuncName:(NSString *)funcName;
+ (id<MTLComputePipelineState>)computePSOWithFuncName:(NSString *)kernelName;

+ (MTLRenderPassDescriptor *)renderPassDescWithTexture:(id<MTLTexture>)texture;
+ (MTLRenderPassDescriptor *)renderPassDescWithTexture:(id<MTLTexture>)texture clearColor:(MTLClearColor)clearColor;

+ (void)startCapture;
+ (void)stopCapture;
@end

NS_ASSUME_NONNULL_END
