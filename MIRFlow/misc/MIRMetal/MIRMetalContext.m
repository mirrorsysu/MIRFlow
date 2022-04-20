//
//  MIRMetalContext.m
//  MIRFaceMorph
//
//  Created by MirrorMac on 2021/11/26.
//

#import "MIRMetalContext.h"

@interface MIRMetalContext ()
@property (nonatomic) id<MTLDevice> device;
@property (nonatomic) id<MTLCommandQueue> commandQueue;
@property (nonatomic) id<MTLLibrary> defaultLibrary;
@property (nonatomic) MTLCaptureManager *captureManager;

@property (nonatomic) NSCache *functionCache;
@property (nonatomic) NSCache *renderPSOCache;
@property (nonatomic) NSCache *computePSOCache;
@end

@implementation MIRMetalContext


#pragma mark - life cycle
+ (instancetype)sharedInstance {
    static id __instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        __instance = [[self alloc] init];
    });
    return __instance;
}

- (instancetype)init {
    if (self = [super init]) {
        self.device = MTLCreateSystemDefaultDevice();
        self.commandQueue = [self.device newCommandQueue];
        self.defaultLibrary = [self.device newDefaultLibrary];
        self.functionCache = [[NSCache alloc] init];
        self.renderPSOCache = [[NSCache alloc] init];
        self.computePSOCache = [[NSCache alloc] init];
    }
    return self;
}


#pragma mark - property
+ (id<MTLDevice>)device {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    return SELF.device;
}

+ (id<MTLCommandQueue>)commandQueue {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    return SELF.commandQueue;
}

+ (id<MTLLibrary>)defaultLibrary {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    return SELF.defaultLibrary;
}

#pragma mark -
+ (id<MTLCommandBuffer>)makeCommandBuffer:(id<MTLCommandQueue>)commandQueue {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    if (@available(iOS 14.0, *)) {
#ifdef DEBUG
        // more debug info
        static MTLCommandBufferDescriptor *desc;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            desc = [[MTLCommandBufferDescriptor alloc] init];
            desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        });
        return [SELF.commandQueue commandBufferWithDescriptor:desc];
#endif
    }
    return [SELF.commandQueue commandBuffer];
}

+ (id<MTLCommandBuffer>)makeCommandBuffer {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    return [self makeCommandBuffer:SELF.commandQueue];
}


#pragma mark -
+ (id<MTLFunction>)functionForFunctionName:(NSString *)name {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    id<MTLFunction> function = [SELF.functionCache objectForKey:name];
    if (!function) {
        function = [SELF.defaultLibrary newFunctionWithName:name];
        function.label = name;
        [SELF.functionCache setObject:function forKey:name];
    }
    return function;
}

+ (MTLRenderPipelineDescriptor *)renderPipelineDescWithVertexName:(NSString *)vertexName
                                                     fragmentName:(NSString *)fragmentName {
    NSString *name = [NSString stringWithFormat:@"v:%@_f:%@", vertexName, fragmentName];
    MTLRenderPipelineDescriptor *desc;{
        id<MTLFunction> vertexFunc = [self functionForFunctionName:vertexName];
        id<MTLFunction> fragmentFunc = [self functionForFunctionName:fragmentName];
        NSCAssert(vertexFunc, @"找不到vertexFunc, 请检查vertexName");
        NSCAssert(fragmentFunc, @"找不到fragmentFunc, 请检查fragmentName");
        
        desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.label = name;
        desc.vertexFunction = vertexFunc;
        desc.fragmentFunction = fragmentFunc;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    }
    
    return desc;
}

+ (id<MTLRenderPipelineState>)renderPSOWithVertexName:(NSString *)vertexName
                                         fragmentName:(NSString *)fragmentName {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    NSString *name = [NSString stringWithFormat:@"v:%@_f:%@", vertexName, fragmentName];
    id<MTLRenderPipelineState> pso = [SELF.renderPSOCache objectForKey:name];
    if (!pso) {
        MTLRenderPipelineDescriptor *desc = [self renderPipelineDescWithVertexName:vertexName fragmentName:fragmentName];
        NSError *error;
        pso = [SELF.device newRenderPipelineStateWithDescriptor:desc error:&error];
        NSCParameterAssert(!error);
        [SELF.renderPSOCache setObject:pso forKey:name];
    }
    return pso;
}

+ (MTLComputePipelineDescriptor *)computePipelineDescWithFuncName:(NSString *)funcName {
    NSString *name = [NSString stringWithFormat:@"kernel:%@", funcName];
    MTLComputePipelineDescriptor *desc;{
        id<MTLFunction> func = [self functionForFunctionName:funcName];
        NSCAssert(func, @"找不到function, 请检查funcName");
        desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.label = name;
        desc.computeFunction = func;
    }
    
    return desc;
}

+ (id<MTLComputePipelineState>)computePSOWithFuncName:(NSString *)funcName {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    NSString *name = [NSString stringWithFormat:@"kernel:%@", funcName];
    id<MTLComputePipelineState> pso = [SELF.computePSOCache objectForKey:name];
    if (!pso) {
        MTLComputePipelineDescriptor *desc = [self computePipelineDescWithFuncName:funcName];
        NSError *error;
        pso = [SELF.device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
        NSCParameterAssert(!error);
        [SELF.renderPSOCache setObject:pso forKey:name];
    }
    return pso;
}


#pragma mark -
+ (MTLRenderPassDescriptor *)renderPassDescWithTexture:(id<MTLTexture>)texture {
    MTLRenderPassDescriptor *renderPassDesc; {
        renderPassDesc = [[MTLRenderPassDescriptor alloc] init];
        renderPassDesc.colorAttachments[0].texture = texture;
        renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    }
    return renderPassDesc;
}

+ (MTLRenderPassDescriptor *)renderPassDescWithTexture:(id<MTLTexture>)texture clearColor:(MTLClearColor)clearColor {
    MTLRenderPassDescriptor *renderPassDesc; {
        renderPassDesc = [[MTLRenderPassDescriptor alloc] init];
        renderPassDesc.colorAttachments[0].texture = texture;
        renderPassDesc.colorAttachments[0].clearColor = clearColor;
        renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    }
    return renderPassDesc;
}


#pragma mark -
+ (void)startCapture {
    MIRMetalContext *SELF = [MIRMetalContext sharedInstance];
    MTLCaptureDescriptor *desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = SELF.commandQueue;
    NSError *error = nil;
    [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:desc error:&error];
    NSCParameterAssert(!error);
}

+ (void)stopCapture {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

@end
