//
//  MIRFlow.m
//  MIRFlow
//
//  Created by MirrorMac on 2022/4/19.
//

#import "MIRFlow.h"

#import "MIRMetalContext.h"

#import "MIRSpatialGradient.h"
#import "MIRPrecomputeStructureTensor.h"
#import "MIRInvertSearch.h"
#import "MIRDensification.h"

@interface MIRFlow () {
    int _coarsest_scale;
    int _finest_scale;
    
    int _num_iter;
    int _grad_descent_iter;
    
    int _patch_stride;
    int _patch_size;
    
    int _border_size;
    
    int _inputW;
    int _inputH;
    
    int _coarsest_width;
    int _coarsest_height;
    int _ws, _hs, _w, _h;
    
    id<MTLTexture> _I0;
    id<MTLTexture> _I1;
    
    NSMutableArray<id<MTLBuffer>> *_I0s;
    NSMutableArray<id<MTLBuffer>> *_I1s;
    NSMutableArray<id<MTLBuffer>> *_I1s_ext;
    NSMutableArray<id<MTLBuffer>> *_I0xs;
    NSMutableArray<id<MTLBuffer>> *_I0ys;
    NSMutableArray<id<MTLBuffer>> *_Us;
    
    id<MTLBuffer> _S;
    
    id<MTLBuffer> _I0xx_buf;
    id<MTLBuffer> _I0yy_buf;
    id<MTLBuffer> _I0xy_buf;
    id<MTLBuffer> _I0x_buf;
    id<MTLBuffer> _I0y_buf;
    
    id<MTLBuffer> _I0xx_aux;
    id<MTLBuffer> _I0yy_aux;
    id<MTLBuffer> _I0xy_aux;
    id<MTLBuffer> _I0x_aux;
    id<MTLBuffer> _I0y_aux;
}
@end

@implementation MIRFlow
- (instancetype)init {
    if (self = [super init]) {
        _coarsest_scale = 6;
        _finest_scale = 2;
        
        _num_iter = 2;
        _grad_descent_iter = 16;
        
        _patch_stride = 4;
        _patch_size = 8;
        
        _border_size = 16;
        
        _inputW = 100;
        _inputH = 100;
    }
    return self;
}

+ (id<MTLCommandBuffer>)grayscale:(id<MTLCommandBuffer>)commandBuffer input:(id<MTLTexture>)input output:(id<MTLTexture>)output {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_grayscale"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_grayscale"];
        [encoder setComputePipelineState:pso];
        
        [encoder setTexture:input atIndex:0];
        [encoder setTexture:output atIndex:1];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((output.width + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (output.height + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    return commandBuffer;
}

+ (id<MTLCommandBuffer>)testU:(id<MTLCommandBuffer>)commandBuffer input:(id<MTLBuffer>)input output1:(id<MTLTexture>)output1 output2:(id<MTLTexture>)output2 w:(int)w h:(int)h {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_testU"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_testU"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setTexture:output1 atIndex:0];
        [encoder setTexture:output2 atIndex:1];
        [encoder setBytes:&w length:sizeof(int) atIndex:1];
        [encoder setBytes:&h length:sizeof(int) atIndex:2];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((w + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (h + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    return commandBuffer;
}

+ (id<MTLCommandBuffer>)testI:(id<MTLCommandBuffer>)commandBuffer input:(id<MTLBuffer>)input output:(id<MTLTexture>)output w:(int)w h:(int)h {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_testI"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_testI"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setTexture:output atIndex:0];
        [encoder setBytes:&w length:sizeof(int) atIndex:1];
        [encoder setBytes:&h length:sizeof(int) atIndex:2];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((w + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (h + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    return commandBuffer;
}

+ (id<MTLCommandBuffer>)resizeU:(id<MTLCommandBuffer>)commandBuffer
                            src:(id<MTLBuffer>)src
                            dst:(id<MTLBuffer>)dst
                           dstW:(int)dstW
                           dstH:(int)dstH {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_resizeU"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_resizeU"];
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:src offset:0 atIndex:0];
        [encoder setBuffer:dst offset:0 atIndex:1];
        [encoder setBytes:&dstW length:sizeof(int) atIndex:2];
        [encoder setBytes:&dstH length:sizeof(int) atIndex:3];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((dstW + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (dstH + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    return commandBuffer;
}

+ (id<MTLCommandBuffer>)scaleInput:(id<MTLCommandBuffer>)commandBuffer texture:(id<MTLTexture>)texture buffer:(id<MTLBuffer>)buffer width:(int)width height:(int)height {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_scaleInput"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_scaleInput"];
        [encoder setComputePipelineState:pso];
        
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBytes:&width length:sizeof(int) atIndex:1];
        [encoder setBytes:&height length:sizeof(int) atIndex:2];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((width + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (height + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    return commandBuffer;
}

+ (id<MTLCommandBuffer>)copyMakeBorder:(id<MTLCommandBuffer>)commandBuffer texture:(id<MTLTexture>)texture buffer:(id<MTLBuffer>)buffer width:(int)width height:(int)height border:(int)border {
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setLabel:@"MIRFlow_copyMakeBorder"];
        id<MTLComputePipelineState> pso = [MIRMetalContext computePSOWithFuncName:@"MIRFlow_copyMakeBorder"];
        [encoder setComputePipelineState:pso];
        
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBytes:&width length:sizeof(int) atIndex:1];
        [encoder setBytes:&height length:sizeof(int) atIndex:2];
        [encoder setBytes:&border length:sizeof(int) atIndex:3];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake((width + 2*border + threadgroupSize.width - 1) / threadgroupSize.width,
                                               (height + 2*border + threadgroupSize.height - 1) / threadgroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    return commandBuffer;
}

- (void)generateBufferWithWidth:(int)width height:(int)height {
    _inputW = width;
    _inputH = height;
    
    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;
    
    _I0s = [NSMutableArray arrayWithCapacity:_coarsest_scale];
    _I1s = [NSMutableArray arrayWithCapacity:_coarsest_scale];
    _I1s_ext = [NSMutableArray arrayWithCapacity:_coarsest_scale];
    _I0xs = [NSMutableArray arrayWithCapacity:_coarsest_scale];
    _I0ys = [NSMutableArray arrayWithCapacity:_coarsest_scale];
    _Us = [NSMutableArray arrayWithCapacity:_coarsest_scale];
    for (int i = 0; i <= _coarsest_scale; i++) {
        [_I0s addObject:(id<MTLBuffer>)NSNull.null];
        [_I1s addObject:(id<MTLBuffer>)NSNull.null];
        [_I1s_ext addObject:(id<MTLBuffer>)NSNull.null];
        [_I0xs addObject:(id<MTLBuffer>)NSNull.null];
        [_I0ys addObject:(id<MTLBuffer>)NSNull.null];
        [_Us addObject:(id<MTLBuffer>)NSNull.null];
        
        if (i == _finest_scale) {
            cur_rows = height / fraction;
            cur_cols = width / fraction;
            
            _I0s[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(unsigned char) options:MTLResourceStorageModeShared];
            _I1s[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(unsigned char) options:MTLResourceStorageModeShared];
            
            _S = [MIRMetalContext.device newBufferWithLength:(cur_rows/_patch_stride) *(cur_cols/_patch_stride) * sizeof(vector_float2) options:MTLResourceStorageModeShared];
            
            _I0xx_buf = [MIRMetalContext.device newBufferWithLength:(cur_rows/_patch_stride) *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0yy_buf = [MIRMetalContext.device newBufferWithLength:(cur_rows/_patch_stride) *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0xy_buf = [MIRMetalContext.device newBufferWithLength:(cur_rows/_patch_stride) *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0x_buf = [MIRMetalContext.device newBufferWithLength:(cur_rows/_patch_stride) *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0y_buf = [MIRMetalContext.device newBufferWithLength:(cur_rows/_patch_stride) *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            
            _I0xx_aux = [MIRMetalContext.device newBufferWithLength:cur_rows *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0yy_aux = [MIRMetalContext.device newBufferWithLength:cur_rows *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0xy_aux = [MIRMetalContext.device newBufferWithLength:cur_rows *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0x_aux = [MIRMetalContext.device newBufferWithLength:cur_rows *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
            _I0y_aux = [MIRMetalContext.device newBufferWithLength:cur_rows *(cur_cols/_patch_stride) * sizeof(float) options:MTLResourceStorageModeShared];
        } else if (i > _finest_scale) {
            cur_rows /= 2;
            cur_cols /= 2;
            
            _I0s[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(unsigned char) options:MTLResourceStorageModeShared];
            _I1s[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(unsigned char) options:MTLResourceStorageModeShared];
        }
        
        if (i >= _finest_scale) {
            _I1s_ext[i] = [MIRMetalContext.device newBufferWithLength:(cur_rows+2*_border_size) * (cur_cols+2*_border_size) * sizeof(unsigned char) options:MTLResourceStorageModeShared];
            
            _I0xs[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(short) options:MTLResourceStorageModeShared];
            _I0ys[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(short) options:MTLResourceStorageModeShared];
            
            _Us[i] = [MIRMetalContext.device newBufferWithLength:cur_rows * cur_cols * sizeof(vector_float2) options:MTLResourceStorageModeShared];
            
            // TODO: xxx
            //            variational_refinement_processors[i]->setAlpha(variational_refinement_alpha);
            //            variational_refinement_processors[i]->setDelta(variational_refinement_delta);
            //            variational_refinement_processors[i]->setGamma(variational_refinement_gamma);
            //            variational_refinement_processors[i]->setSorIterations(5);
            //            variational_refinement_processors[i]->setFixedPointIterations(variational_refinement_iter);
        }
        
        if (i == _coarsest_scale) {
            _coarsest_width = cur_cols;
            _coarsest_height = cur_rows;
        }
        
        fraction *= 2;
    }
}

- (id<MTLCommandBuffer>)prepareBuffer:(id<MTLCommandBuffer>)commandBuffer {
    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;
    
    for (int i = 0; i <= _coarsest_scale; i++) {
        if (i == _finest_scale) {
            cur_rows = _inputH / fraction;
            cur_cols = _inputW / fraction;
            
            commandBuffer = [self.class scaleInput:commandBuffer texture:_I0 buffer:_I0s[i] width:cur_cols height:cur_rows];
            commandBuffer =[self.class scaleInput:commandBuffer texture:_I1 buffer:_I1s[i] width:cur_cols height:cur_rows];
            {   // FIXME: DEBUG
                int w = cur_cols;
                int h = cur_rows;
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
                desc.usage = MTLTextureUsageShaderWrite;
                id<MTLTexture> texture = [MIRMetalContext.device newTextureWithDescriptor:desc];
                commandBuffer = [self.class testI:commandBuffer input:_I0s[i] output:texture w:w h:h];
                commandBuffer = [self.class testI:commandBuffer input:_I1s[i] output:texture w:w h:h];
            }
            
        } else if (i > _finest_scale) {
            cur_rows /= 2;
            cur_cols /= 2;
            commandBuffer = [self.class copyMakeBorder:commandBuffer texture:_I1 buffer:_I1s_ext[i] width:cur_cols height:cur_rows border:_border_size];
            commandBuffer = [self.class scaleInput:commandBuffer texture:_I0 buffer:_I0s[i] width:cur_cols height:cur_rows];
            commandBuffer = [self.class scaleInput:commandBuffer texture:_I1 buffer:_I1s[i] width:cur_cols height:cur_rows];
            {   // FIXME: DEBUG
                int w = cur_cols;
                int h = cur_rows;
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
                desc.usage = MTLTextureUsageShaderWrite;
                id<MTLTexture> texture = [MIRMetalContext.device newTextureWithDescriptor:desc];
                commandBuffer = [self.class testI:commandBuffer input:_I0s[i] output:texture w:w h:h];
                commandBuffer = [self.class testI:commandBuffer input:_I1s[i] output:texture w:w h:h];
            }
            {   // FIXME: DEBUG
                int w = cur_cols + 2*_border_size;
                int h = cur_rows + 2*_border_size;
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
                desc.usage = MTLTextureUsageShaderWrite;
                id<MTLTexture> texture = [MIRMetalContext.device newTextureWithDescriptor:desc];
                commandBuffer = [self.class testI:commandBuffer input:_I1s_ext[i] output:texture w:w h:h];
            }
        }
        
        if (i >= _finest_scale) {
            MIRSpatialGradientOpt opt = { .w=cur_cols, .h=cur_rows };
            commandBuffer = [MIRSpatialGradient encode:commandBuffer src:_I0s[i] vx:_I0xs[i] vy:_I0ys[i] opt:opt];
        }
        
        fraction *= 2;
    }
    
    return commandBuffer;
}

- (id<MTLCommandBuffer>)encode:(id<MTLCommandBuffer>)commandBuffer I0:(id<MTLTexture>)I0 I1:(id<MTLTexture>)I1 {
    _I0 = I0;
    _I1 = I1;
    
    commandBuffer = [self prepareBuffer:commandBuffer];
    
    memset(_Us[_coarsest_scale].contents, 0, _Us[_coarsest_scale].length);
    
    _w = _coarsest_width;
    _h = _coarsest_height;
    for (int i = _coarsest_scale; i >= _finest_scale; i--) {
        if (i < _coarsest_scale) {
            _w *= 2;
            _h *= 2;
        }
        _ws = 1 + (_w - _patch_size) / _patch_stride;
        _hs = 1 + (_h - _patch_size) / _patch_stride;
        
        {
            MIRPrecomputeStructureTensorOpt opt = {
                .w = _w,
                .h = _h,
                .patch_size = _patch_size,
                .patch_stride = _patch_stride,
                .ws = _ws
            };
            commandBuffer = [MIRPrecomputeStructureTensor encode:commandBuffer I0x:_I0xs[i] I0y:_I0ys[i] I0xx_aux:_I0xx_aux I0yy_aux:_I0yy_aux I0xy_aux:_I0xy_aux I0x_aux:_I0x_aux I0y_aux:_I0y_aux I0xx_buf:_I0xx_buf I0yy_buf:_I0yy_buf I0xy_buf:_I0xy_buf I0x_buf:_I0x_buf I0y_buf:
                             _I0y_buf opt:opt];
        }
        
        {
            int num_inner_iter = (int)floor(_grad_descent_iter / (float)_num_iter);
            MIRInvertSearchOpt opt = {
                .w = _w,
                .h = _h,
                .patch_size = _patch_size,
                .patch_stride = _patch_stride,
                .ws = _ws,
                .hs = _hs,
                .border_size = _border_size,
                .num_inner_iter = num_inner_iter
            };
            for (int iter = 0; iter < _num_iter; iter++) {
                if (iter == 0) {
                    commandBuffer = [MIRInvertSearch encode_fwd1:commandBuffer U:_Us[i] I0:_I0s[i] I1:_I1s_ext[i] S:_S opt:opt];
                    commandBuffer = [MIRInvertSearch encode_fwd2:commandBuffer U:_Us[i] I0:_I0s[i] I1:_I1s_ext[i] I0x:_I0xs[i] I0y:_I0ys[i] I0xx_buf:_I0xx_buf I0yy_buf:_I0yy_buf I0xy_buf:_I0xy_buf I0x_buf:_I0x_buf I0y_buf:_I0y_buf S:_S opt:opt];
                } else {
                    commandBuffer = [MIRInvertSearch encode_bwd1:commandBuffer I0:_I0s[i] I1:_I1s_ext[i] S:_S opt:opt];
                    commandBuffer = [MIRInvertSearch encode_bwd2:commandBuffer I0:_I0s[i] I1:_I1s_ext[i] I0x:_I0xs[i] I0y:_I0ys[i] I0xx_buf:_I0xx_buf I0yy_buf:_I0yy_buf I0xy_buf:_I0xy_buf I0x_buf:_I0x_buf I0y_buf:_I0y_buf S:_S opt:opt];
                }
            }
        }
        {
            MIRDensificationOpt opt = {
                .w = _w,
                .h = _h,
                .patch_size = _patch_size,
                .patch_stride = _patch_stride,
                .ws = _ws
            };
            commandBuffer = [MIRDensification encode:commandBuffer S:_S I0:_I0s[i] I1:_I1s[i] U:_Us[i] opt:opt];
        }
        //        if (variational_refinement_iter > 0)
        //        {
        //            std::vector<Mat> U_channels;
        //            split(u_U[i], U_channels); CV_Assert(U_channels.size() == 2);
        //            variational_refinement_processors[i]->calcUV(u_I0s[i], u_I1s[i],
        //                                                         U_channels[0], U_channels[1]);
        //            merge(U_channels, u_U[i]);
        //        }
        //
        
        {   // FIXME: DEBUG
            int w = _w ;
            int h = _h;
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
            desc.usage = MTLTextureUsageShaderWrite;
            id<MTLTexture> texture1 = [MIRMetalContext.device newTextureWithDescriptor:desc];
            id<MTLTexture> texture2 = [MIRMetalContext.device newTextureWithDescriptor:desc];
            commandBuffer = [self.class testU:commandBuffer input:_Us[i] output1:texture1 output2:texture2 w:w h:h];
        }
        
        if (i > _finest_scale) {
            commandBuffer = [self.class resizeU:commandBuffer src:_Us[i] dst:_Us[i - 1] dstW:_w*2 dstH:_h*2];
            
            {   // FIXME: DEBUG
                int w = _w*2;
                int h = _h*2;
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
                desc.usage = MTLTextureUsageShaderWrite;
                id<MTLTexture> texture1 = [MIRMetalContext.device newTextureWithDescriptor:desc];
                id<MTLTexture> texture2 = [MIRMetalContext.device newTextureWithDescriptor:desc];
                commandBuffer = [self.class testU:commandBuffer input:_Us[i - 1] output1:texture1 output2:texture2 w:w h:h];
            }
        }
    }
    
    return commandBuffer;
}


@end
