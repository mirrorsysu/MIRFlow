//
//  main.m
//  MIRFlow
//
//  Created by Mirror on 2022/4/16.
//

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

#import "MIRMetalContext.h"

#import "MIRFlowShaderTypes.h"

#import "MIRSpatialGradient.h"
#import "MIRPrecomputeStructureTensor.h"
#import "MIRInvertSearch.h"
#import "MIRFlow.h"

void testSpatialGradient(void);
void testPrecompute(void);
void testFwd1(void);

id<MTLTexture> loadTexture(NSString *path, int w, int h) {
    int bpr = (w * 4 + 256 - 1) / 256 * 256;
    id<MTLBuffer> buffer = [MIRMetalContext.device newBufferWithLength:bpr * h options:MTLResourceStorageModeShared];
    id<MTLTexture> input; {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
        desc.resourceOptions = MTLResourceStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead;
        input = [buffer newTextureWithDescriptor:desc offset:0 bytesPerRow:bpr];
    }
    id<MTLTexture> output; {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
        desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        output = [MIRMetalContext.device newTextureWithDescriptor:desc];
    }
    {
        NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];
        NSData *imageData = [image TIFFRepresentation];
        NSCParameterAssert(imageData);
        CGImageSourceRef imageSource = CGImageSourceCreateWithData((CFDataRef)imageData,  NULL);
        CGImageRef imageRef = CGImageSourceCreateImageAtIndex(imageSource, 0, NULL);
        
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(input.buffer.contents, w, h, 8, input.bufferBytesPerRow, colorSpace, kCGImageByteOrder32Little | kCGImageAlphaPremultipliedFirst);
        CGContextDrawImage(context, CGRectMake(0, 0, w, h), imageRef);
        
        CGColorSpaceRelease(colorSpace);
        CGContextRelease(context);
        CGImageRelease(imageRef);
    }
    
    {
        id<MTLCommandBuffer> commandBuffer = [MIRMetalContext makeCommandBuffer];
        commandBuffer = [MIRFlow grayscale:commandBuffer input:input output:output];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    return output;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Hello, World!");
        
        int w = 1000;
        int h = 1000;
        
        [MIRMetalContext startCapture];
        id<MTLTexture> I0 = loadTexture(@"/Users/guangzhuiyuandev/Desktop/1.jpg", w, h);
        id<MTLTexture> I1 = loadTexture(@"/Users/guangzhuiyuandev/Desktop/2.jpeg", w, h);
        
        MIRFlow *flow = [[MIRFlow alloc] init];
        [flow generateBufferWithWidth:w height:h];
        
        id<MTLCommandBuffer> commandBuffer = [MIRMetalContext makeCommandBuffer];
        commandBuffer = [flow encode:commandBuffer I0:I0 I1:I1];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        [MIRMetalContext stopCapture];
        
    }
    return 0;
}

void testSpatialGradient() {
    int w = 10;
    int h = 10;
    int bpr = sizeof(unsigned char) * 10;
    
    id<MTLBuffer> src = [MIRMetalContext.device newBufferWithLength:h * bpr options:MTLResourceStorageModeShared];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            *((unsigned char *)src.contents + i * bpr + j) = i * w + j;
        }
    }
    id<MTLBuffer> vx = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(short) options:MTLResourceStorageModeShared];
    id<MTLBuffer> vy = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(short) options:MTLResourceStorageModeShared];
    MIRSpatialGradientOpt opt = { .w=w, .h=h };
    
    id<MTLCommandBuffer> commandBuffer = [MIRMetalContext.commandQueue commandBuffer];
    commandBuffer = [MIRSpatialGradient encode:commandBuffer src:src vx:vx vy:vy opt:opt];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%3d ", (*((short *)vx.contents + i * w + j)));
        }
        printf("x\n");
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%3d ", (*((short *)vy.contents + i * w + j)));
        }
        printf("y\n");
    }
}

void testPrecompute() {
    int w = 16;
    int h = 12;
    int bpr = sizeof(unsigned char) * w;
    int patchSize = 8;
    int patchStride = 4;
    int ws = 1 + (w - patchSize) / patchStride;
    int hs = 1 + (h - patchSize) / patchStride;
    
    id<MTLBuffer> src = [MIRMetalContext.device newBufferWithLength:h * bpr options:MTLResourceStorageModeShared];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            *((unsigned char *)src.contents + i * bpr + j) = i + j;
        }
    }
    id<MTLBuffer> I0x = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(short) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0y = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(short) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0xx_aux = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * h * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0yy_aux = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * h * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0xy_aux = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * h * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0x_aux = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * h * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0y_aux = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * h * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0xx_buf = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * (h / patchStride) * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0yy_buf = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * (h / patchStride) * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0xy_buf = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * (h / patchStride) * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0x_buf = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * (h / patchStride) * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> I0y_buf = [MIRMetalContext.device newBufferWithLength:(w / patchStride) * (h / patchStride) * sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [MIRMetalContext.commandQueue commandBuffer];
    {
        MIRSpatialGradientOpt opt = { .w=w, .h=h };
        commandBuffer = [MIRSpatialGradient encode:commandBuffer src:src vx:I0x vy:I0y opt:opt];
    }
    {
        MIRPrecomputeStructureTensorOpt opt = { .w=w, .h=h, .ws=ws, .patchSize=patchSize, .patchStride=patchStride};
        commandBuffer = [MIRPrecomputeStructureTensor encode:commandBuffer I0x:I0x I0y:I0y I0xx_aux:I0xx_aux I0yy_aux:I0yy_aux I0xy_aux:I0xy_aux I0x_aux:I0x_aux I0y_aux:I0y_aux I0xx_buf:I0xx_buf I0yy_buf:I0yy_buf I0xy_buf:I0xy_buf I0x_buf:I0x_buf I0y_buf:I0y_buf opt:opt];
    }
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    //    for (int i = 0; i < h; i++) {
    //        for (int j = 0; j < w; j++) {
    //            printf("%3d ", (*((char *)src.contents + i * w + j)));
    //        }
    //        printf("src\n");
    //    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%3d ", (*((short *)I0x.contents + i * w + j)));
        }
        printf("x\n");
    }
    printf("============ \n");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < ws; j++) {
            printf("%.1f ", (*((float *)I0x_aux.contents + i * ws + j)));
        }
        printf("xaux\n");
    }
    printf("============ \n");
    for (int i = 0; i < hs; i++) {
        for (int j = 0; j < ws; j++) {
            printf("%.1f ", (*((float *)I0x_buf.contents + i * ws + j)));
        }
        printf("xbuf\n");
    }
    //    printf("============ \n");
    //    for (int i = 0; i < h; i++) {
    //        for (int j = 0; j < w; j++) {
    //            printf("%3d ", (*((short *)I0y.contents + i * w + j)));
    //        }
    //        printf("y\n");
    //    }
    //    printf("============ \n");
    //    for (int i = 0; i < hs; i++) {
    //        for (int j = 0; j < ws; j++) {
    //            printf("%3f ", (*((float *)I0y_buf.contents + i * ws + j)));
    //        }
    //        printf("ybuf\n");
    //    }
}


void testFwd1(void) {
    int w = 12;
    int h = 12;
    int borderSize = 16;
    int patchSize = 8;
    int patchStride = 4;
    int ws = 1 + (w - patchSize) / patchStride;
    int hs = 1 + (h - patchSize) / patchStride;
    
    id<MTLBuffer> I0 = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(short) options:MTLResourceStorageModeShared];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            *((unsigned char *)I0.contents + i * w + j) = 1;
        }
    }
    id<MTLBuffer> I1_ext = [MIRMetalContext.device newBufferWithLength:(w + 2 * borderSize) * (h + 2 * borderSize) * sizeof(short) options:MTLResourceStorageModeShared];
    for (int i = 0; i < (h + 2 * borderSize); i++) {
        for (int j = 0; j < (w + 2 * borderSize); j++) {
            *((unsigned char *)I1_ext.contents + i * (w + 2 * borderSize) + j) = 2;
        }
    }
    id<MTLBuffer> U = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(vector_float2) options:MTLResourceStorageModeShared];
    id<MTLBuffer> S = [MIRMetalContext.device newBufferWithLength:w * h * sizeof(vector_float2) options:MTLResourceStorageModeShared];
    
    [MIRMetalContext startCapture];
    
    id<MTLCommandBuffer> commandBuffer = [MIRMetalContext.commandQueue commandBuffer];
    {
        MIRInvertSearchOpt opt = { .w=w, .h=h, .ws=ws, .hs=hs, .patchSize=patchSize, .patchStride=patchStride, .borderSize=borderSize};
        commandBuffer = [MIRInvertSearch encode_fw1:commandBuffer U:U I0:I0 I1:I1_ext S:S opt:opt];
    }
    
    [commandBuffer commit];
    
    [MIRMetalContext stopCapture];
    
    [commandBuffer waitUntilCompleted];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            vector_float2 p = *((vector_float2 *)S.contents + i * w + j);
            printf("(%.0f, %.0f) ", p.x, p.y);
        }
        printf("x\n");
    }
}
