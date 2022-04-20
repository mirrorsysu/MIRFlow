//
//  MIRFlowShaderTypes.h
//  MIRFlow
//
//  Created by Mirror on 2022/4/16.
//

#ifndef MIRFlowShaderTypes_h
#define MIRFlowShaderTypes_h

typedef enum {
    MIRSpatialGradient_vx = 0,
    MIRSpatialGradient_vy,
    MIRSpatialGradient_src,
    MIRSpatialGradient_opt
} MIRSpatialGradientParams;

typedef struct MIRSpatialGradientOpt {
    int w;
    int h;
} MIRSpatialGradientOpt;

typedef enum {
    MIRPrecomputeStructureTensor_I0x = 0,
    MIRPrecomputeStructureTensor_I0y,
    MIRPrecomputeStructureTensor_opt,
    MIRPrecomputeStructureTensor_I0xx_aux,
    MIRPrecomputeStructureTensor_I0yy_aux,
    MIRPrecomputeStructureTensor_I0xy_aux,
    MIRPrecomputeStructureTensor_I0x_aux,
    MIRPrecomputeStructureTensor_I0y_aux,
    MIRPrecomputeStructureTensor_I0xx_buf,
    MIRPrecomputeStructureTensor_I0yy_buf,
    MIRPrecomputeStructureTensor_I0xy_buf,
    MIRPrecomputeStructureTensor_I0x_buf,
    MIRPrecomputeStructureTensor_I0y_buf,
} MIRPrecomputeStructureTensorHorParam;

typedef struct MIRPrecomputeStructureTensorOpt {
    int w;
    int h;
    int patchSize;
    int patchStride;
    int ws;
} MIRPrecomputeStructureTensorOpt;


typedef enum {
    MIRInvertSearch_U = 0,
    MIRInvertSearch_I0,
    MIRInvertSearch_I1,
    MIRInvertSearch_opt,
    MIRInvertSearch_S,
    MIRInvertSearch_smem,
} MIRInvertSearchParam;

typedef struct MIRInvertSearchOpt {
    int w;
    int h;
    int patchSize;
    int patchStride;
    int ws;
    int hs;
    int borderSize;
} MIRInvertSearchOpt;


#endif /* MIRFlowShaderTypes_h */
