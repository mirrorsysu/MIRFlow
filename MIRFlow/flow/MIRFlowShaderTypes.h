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
    int patch_size;
    int patch_stride;
    int ws;
} MIRPrecomputeStructureTensorOpt;


typedef enum {
    MIRInvertSearch_U = 0,
    MIRInvertSearch_I0,
    MIRInvertSearch_I1,
    
    MIRInvertSearch_I0x,
    MIRInvertSearch_I0y,
    MIRInvertSearch_I0xx_buf,
    MIRInvertSearch_I0yy_buf,
    MIRInvertSearch_I0xy_buf,
    MIRInvertSearch_I0x_buf,
    MIRInvertSearch_I0y_buf,
    
    MIRInvertSearch_opt,
    MIRInvertSearch_S,
    MIRInvertSearch_smem,
} MIRInvertSearchParam;

typedef struct MIRInvertSearchOpt {
    int w;
    int h;
    int patch_size;
    int patch_stride;
    int ws;
    int hs;
    int border_size;
    int num_inner_iter;
} MIRInvertSearchOpt;

typedef enum {
    MIRDensification_S = 0,
    MIRDensification_I0,
    MIRDensification_I1,
    MIRDensification_opt,
    MIRDensification_U,
} MIRDensificationParam;

typedef struct MIRDensificationOpt {
    int w;
    int h;
    int patch_size;
    int patch_stride;
    int ws;
} MIRDensificationOpt;


#endif /* MIRFlowShaderTypes_h */
