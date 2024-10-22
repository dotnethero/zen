﻿// ReSharper disable All

global using static Zen.CUDA.Interop.CudaRuntime;

using System.Runtime.InteropServices;

namespace Zen.CUDA.Interop;

internal static unsafe class CudaRuntime
{
    const string __DllName = "cudart64_12.dll";

    [DllImport(__DllName, EntryPoint = "cudaDeviceReset", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceReset();

    [DllImport(__DllName, EntryPoint = "cudaDeviceSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceSynchronize();

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceSetLimit(cudaLimit limit, nuint value);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetLimit(nuint* pValue, cudaLimit limit);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetTexture1DLinearMaxWidth", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetTexture1DLinearMaxWidth(nuint* maxWidthInElements, cudaChannelFormatDesc* fmtDesc, int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetStreamPriorityRange", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetByPCIBusId", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetByPCIBusId(int* device, byte* pciBusId);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetPCIBusId", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetPCIBusId(byte* pciBusId, int len, int device);

    [DllImport(__DllName, EntryPoint = "cudaIpcGetEventHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaIpcGetEventHandle(cudaIpcEventHandle_st* handle, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaIpcOpenEventHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaIpcOpenEventHandle(CUevent_st** @event, cudaIpcEventHandle_st handle);

    [DllImport(__DllName, EntryPoint = "cudaIpcGetMemHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaIpcGetMemHandle(cudaIpcMemHandle_st* handle, void* devPtr);

    [DllImport(__DllName, EntryPoint = "cudaIpcOpenMemHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_st handle, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaIpcCloseMemHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaIpcCloseMemHandle(void* devPtr);

    [DllImport(__DllName, EntryPoint = "cudaDeviceFlushGPUDirectRDMAWrites", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope);

    [DllImport(__DllName, EntryPoint = "cudaDeviceRegisterAsyncNotification", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceRegisterAsyncNotification(int device, delegate* unmanaged[Cdecl]<cudaAsyncNotificationInfo*, void*, cudaAsyncCallbackEntry*, void> callbackFunc, void* userData, cudaAsyncCallbackEntry** callback);

    [DllImport(__DllName, EntryPoint = "cudaDeviceUnregisterAsyncNotification", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackEntry* callback);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetSharedMemConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetSharedMemConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);

    [DllImport(__DllName, EntryPoint = "cudaThreadExit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadExit();

    [DllImport(__DllName, EntryPoint = "cudaThreadSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadSynchronize();

    [DllImport(__DllName, EntryPoint = "cudaThreadSetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadSetLimit(cudaLimit limit, nuint value);

    [DllImport(__DllName, EntryPoint = "cudaThreadGetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadGetLimit(nuint* pValue, cudaLimit limit);

    [DllImport(__DllName, EntryPoint = "cudaThreadGetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaThreadSetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaGetLastError", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetLastError();

    [DllImport(__DllName, EntryPoint = "cudaPeekAtLastError", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaPeekAtLastError();

    [DllImport(__DllName, EntryPoint = "cudaGetErrorName", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern byte* cudaGetErrorName(cudaError error);

    [DllImport(__DllName, EntryPoint = "cudaGetErrorString", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern byte* cudaGetErrorString(cudaError error);

    [DllImport(__DllName, EntryPoint = "cudaGetDeviceCount", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetDeviceCount(int* count);

    [DllImport(__DllName, EntryPoint = "cudaGetDeviceProperties_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetDefaultMemPool", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetDefaultMemPool(CUmemPoolHandle_st** memPool, int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetMemPool", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceSetMemPool(int device, CUmemPoolHandle_st* memPool);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetMemPool", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetMemPool(CUmemPoolHandle_st** memPool, int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetNvSciSyncAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetP2PAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);

    [DllImport(__DllName, EntryPoint = "cudaChooseDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaChooseDevice(int* device, cudaDeviceProp* prop);

    [DllImport(__DllName, EntryPoint = "cudaInitDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaInitDevice(int device, uint deviceFlags, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaSetDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaSetDevice(int device);

    [DllImport(__DllName, EntryPoint = "cudaGetDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetDevice(int* device);

    [DllImport(__DllName, EntryPoint = "cudaSetValidDevices", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaSetValidDevices(int* device_arr, int len);

    [DllImport(__DllName, EntryPoint = "cudaSetDeviceFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaSetDeviceFlags(uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGetDeviceFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetDeviceFlags(uint* flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamCreate(cudaStream** pStream);

    [DllImport(__DllName, EntryPoint = "cudaStreamCreateWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamCreateWithFlags(cudaStream** pStream, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamCreateWithPriority", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamCreateWithPriority(cudaStream** pStream, uint flags, int priority);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetPriority", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamGetPriority(cudaStream* hStream, int* priority);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamGetFlags(cudaStream* hStream, uint* flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetId", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamGetId(cudaStream* hStream, ulong* streamId);

    [DllImport(__DllName, EntryPoint = "cudaCtxResetPersistingL2Cache", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaCtxResetPersistingL2Cache();

    [DllImport(__DllName, EntryPoint = "cudaStreamCopyAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamCopyAttributes(cudaStream* dst, cudaStream* src);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamGetAttribute(cudaStream* hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out);

    [DllImport(__DllName, EntryPoint = "cudaStreamSetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamSetAttribute(cudaStream* hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value);

    [DllImport(__DllName, EntryPoint = "cudaStreamDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamDestroy(cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaStreamWaitEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamWaitEvent(cudaStream* stream, CUevent_st* @event, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamAddCallback", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamAddCallback(cudaStream* stream, delegate* unmanaged[Cdecl]<cudaStream*, cudaError, void*, void> callback, void* userData, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamSynchronize(cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaStreamQuery", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamQuery(cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaStreamAttachMemAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamAttachMemAsync(cudaStream* stream, void* devPtr, nuint length, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamBeginCapture", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamBeginCapture(cudaStream* stream, cudaStreamCaptureMode mode);

    [DllImport(__DllName, EntryPoint = "cudaStreamBeginCaptureToGraph", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamBeginCaptureToGraph(cudaStream* stream, CUgraph_st* graph, CUgraphNode_st** dependencies, cudaGraphEdgeData_st* dependencyData, nuint numDependencies, cudaStreamCaptureMode mode);

    [DllImport(__DllName, EntryPoint = "cudaThreadExchangeStreamCaptureMode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode);

    [DllImport(__DllName, EntryPoint = "cudaStreamEndCapture", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamEndCapture(cudaStream* stream, CUgraph_st** pGraph);

    [DllImport(__DllName, EntryPoint = "cudaStreamIsCapturing", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamIsCapturing(cudaStream* stream, cudaStreamCaptureStatus* pCaptureStatus);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetCaptureInfo_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamGetCaptureInfo_v2(cudaStream* stream, cudaStreamCaptureStatus* captureStatus_out, ulong* id_out, CUgraph_st** graph_out, CUgraphNode_st*** dependencies_out, nuint* numDependencies_out);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetCaptureInfo_v3", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamGetCaptureInfo_v3(cudaStream* stream, cudaStreamCaptureStatus* captureStatus_out, ulong* id_out, CUgraph_st** graph_out, CUgraphNode_st*** dependencies_out, cudaGraphEdgeData_st** edgeData_out, nuint* numDependencies_out);

    [DllImport(__DllName, EntryPoint = "cudaStreamUpdateCaptureDependencies", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamUpdateCaptureDependencies(cudaStream* stream, CUgraphNode_st** dependencies, nuint numDependencies, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamUpdateCaptureDependencies_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaStreamUpdateCaptureDependencies_v2(cudaStream* stream, CUgraphNode_st** dependencies, cudaGraphEdgeData_st* dependencyData, nuint numDependencies, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaEventCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventCreate(CUevent_st** @event);

    [DllImport(__DllName, EntryPoint = "cudaEventCreateWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventCreateWithFlags(CUevent_st** @event, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaEventRecord", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventRecord(CUevent_st* @event, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaEventRecordWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventRecordWithFlags(CUevent_st* @event, cudaStream* stream, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaEventQuery", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventQuery(CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaEventSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventSynchronize(CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaEventDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventDestroy(CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaEventElapsedTime", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaEventElapsedTime(float* ms, CUevent_st* start, CUevent_st* end);

    [DllImport(__DllName, EntryPoint = "cudaImportExternalMemory", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaImportExternalMemory(CUexternalMemory_st** extMem_out, cudaExternalMemoryHandleDesc* memHandleDesc);

    [DllImport(__DllName, EntryPoint = "cudaExternalMemoryGetMappedBuffer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaExternalMemoryGetMappedBuffer(void** devPtr, CUexternalMemory_st* extMem, cudaExternalMemoryBufferDesc* bufferDesc);

    [DllImport(__DllName, EntryPoint = "cudaExternalMemoryGetMappedMipmappedArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray** mipmap, CUexternalMemory_st* extMem, cudaExternalMemoryMipmappedArrayDesc* mipmapDesc);

    [DllImport(__DllName, EntryPoint = "cudaDestroyExternalMemory", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDestroyExternalMemory(CUexternalMemory_st* extMem);

    [DllImport(__DllName, EntryPoint = "cudaImportExternalSemaphore", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaImportExternalSemaphore(CUexternalSemaphore_st** extSem_out, cudaExternalSemaphoreHandleDesc* semHandleDesc);

    [DllImport(__DllName, EntryPoint = "cudaSignalExternalSemaphoresAsync_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaSignalExternalSemaphoresAsync_v2(CUexternalSemaphore_st** extSemArray, cudaExternalSemaphoreSignalParams* paramsArray, uint numExtSems, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaWaitExternalSemaphoresAsync_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaWaitExternalSemaphoresAsync_v2(CUexternalSemaphore_st** extSemArray, cudaExternalSemaphoreWaitParams* paramsArray, uint numExtSems, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaDestroyExternalSemaphore", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDestroyExternalSemaphore(CUexternalSemaphore_st* extSem);

    [DllImport(__DllName, EntryPoint = "cudaLaunchKernel", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaLaunchKernel(void* func, dim3 gridDim, dim3 blockDim, void** args, nuint sharedMem, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaLaunchKernelExC", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaLaunchKernelExC(cudaLaunchConfig_st* config, void* func, void** args);

    [DllImport(__DllName, EntryPoint = "cudaLaunchCooperativeKernel", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaLaunchCooperativeKernel(void* func, dim3 gridDim, dim3 blockDim, void** args, nuint sharedMem, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaLaunchCooperativeKernelMultiDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams* launchParamsList, uint numDevices, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaFuncSetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFuncSetCacheConfig(void* func, cudaFuncCache cacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaFuncGetAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFuncGetAttributes(cudaFuncAttributes* attr, void* func);

    [DllImport(__DllName, EntryPoint = "cudaFuncSetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFuncSetAttribute(void* func, cudaFuncAttribute attr, int value);

    [DllImport(__DllName, EntryPoint = "cudaFuncGetName", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFuncGetName(byte** name, void* func);

    [DllImport(__DllName, EntryPoint = "cudaFuncGetParamInfo", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFuncGetParamInfo(void* func, nuint paramIndex, nuint* paramOffset, nuint* paramSize);

    [DllImport(__DllName, EntryPoint = "cudaSetDoubleForDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaSetDoubleForDevice(double* d);

    [DllImport(__DllName, EntryPoint = "cudaSetDoubleForHost", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaSetDoubleForHost(double* d);

    [DllImport(__DllName, EntryPoint = "cudaLaunchHostFunc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaLaunchHostFunc(cudaStream* stream, delegate* unmanaged[Cdecl]<void*, void> fn_, void* userData);

    [DllImport(__DllName, EntryPoint = "cudaFuncSetSharedMemConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFuncSetSharedMemConfig(void* func, cudaSharedMemConfig config);

    [DllImport(__DllName, EntryPoint = "cudaOccupancyMaxActiveBlocksPerMultiprocessor", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, void* func, int blockSize, nuint dynamicSMemSize);

    [DllImport(__DllName, EntryPoint = "cudaOccupancyAvailableDynamicSMemPerBlock", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaOccupancyAvailableDynamicSMemPerBlock(nuint* dynamicSmemSize, void* func, int numBlocks, int blockSize);

    [DllImport(__DllName, EntryPoint = "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, void* func, int blockSize, nuint dynamicSMemSize, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaOccupancyMaxPotentialClusterSize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaOccupancyMaxPotentialClusterSize(int* clusterSize, void* func, cudaLaunchConfig_st* launchConfig);

    [DllImport(__DllName, EntryPoint = "cudaOccupancyMaxActiveClusters", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaOccupancyMaxActiveClusters(int* numClusters, void* func, cudaLaunchConfig_st* launchConfig);

    [DllImport(__DllName, EntryPoint = "cudaMallocManaged", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocManaged(void** devPtr, nuint size, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaMalloc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMalloc(void** devPtr, nuint size);

    [DllImport(__DllName, EntryPoint = "cudaMallocHost", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocHost(void** ptr, nuint size);

    [DllImport(__DllName, EntryPoint = "cudaMallocPitch", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocPitch(void** devPtr, nuint* pitch, nuint width, nuint height);

    [DllImport(__DllName, EntryPoint = "cudaMallocArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocArray(cudaArray** array, cudaChannelFormatDesc* desc, nuint width, nuint height, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaFree", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFree(void* devPtr);

    [DllImport(__DllName, EntryPoint = "cudaFreeHost", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFreeHost(void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaFreeArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFreeArray(cudaArray* array);

    [DllImport(__DllName, EntryPoint = "cudaFreeMipmappedArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFreeMipmappedArray(cudaMipmappedArray* mipmappedArray);

    [DllImport(__DllName, EntryPoint = "cudaHostAlloc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaHostAlloc(void** pHost, nuint size, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaHostRegister", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaHostRegister(void* ptr, nuint size, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaHostUnregister", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaHostUnregister(void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaHostGetDevicePointer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaHostGetDevicePointer(void** pDevice, void* pHost, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaHostGetFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaHostGetFlags(uint* pFlags, void* pHost);

    [DllImport(__DllName, EntryPoint = "cudaMalloc3D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);

    [DllImport(__DllName, EntryPoint = "cudaMalloc3DArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMalloc3DArray(cudaArray** array, cudaChannelFormatDesc* desc, cudaExtent extent, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaMallocMipmappedArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocMipmappedArray(cudaMipmappedArray** mipmappedArray, cudaChannelFormatDesc* desc, cudaExtent extent, uint numLevels, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGetMipmappedArrayLevel", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetMipmappedArrayLevel(cudaArray** levelArray, cudaMipmappedArray* mipmappedArray, uint level);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy3D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy3D(cudaMemcpy3DParms* p);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy3DPeer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy3DPeer(cudaMemcpy3DPeerParms* p);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy3DAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy3DAsync(cudaMemcpy3DParms* p, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy3DPeerAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy3DPeerAsync(cudaMemcpy3DPeerParms* p, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemGetInfo", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemGetInfo(nuint* free, nuint* total);

    [DllImport(__DllName, EntryPoint = "cudaArrayGetInfo", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, uint* flags, cudaArray* array);

    [DllImport(__DllName, EntryPoint = "cudaArrayGetPlane", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaArrayGetPlane(cudaArray** pPlaneArray, cudaArray* hArray, uint planeIdx);

    [DllImport(__DllName, EntryPoint = "cudaArrayGetMemoryRequirements", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray* array, int device);

    [DllImport(__DllName, EntryPoint = "cudaMipmappedArrayGetMemoryRequirements", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray* mipmap, int device);

    [DllImport(__DllName, EntryPoint = "cudaArrayGetSparseProperties", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray* array);

    [DllImport(__DllName, EntryPoint = "cudaMipmappedArrayGetSparseProperties", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray* mipmap);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy(void* dst, void* src, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyPeer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyPeer(void* dst, int dstDevice, void* src, int srcDevice, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2D(void* dst, nuint dpitch, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DToArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2DToArray(cudaArray* dst, nuint wOffset, nuint hOffset, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DFromArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2DFromArray(void* dst, nuint dpitch, cudaArray* src, nuint wOffset, nuint hOffset, nuint width, nuint height, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DArrayToArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2DArrayToArray(cudaArray* dst, nuint wOffsetDst, nuint hOffsetDst, cudaArray* src, nuint wOffsetSrc, nuint hOffsetSrc, nuint width, nuint height, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyToSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyToSymbol(void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyFromSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyFromSymbol(void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyAsync(void* dst, void* src, nuint count, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyPeerAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyPeerAsync(void* dst, int dstDevice, void* src, int srcDevice, nuint count, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2DAsync(void* dst, nuint dpitch, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DToArrayAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2DToArrayAsync(cudaArray* dst, nuint wOffset, nuint hOffset, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DFromArrayAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpy2DFromArrayAsync(void* dst, nuint dpitch, cudaArray* src, nuint wOffset, nuint hOffset, nuint width, nuint height, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyToSymbolAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyToSymbolAsync(void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyFromSymbolAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyFromSymbolAsync(void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemset", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemset(void* devPtr, int value, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemset2D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemset2D(void* devPtr, nuint pitch, int value, nuint width, nuint height);

    [DllImport(__DllName, EntryPoint = "cudaMemset3D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);

    [DllImport(__DllName, EntryPoint = "cudaMemsetAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemsetAsync(void* devPtr, int value, nuint count, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemset2DAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemset2DAsync(void* devPtr, nuint pitch, int value, nuint width, nuint height, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemset3DAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGetSymbolAddress", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetSymbolAddress(void** devPtr, void* symbol);

    [DllImport(__DllName, EntryPoint = "cudaGetSymbolSize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetSymbolSize(nuint* size, void* symbol);

    [DllImport(__DllName, EntryPoint = "cudaMemPrefetchAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPrefetchAsync(void* devPtr, nuint count, int dstDevice, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemPrefetchAsync_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPrefetchAsync_v2(void* devPtr, nuint count, cudaMemLocation location, uint flags, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemAdvise", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemAdvise(void* devPtr, nuint count, cudaMemoryAdvise advice, int device);

    [DllImport(__DllName, EntryPoint = "cudaMemAdvise_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemAdvise_v2(void* devPtr, nuint count, cudaMemoryAdvise advice, cudaMemLocation location);

    [DllImport(__DllName, EntryPoint = "cudaMemRangeGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemRangeGetAttribute(void* data, nuint dataSize, cudaMemRangeAttribute attribute, void* devPtr, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemRangeGetAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemRangeGetAttributes(void** data, nuint* dataSizes, cudaMemRangeAttribute* attributes, nuint numAttributes, void* devPtr, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyToArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyToArray(cudaArray* dst, nuint wOffset, nuint hOffset, void* src, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyFromArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyFromArray(void* dst, cudaArray* src, nuint wOffset, nuint hOffset, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyArrayToArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyArrayToArray(cudaArray* dst, nuint wOffsetDst, nuint hOffsetDst, cudaArray* src, nuint wOffsetSrc, nuint hOffsetSrc, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyToArrayAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyToArrayAsync(cudaArray* dst, nuint wOffset, nuint hOffset, void* src, nuint count, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyFromArrayAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemcpyFromArrayAsync(void* dst, cudaArray* src, nuint wOffset, nuint hOffset, nuint count, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMallocAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocAsync(void** devPtr, nuint size, cudaStream* hStream);

    [DllImport(__DllName, EntryPoint = "cudaFreeAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaFreeAsync(void* devPtr, cudaStream* hStream);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolTrimTo", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolTrimTo(CUmemPoolHandle_st* memPool, nuint minBytesToKeep);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolSetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolSetAttribute(CUmemPoolHandle_st* memPool, cudaMemPoolAttr attr, void* value);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolGetAttribute(CUmemPoolHandle_st* memPool, cudaMemPoolAttr attr, void* value);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolSetAccess", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolSetAccess(CUmemPoolHandle_st* memPool, cudaMemAccessDesc* descList, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolGetAccess", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolGetAccess(cudaMemAccessFlags* flags, CUmemPoolHandle_st* memPool, cudaMemLocation* location);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolCreate(CUmemPoolHandle_st** memPool, cudaMemPoolProps* poolProps);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolDestroy(CUmemPoolHandle_st* memPool);

    [DllImport(__DllName, EntryPoint = "cudaMallocFromPoolAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMallocFromPoolAsync(void** ptr, nuint size, CUmemPoolHandle_st* memPool, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolExportToShareableHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolExportToShareableHandle(void* shareableHandle, CUmemPoolHandle_st* memPool, cudaMemAllocationHandleType handleType, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolImportFromShareableHandle", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolImportFromShareableHandle(CUmemPoolHandle_st** memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolExportPointer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaMemPoolImportPointer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaMemPoolImportPointer(void** ptr, CUmemPoolHandle_st* memPool, cudaMemPoolPtrExportData* exportData);

    [DllImport(__DllName, EntryPoint = "cudaPointerGetAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaPointerGetAttributes(cudaPointerAttributes* attributes, void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaDeviceCanAccessPeer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);

    [DllImport(__DllName, EntryPoint = "cudaDeviceEnablePeerAccess", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceEnablePeerAccess(int peerDevice, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaDeviceDisablePeerAccess", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceDisablePeerAccess(int peerDevice);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsUnregisterResource", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsUnregisterResource(cudaGraphicsResource* resource);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsResourceSetMapFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsResourceSetMapFlags(cudaGraphicsResource* resource, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsMapResources", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsMapResources(int count, cudaGraphicsResource** resources, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsUnmapResources", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsUnmapResources(int count, cudaGraphicsResource** resources, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsResourceGetMappedPointer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsResourceGetMappedPointer(void** devPtr, nuint* size, cudaGraphicsResource* resource);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsSubResourceGetMappedArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsSubResourceGetMappedArray(cudaArray** array, cudaGraphicsResource* resource, uint arrayIndex, uint mipLevel);

    [DllImport(__DllName, EntryPoint = "cudaGraphicsResourceGetMappedMipmappedArray", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray** mipmappedArray, cudaGraphicsResource* resource);

    [DllImport(__DllName, EntryPoint = "cudaGetChannelDesc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray* array);

    [DllImport(__DllName, EntryPoint = "cudaCreateChannelDesc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f);

    [DllImport(__DllName, EntryPoint = "cudaCreateTextureObject", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaCreateTextureObject(ulong* pTexObject, cudaResourceDesc* pResDesc, cudaTextureDesc* pTexDesc, cudaResourceViewDesc* pResViewDesc);

    [DllImport(__DllName, EntryPoint = "cudaDestroyTextureObject", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDestroyTextureObject(ulong texObject);

    [DllImport(__DllName, EntryPoint = "cudaGetTextureObjectResourceDesc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, ulong texObject);

    [DllImport(__DllName, EntryPoint = "cudaGetTextureObjectTextureDesc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, ulong texObject);

    [DllImport(__DllName, EntryPoint = "cudaGetTextureObjectResourceViewDesc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, ulong texObject);

    [DllImport(__DllName, EntryPoint = "cudaCreateSurfaceObject", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaCreateSurfaceObject(ulong* pSurfObject, cudaResourceDesc* pResDesc);

    [DllImport(__DllName, EntryPoint = "cudaDestroySurfaceObject", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDestroySurfaceObject(ulong surfObject);

    [DllImport(__DllName, EntryPoint = "cudaGetSurfaceObjectResourceDesc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, ulong surfObject);

    [DllImport(__DllName, EntryPoint = "cudaDriverGetVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDriverGetVersion(int* driverVersion);

    [DllImport(__DllName, EntryPoint = "cudaRuntimeGetVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaRuntimeGetVersion(int* runtimeVersion);

    [DllImport(__DllName, EntryPoint = "cudaGraphCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphCreate(CUgraph_st** pGraph, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddKernelNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddKernelNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaKernelNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphKernelNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphKernelNodeGetParams(CUgraphNode_st* node, cudaKernelNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphKernelNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphKernelNodeSetParams(CUgraphNode_st* node, cudaKernelNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphKernelNodeCopyAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphKernelNodeCopyAttributes(CUgraphNode_st* hSrc, CUgraphNode_st* hDst);

    [DllImport(__DllName, EntryPoint = "cudaGraphKernelNodeGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphKernelNodeGetAttribute(CUgraphNode_st* hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out);

    [DllImport(__DllName, EntryPoint = "cudaGraphKernelNodeSetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphKernelNodeSetAttribute(CUgraphNode_st* hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemcpyNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemcpyNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaMemcpy3DParms* pCopyParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemcpyNodeToSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemcpyNodeToSymbol(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemcpyNodeFromSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemcpyNodeFromSymbol(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemcpyNode1D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemcpyNode1D(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, void* dst, void* src, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemcpyNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemcpyNodeGetParams(CUgraphNode_st* node, cudaMemcpy3DParms* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemcpyNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemcpyNodeSetParams(CUgraphNode_st* node, cudaMemcpy3DParms* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemcpyNodeSetParamsToSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemcpyNodeSetParamsToSymbol(CUgraphNode_st* node, void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemcpyNodeSetParamsFromSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemcpyNodeSetParamsFromSymbol(CUgraphNode_st* node, void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemcpyNodeSetParams1D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemcpyNodeSetParams1D(CUgraphNode_st* node, void* dst, void* src, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemsetNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemsetNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaMemsetParams* pMemsetParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemsetNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemsetNodeGetParams(CUgraphNode_st* node, cudaMemsetParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemsetNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemsetNodeSetParams(CUgraphNode_st* node, cudaMemsetParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddHostNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddHostNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaHostNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphHostNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphHostNodeGetParams(CUgraphNode_st* node, cudaHostNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphHostNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphHostNodeSetParams(CUgraphNode_st* node, cudaHostNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddChildGraphNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddChildGraphNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, CUgraph_st* childGraph);

    [DllImport(__DllName, EntryPoint = "cudaGraphChildGraphNodeGetGraph", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphChildGraphNodeGetGraph(CUgraphNode_st* node, CUgraph_st** pGraph);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddEmptyNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddEmptyNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddEventRecordNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddEventRecordNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaGraphEventRecordNodeGetEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphEventRecordNodeGetEvent(CUgraphNode_st* node, CUevent_st** event_out);

    [DllImport(__DllName, EntryPoint = "cudaGraphEventRecordNodeSetEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphEventRecordNodeSetEvent(CUgraphNode_st* node, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddEventWaitNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddEventWaitNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaGraphEventWaitNodeGetEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphEventWaitNodeGetEvent(CUgraphNode_st* node, CUevent_st** event_out);

    [DllImport(__DllName, EntryPoint = "cudaGraphEventWaitNodeSetEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphEventWaitNodeSetEvent(CUgraphNode_st* node, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddExternalSemaphoresSignalNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddExternalSemaphoresSignalNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaExternalSemaphoreSignalNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExternalSemaphoresSignalNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode_st* hNode, cudaExternalSemaphoreSignalNodeParams* params_out);

    [DllImport(__DllName, EntryPoint = "cudaGraphExternalSemaphoresSignalNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode_st* hNode, cudaExternalSemaphoreSignalNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddExternalSemaphoresWaitNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddExternalSemaphoresWaitNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaExternalSemaphoreWaitNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExternalSemaphoresWaitNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode_st* hNode, cudaExternalSemaphoreWaitNodeParams* params_out);

    [DllImport(__DllName, EntryPoint = "cudaGraphExternalSemaphoresWaitNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode_st* hNode, cudaExternalSemaphoreWaitNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemAllocNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemAllocNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaMemAllocNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemAllocNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemAllocNodeGetParams(CUgraphNode_st* node, cudaMemAllocNodeParams* params_out);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddMemFreeNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddMemFreeNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, void* dptr);

    [DllImport(__DllName, EntryPoint = "cudaGraphMemFreeNodeGetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphMemFreeNodeGetParams(CUgraphNode_st* node, void* dptr_out);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGraphMemTrim", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGraphMemTrim(int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetGraphMemAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetGraphMemAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);

    [DllImport(__DllName, EntryPoint = "cudaGraphClone", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphClone(CUgraph_st** pGraphClone, CUgraph_st* originalGraph);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeFindInClone", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeFindInClone(CUgraphNode_st** pNode, CUgraphNode_st* originalNode, CUgraph_st* clonedGraph);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeGetType", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeGetType(CUgraphNode_st* node, cudaGraphNodeType* pType);

    [DllImport(__DllName, EntryPoint = "cudaGraphGetNodes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphGetNodes(CUgraph_st* graph, CUgraphNode_st** nodes, nuint* numNodes);

    [DllImport(__DllName, EntryPoint = "cudaGraphGetRootNodes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphGetRootNodes(CUgraph_st* graph, CUgraphNode_st** pRootNodes, nuint* pNumRootNodes);

    [DllImport(__DllName, EntryPoint = "cudaGraphGetEdges", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphGetEdges(CUgraph_st* graph, CUgraphNode_st** from, CUgraphNode_st** to, nuint* numEdges);

    [DllImport(__DllName, EntryPoint = "cudaGraphGetEdges_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphGetEdges_v2(CUgraph_st* graph, CUgraphNode_st** from, CUgraphNode_st** to, cudaGraphEdgeData_st* edgeData, nuint* numEdges);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeGetDependencies", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeGetDependencies(CUgraphNode_st* node, CUgraphNode_st** pDependencies, nuint* pNumDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeGetDependencies_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeGetDependencies_v2(CUgraphNode_st* node, CUgraphNode_st** pDependencies, cudaGraphEdgeData_st* edgeData, nuint* pNumDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeGetDependentNodes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeGetDependentNodes(CUgraphNode_st* node, CUgraphNode_st** pDependentNodes, nuint* pNumDependentNodes);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeGetDependentNodes_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeGetDependentNodes_v2(CUgraphNode_st* node, CUgraphNode_st** pDependentNodes, cudaGraphEdgeData_st* edgeData, nuint* pNumDependentNodes);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddDependencies", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddDependencies(CUgraph_st* graph, CUgraphNode_st** from, CUgraphNode_st** to, nuint numDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddDependencies_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddDependencies_v2(CUgraph_st* graph, CUgraphNode_st** from, CUgraphNode_st** to, cudaGraphEdgeData_st* edgeData, nuint numDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphRemoveDependencies", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphRemoveDependencies(CUgraph_st* graph, CUgraphNode_st** from, CUgraphNode_st** to, nuint numDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphRemoveDependencies_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphRemoveDependencies_v2(CUgraph_st* graph, CUgraphNode_st** from, CUgraphNode_st** to, cudaGraphEdgeData_st* edgeData, nuint numDependencies);

    [DllImport(__DllName, EntryPoint = "cudaGraphDestroyNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphDestroyNode(CUgraphNode_st* node);

    [DllImport(__DllName, EntryPoint = "cudaGraphInstantiate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphInstantiate(CUgraphExec_st** pGraphExec, CUgraph_st* graph, ulong flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphInstantiateWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphInstantiateWithFlags(CUgraphExec_st** pGraphExec, CUgraph_st* graph, ulong flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphInstantiateWithParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphInstantiateWithParams(CUgraphExec_st** pGraphExec, CUgraph_st* graph, cudaGraphInstantiateParams_st* instantiateParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecGetFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecGetFlags(CUgraphExec_st* graphExec, ulong* flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecKernelNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecKernelNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, cudaKernelNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecMemcpyNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecMemcpyNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, cudaMemcpy3DParms* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecMemcpyNodeSetParamsToSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecMemcpyNodeSetParamsToSymbol(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecMemcpyNodeSetParamsFromSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecMemcpyNodeSetParamsFromSymbol(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecMemcpyNodeSetParams1D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecMemcpyNodeSetParams1D(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, void* dst, void* src, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecMemsetNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecMemsetNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, cudaMemsetParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecHostNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecHostNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, cudaHostNodeParams* pNodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecChildGraphNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecChildGraphNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* node, CUgraph_st* childGraph);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecEventRecordNodeSetEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecEventRecordNodeSetEvent(CUgraphExec_st* hGraphExec, CUgraphNode_st* hNode, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecEventWaitNodeSetEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecEventWaitNodeSetEvent(CUgraphExec_st* hGraphExec, CUgraphNode_st* hNode, CUevent_st* @event);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecExternalSemaphoresSignalNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* hNode, cudaExternalSemaphoreSignalNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecExternalSemaphoresWaitNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec_st* hGraphExec, CUgraphNode_st* hNode, cudaExternalSemaphoreWaitNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeSetEnabled", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeSetEnabled(CUgraphExec_st* hGraphExec, CUgraphNode_st* hNode, uint isEnabled);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeGetEnabled", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeGetEnabled(CUgraphExec_st* hGraphExec, CUgraphNode_st* hNode, uint* isEnabled);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecUpdate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecUpdate(CUgraphExec_st* hGraphExec, CUgraph_st* hGraph, cudaGraphExecUpdateResultInfo_st* resultInfo);

    [DllImport(__DllName, EntryPoint = "cudaGraphUpload", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphUpload(CUgraphExec_st* graphExec, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGraphLaunch", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphLaunch(CUgraphExec_st* graphExec, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecDestroy(CUgraphExec_st* graphExec);

    [DllImport(__DllName, EntryPoint = "cudaGraphDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphDestroy(CUgraph_st* graph);

    [DllImport(__DllName, EntryPoint = "cudaGraphDebugDotPrint", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphDebugDotPrint(CUgraph_st* graph, byte* path, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaUserObjectCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaUserObjectCreate(CUuserObject_st** object_out, void* ptr, delegate* unmanaged[Cdecl]<void*, void> destroy, uint initialRefcount, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaUserObjectRetain", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaUserObjectRetain(CUuserObject_st* @object, uint count);

    [DllImport(__DllName, EntryPoint = "cudaUserObjectRelease", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaUserObjectRelease(CUuserObject_st* @object, uint count);

    [DllImport(__DllName, EntryPoint = "cudaGraphRetainUserObject", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphRetainUserObject(CUgraph_st* graph, CUuserObject_st* @object, uint count, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphReleaseUserObject", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphReleaseUserObject(CUgraph_st* graph, CUuserObject_st* @object, uint count);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddNode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddNode(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, nuint numDependencies, cudaGraphNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphAddNode_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphAddNode_v2(CUgraphNode_st** pGraphNode, CUgraph_st* graph, CUgraphNode_st** pDependencies, cudaGraphEdgeData_st* dependencyData, nuint numDependencies, cudaGraphNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphNodeSetParams(CUgraphNode_st* node, cudaGraphNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecNodeSetParams", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphExecNodeSetParams(CUgraphExec_st* graphExec, CUgraphNode_st* node, cudaGraphNodeParams* nodeParams);

    [DllImport(__DllName, EntryPoint = "cudaGraphConditionalHandleCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGraphConditionalHandleCreate(ulong* pHandle_out, CUgraph_st* graph, uint defaultLaunchValue, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGetDriverEntryPoint", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetDriverEntryPoint(byte* symbol, void** funcPtr, ulong flags, cudaDriverEntryPointQueryResult* driverStatus);

    [DllImport(__DllName, EntryPoint = "cudaGetDriverEntryPointByVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetDriverEntryPointByVersion(byte* symbol, void** funcPtr, uint cudaVersion, ulong flags, cudaDriverEntryPointQueryResult* driverStatus);

    [DllImport(__DllName, EntryPoint = "cudaGetExportTable", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetExportTable(void** ppExportTable, CUuuid_st* pExportTableId);

    [DllImport(__DllName, EntryPoint = "cudaGetFuncBySymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetFuncBySymbol(CUfunc_st** functionPtr, void* symbolPtr);

    [DllImport(__DllName, EntryPoint = "cudaGetKernel", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaError cudaGetKernel(CUkern_st** kernelPtr, void* entryFuncAddr);


}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct _iobuf
{
    public void* _Placeholder;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct dim3
{
    public uint x;
    public uint y;
    public uint z;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaChannelFormatDesc
{
    public int x;
    public int y;
    public int z;
    public int w;
    public cudaChannelFormatKind f;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaArray
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMipmappedArray
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaArraySparseProperties
{
    public cudaArraySparseProperties__bindgen_ty_1 tileExtent;
    public uint miptailFirstLevel;
    public ulong miptailSize;
    public uint flags;
    public fixed uint reserved[4];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaArraySparseProperties__bindgen_ty_1
{
    public uint width;
    public uint height;
    public uint depth;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaArrayMemoryRequirements
{
    public nuint size;
    public nuint alignment;
    public fixed uint reserved[4];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaPitchedPtr
{
    public void* ptr;
    public nuint pitch;
    public nuint xsize;
    public nuint ysize;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExtent
{
    public nuint width;
    public nuint height;
    public nuint depth;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaPos
{
    public nuint x;
    public nuint y;
    public nuint z;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemcpy3DParms
{
    public cudaArray* srcArray;
    public cudaPos srcPos;
    public cudaPitchedPtr srcPtr;
    public cudaArray* dstArray;
    public cudaPos dstPos;
    public cudaPitchedPtr dstPtr;
    public cudaExtent extent;
    public cudaMemcpyKind kind;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemcpyNodeParams
{
    public int flags;
    public fixed int reserved[3];
    public cudaMemcpy3DParms copyParams;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemcpy3DPeerParms
{
    public cudaArray* srcArray;
    public cudaPos srcPos;
    public cudaPitchedPtr srcPtr;
    public int srcDevice;
    public cudaArray* dstArray;
    public cudaPos dstPos;
    public cudaPitchedPtr dstPtr;
    public int dstDevice;
    public cudaExtent extent;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemsetParams
{
    public void* dst;
    public nuint pitch;
    public uint value;
    public uint elementSize;
    public nuint width;
    public nuint height;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemsetParamsV2
{
    public void* dst;
    public nuint pitch;
    public uint value;
    public uint elementSize;
    public nuint width;
    public nuint height;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaAccessPolicyWindow
{
    public void* base_ptr;
    public nuint num_bytes;
    public float hitRatio;
    public cudaAccessProperty hitProp;
    public cudaAccessProperty missProp;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaHostNodeParams
{
    public delegate* unmanaged[Cdecl]<void*, void> fn_;
    public void* userData;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaHostNodeParamsV2
{
    public delegate* unmanaged[Cdecl]<void*, void> fn_;
    public void* userData;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaGraphicsResource
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaResourceDesc
{
    public cudaResourceType resType;
    public cudaResourceDesc__bindgen_ty_1 res;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaResourceDesc__bindgen_ty_1
{
    [FieldOffset(0)]
    public cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 array;
    [FieldOffset(0)]
    public cudaResourceDesc__bindgen_ty_1__bindgen_ty_2 mipmap;
    [FieldOffset(0)]
    public cudaResourceDesc__bindgen_ty_1__bindgen_ty_3 linear;
    [FieldOffset(0)]
    public cudaResourceDesc__bindgen_ty_1__bindgen_ty_4 pitch2D;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_1
{
    public cudaArray* array;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_2
{
    public cudaMipmappedArray* mipmap;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_3
{
    public void* devPtr;
    public cudaChannelFormatDesc desc;
    public nuint sizeInBytes;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_4
{
    public void* devPtr;
    public cudaChannelFormatDesc desc;
    public nuint width;
    public nuint height;
    public nuint pitchInBytes;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaResourceViewDesc
{
    public cudaResourceViewFormat format;
    public nuint width;
    public nuint height;
    public nuint depth;
    public uint firstMipmapLevel;
    public uint lastMipmapLevel;
    public uint firstLayer;
    public uint lastLayer;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaPointerAttributes
{
    public cudaMemoryType type_;
    public int device;
    public void* devicePointer;
    public void* hostPointer;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaFuncAttributes
{
    public nuint sharedSizeBytes;
    public nuint constSizeBytes;
    public nuint localSizeBytes;
    public int maxThreadsPerBlock;
    public int numRegs;
    public int ptxVersion;
    public int binaryVersion;
    public int cacheModeCA;
    public int maxDynamicSharedSizeBytes;
    public int preferredShmemCarveout;
    public int clusterDimMustBeSet;
    public int requiredClusterWidth;
    public int requiredClusterHeight;
    public int requiredClusterDepth;
    public int clusterSchedulingPolicyPreference;
    public int nonPortableClusterSizeAllowed;
    public fixed int reserved[16];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemLocation
{
    public cudaMemLocationType type_;
    public int id;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemAccessDesc
{
    public cudaMemLocation location;
    public cudaMemAccessFlags flags;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemPoolProps
{
    public cudaMemAllocationType allocType;
    public cudaMemAllocationHandleType handleTypes;
    public cudaMemLocation location;
    public void* win32SecurityAttributes;
    public nuint maxSize;
    public fixed byte reserved[56];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemPoolPtrExportData
{
    public fixed byte reserved[64];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemAllocNodeParams
{
    public cudaMemPoolProps poolProps;
    public cudaMemAccessDesc* accessDescs;
    public nuint accessDescCount;
    public nuint bytesize;
    public void* dptr;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemAllocNodeParamsV2
{
    public cudaMemPoolProps poolProps;
    public cudaMemAccessDesc* accessDescs;
    public nuint accessDescCount;
    public nuint bytesize;
    public void* dptr;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaMemFreeNodeParams
{
    public void* dptr;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUuuid_st
{
    public fixed byte bytes[16];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaDeviceProp
{
    public fixed byte name[256];
    public CUuuid_st uuid;
    public fixed byte luid[8];
    public uint luidDeviceNodeMask;
    public nuint totalGlobalMem;
    public nuint sharedMemPerBlock;
    public int regsPerBlock;
    public int warpSize;
    public nuint memPitch;
    public int maxThreadsPerBlock;
    public fixed int maxThreadsDim[3];
    public fixed int maxGridSize[3];
    public int clockRate;
    public nuint totalConstMem;
    public int major;
    public int minor;
    public nuint textureAlignment;
    public nuint texturePitchAlignment;
    public int deviceOverlap;
    public int multiProcessorCount;
    public int kernelExecTimeoutEnabled;
    public int integrated;
    public int canMapHostMemory;
    public int computeMode;
    public int maxTexture1D;
    public int maxTexture1DMipmap;
    public int maxTexture1DLinear;
    public fixed int maxTexture2D[2];
    public fixed int maxTexture2DMipmap[2];
    public fixed int maxTexture2DLinear[3];
    public fixed int maxTexture2DGather[2];
    public fixed int maxTexture3D[3];
    public fixed int maxTexture3DAlt[3];
    public int maxTextureCubemap;
    public fixed int maxTexture1DLayered[2];
    public fixed int maxTexture2DLayered[3];
    public fixed int maxTextureCubemapLayered[2];
    public int maxSurface1D;
    public fixed int maxSurface2D[2];
    public fixed int maxSurface3D[3];
    public fixed int maxSurface1DLayered[2];
    public fixed int maxSurface2DLayered[3];
    public int maxSurfaceCubemap;
    public fixed int maxSurfaceCubemapLayered[2];
    public nuint surfaceAlignment;
    public int concurrentKernels;
    public int ECCEnabled;
    public int pciBusID;
    public int pciDeviceID;
    public int pciDomainID;
    public int tccDriver;
    public int asyncEngineCount;
    public int unifiedAddressing;
    public int memoryClockRate;
    public int memoryBusWidth;
    public int l2CacheSize;
    public int persistingL2CacheMaxSize;
    public int maxThreadsPerMultiProcessor;
    public int streamPrioritiesSupported;
    public int globalL1CacheSupported;
    public int localL1CacheSupported;
    public nuint sharedMemPerMultiprocessor;
    public int regsPerMultiprocessor;
    public int managedMemory;
    public int isMultiGpuBoard;
    public int multiGpuBoardGroupID;
    public int hostNativeAtomicSupported;
    public int singleToDoublePrecisionPerfRatio;
    public int pageableMemoryAccess;
    public int concurrentManagedAccess;
    public int computePreemptionSupported;
    public int canUseHostPointerForRegisteredMem;
    public int cooperativeLaunch;
    public int cooperativeMultiDeviceLaunch;
    public nuint sharedMemPerBlockOptin;
    public int pageableMemoryAccessUsesHostPageTables;
    public int directManagedMemAccessFromHost;
    public int maxBlocksPerMultiProcessor;
    public int accessPolicyMaxWindowSize;
    public nuint reservedSharedMemPerBlock;
    public int hostRegisterSupported;
    public int sparseCudaArraySupported;
    public int hostRegisterReadOnlySupported;
    public int timelineSemaphoreInteropSupported;
    public int memoryPoolsSupported;
    public int gpuDirectRDMASupported;
    public uint gpuDirectRDMAFlushWritesOptions;
    public int gpuDirectRDMAWritesOrdering;
    public uint memoryPoolSupportedHandleTypes;
    public int deferredMappingCudaArraySupported;
    public int ipcEventSupported;
    public int clusterLaunch;
    public int unifiedFunctionPointers;
    public fixed int reserved2[2];
    public fixed int reserved1[1];
    public fixed int reserved[60];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaIpcEventHandle_st
{
    public fixed byte reserved[64];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaIpcMemHandle_st
{
    public fixed byte reserved[64];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalMemoryHandleDesc
{
    public cudaExternalMemoryHandleType type_;
    public cudaExternalMemoryHandleDesc__bindgen_ty_1 handle;
    public ulong size;
    public uint flags;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaExternalMemoryHandleDesc__bindgen_ty_1
{
    [FieldOffset(0)]
    public int fd;
    [FieldOffset(0)]
    public cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1 win32;
    [FieldOffset(0)]
    public void* nvSciBufObject;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1
{
    public void* handle;
    public void* name;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalMemoryBufferDesc
{
    public ulong offset;
    public ulong size;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalMemoryMipmappedArrayDesc
{
    public ulong offset;
    public cudaChannelFormatDesc formatDesc;
    public cudaExtent extent;
    public uint flags;
    public uint numLevels;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreHandleDesc
{
    public cudaExternalSemaphoreHandleType type_;
    public cudaExternalSemaphoreHandleDesc__bindgen_ty_1 handle;
    public uint flags;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaExternalSemaphoreHandleDesc__bindgen_ty_1
{
    [FieldOffset(0)]
    public int fd;
    [FieldOffset(0)]
    public cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1 win32;
    [FieldOffset(0)]
    public void* nvSciSyncObj;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1
{
    public void* handle;
    public void* name;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreSignalParams
{
    public cudaExternalSemaphoreSignalParams__bindgen_ty_1 @params;
    public uint flags;
    public fixed uint reserved[16];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreSignalParams__bindgen_ty_1
{
    public cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_1 fence;
    public cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2 nvSciSync;
    public cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_3 keyedMutex;
    public fixed uint reserved[12];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_1
{
    public ulong value;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2
{
    [FieldOffset(0)]
    public void* fence;
    [FieldOffset(0)]
    public ulong reserved;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_3
{
    public ulong key;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreWaitParams
{
    public cudaExternalSemaphoreWaitParams__bindgen_ty_1 @params;
    public uint flags;
    public fixed uint reserved[16];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreWaitParams__bindgen_ty_1
{
    public cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_1 fence;
    public cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2 nvSciSync;
    public cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_3 keyedMutex;
    public fixed uint reserved[10];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_1
{
    public ulong value;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2
{
    [FieldOffset(0)]
    public void* fence;
    [FieldOffset(0)]
    public ulong reserved;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_3
{
    public ulong key;
    public uint timeoutMs;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaStream
{
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUevent_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUexternalMemory_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUexternalSemaphore_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUgraph_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUgraphNode_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUuserObject_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUfunc_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUkern_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUmemPoolHandle_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchParams
{
    public void* func;
    public dim3 gridDim;
    public dim3 blockDim;
    public void** args;
    public nuint sharedMem;
    public cudaStream* stream;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaKernelNodeParams
{
    public void* func;
    public dim3 gridDim;
    public dim3 blockDim;
    public uint sharedMemBytes;
    public void** kernelParams;
    public void** extra;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaKernelNodeParamsV2
{
    public void* func;
    public dim3 gridDim;
    public dim3 blockDim;
    public uint sharedMemBytes;
    public void** kernelParams;
    public void** extra;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreSignalNodeParams
{
    public CUexternalSemaphore_st** extSemArray;
    public cudaExternalSemaphoreSignalParams* paramsArray;
    public uint numExtSems;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreSignalNodeParamsV2
{
    public CUexternalSemaphore_st** extSemArray;
    public cudaExternalSemaphoreSignalParams* paramsArray;
    public uint numExtSems;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreWaitNodeParams
{
    public CUexternalSemaphore_st** extSemArray;
    public cudaExternalSemaphoreWaitParams* paramsArray;
    public uint numExtSems;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaExternalSemaphoreWaitNodeParamsV2
{
    public CUexternalSemaphore_st** extSemArray;
    public cudaExternalSemaphoreWaitParams* paramsArray;
    public uint numExtSems;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaConditionalNodeParams
{
    public ulong handle;
    public cudaGraphConditionalNodeType type_;
    public uint size;
    public CUgraph_st** phGraph_out;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaChildGraphNodeParams
{
    public CUgraph_st* graph;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaEventRecordNodeParams
{
    public CUevent_st* @event;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaEventWaitNodeParams
{
    public CUevent_st* @event;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaGraphNodeParams
{
    public cudaGraphNodeType type_;
    public fixed int reserved0[3];
    public cudaGraphNodeParams__bindgen_ty_1 __bindgen_anon_1;
    public long reserved2;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaGraphNodeParams__bindgen_ty_1
{
    [FieldOffset(0)]
    public fixed long reserved1[29];
    [FieldOffset(0)]
    public cudaKernelNodeParamsV2 kernel;
    [FieldOffset(0)]
    public cudaMemcpyNodeParams memcpy;
    [FieldOffset(0)]
    public cudaMemsetParamsV2 memset;
    [FieldOffset(0)]
    public cudaHostNodeParamsV2 host;
    [FieldOffset(0)]
    public cudaChildGraphNodeParams graph;
    [FieldOffset(0)]
    public cudaEventWaitNodeParams eventWait;
    [FieldOffset(0)]
    public cudaEventRecordNodeParams eventRecord;
    [FieldOffset(0)]
    public cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal;
    [FieldOffset(0)]
    public cudaExternalSemaphoreWaitNodeParamsV2 extSemWait;
    [FieldOffset(0)]
    public cudaMemAllocNodeParamsV2 alloc;
    [FieldOffset(0)]
    public cudaMemFreeNodeParams free;
    [FieldOffset(0)]
    public cudaConditionalNodeParams conditional;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaGraphEdgeData_st
{
    public byte from_port;
    public byte to_port;
    public byte type_;
    public fixed byte reserved[5];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUgraphExec_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaGraphInstantiateParams_st
{
    public ulong flags;
    public cudaStream* uploadStream;
    public CUgraphNode_st* errNode_out;
    public cudaGraphInstantiateResult result_out;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaGraphExecUpdateResultInfo_st
{
    public cudaGraphExecUpdateResult result;
    public CUgraphNode_st* errorNode;
    public CUgraphNode_st* errorFromNode;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct CUgraphDeviceUpdatableNode_st
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchMemSyncDomainMap_st
{
    public byte default_;
    public byte remote;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaLaunchAttributeValue
{
    [FieldOffset(0)]
    public fixed byte pad[64];
    [FieldOffset(0)]
    public cudaAccessPolicyWindow accessPolicyWindow;
    [FieldOffset(0)]
    public int cooperative;
    [FieldOffset(0)]
    public cudaSynchronizationPolicy syncPolicy;
    [FieldOffset(0)]
    public cudaLaunchAttributeValue__bindgen_ty_1 clusterDim;
    [FieldOffset(0)]
    public cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference;
    [FieldOffset(0)]
    public int programmaticStreamSerializationAllowed;
    [FieldOffset(0)]
    public cudaLaunchAttributeValue__bindgen_ty_2 programmaticEvent;
    [FieldOffset(0)]
    public int priority;
    [FieldOffset(0)]
    public cudaLaunchMemSyncDomainMap_st memSyncDomainMap;
    [FieldOffset(0)]
    public cudaLaunchMemSyncDomain memSyncDomain;
    [FieldOffset(0)]
    public cudaLaunchAttributeValue__bindgen_ty_3 launchCompletionEvent;
    [FieldOffset(0)]
    public cudaLaunchAttributeValue__bindgen_ty_4 deviceUpdatableKernelNode;
    [FieldOffset(0)]
    public uint sharedMemCarveout;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchAttributeValue__bindgen_ty_1
{
    public uint x;
    public uint y;
    public uint z;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchAttributeValue__bindgen_ty_2
{
    public CUevent_st* @event;
    public int flags;
    public int triggerAtBlockStart;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchAttributeValue__bindgen_ty_3
{
    public CUevent_st* @event;
    public int flags;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchAttributeValue__bindgen_ty_4
{
    public int deviceUpdatable;
    public CUgraphDeviceUpdatableNode_st* devNode;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchAttribute_st
{
    public cudaLaunchAttributeID id;
    public fixed byte pad[4];
    public cudaLaunchAttributeValue val;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaLaunchConfig_st
{
    public dim3 gridDim;
    public dim3 blockDim;
    public nuint dynamicSmemBytes;
    public cudaStream* stream;
    public cudaLaunchAttribute_st* attrs;
    public uint numAttrs;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaAsyncCallbackEntry
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaAsyncNotificationInfo
{
    public cudaAsyncNotificationType_enum type_;
    public cudaAsyncNotificationInfo__bindgen_ty_1 info;
}

[StructLayout(LayoutKind.Explicit)]
public unsafe partial struct cudaAsyncNotificationInfo__bindgen_ty_1
{
    [FieldOffset(0)]
    public cudaAsyncNotificationInfo__bindgen_ty_1__bindgen_ty_1 overBudget;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaAsyncNotificationInfo__bindgen_ty_1__bindgen_ty_1
{
    public ulong bytesOverBudget;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cudaTextureDesc
{
    public fixed byte/* cudaTextureAddressMode, this length is invalid so must keep pointer and can't edit from C# */ addressMode[3];
    public cudaTextureFilterMode filterMode;
    public cudaTextureReadMode readMode;
    public int sRGB;
    public fixed float borderColor[4];
    public int normalizedCoords;
    public uint maxAnisotropy;
    public cudaTextureFilterMode mipmapFilterMode;
    public float mipmapLevelBias;
    public float minMipmapLevelClamp;
    public float maxMipmapLevelClamp;
    public int disableTrilinearOptimization;
    public int seamlessCubemap;
}

public enum libraryPropertyType_t : int
{
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
}

public enum cudaError : int
{
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}

public enum cudaChannelFormatKind : int
{
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
    cudaChannelFormatKindUnsignedNormalized8X1 = 5,
    cudaChannelFormatKindUnsignedNormalized8X2 = 6,
    cudaChannelFormatKindUnsignedNormalized8X4 = 7,
    cudaChannelFormatKindUnsignedNormalized16X1 = 8,
    cudaChannelFormatKindUnsignedNormalized16X2 = 9,
    cudaChannelFormatKindUnsignedNormalized16X4 = 10,
    cudaChannelFormatKindSignedNormalized8X1 = 11,
    cudaChannelFormatKindSignedNormalized8X2 = 12,
    cudaChannelFormatKindSignedNormalized8X4 = 13,
    cudaChannelFormatKindSignedNormalized16X1 = 14,
    cudaChannelFormatKindSignedNormalized16X2 = 15,
    cudaChannelFormatKindSignedNormalized16X4 = 16,
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
    cudaChannelFormatKindSignedBlockCompressed4 = 24,
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
    cudaChannelFormatKindSignedBlockCompressed5 = 26,
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
    cudaChannelFormatKindSignedBlockCompressed6H = 28,
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30,
}

public enum cudaMemoryType : int
{
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
}

public enum cudaMemcpyKind : int
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

public enum cudaAccessProperty : int
{
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2,
}

public enum cudaStreamCaptureStatus : int
{
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
}

public enum cudaStreamCaptureMode : int
{
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}

public enum cudaSynchronizationPolicy : int
{
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4,
}

public enum cudaClusterSchedulingPolicy : int
{
    cudaClusterSchedulingPolicyDefault = 0,
    cudaClusterSchedulingPolicySpread = 1,
    cudaClusterSchedulingPolicyLoadBalancing = 2,
}

public enum cudaResourceType : int
{
    cudaResourceTypeArray = 0,
    cudaResourceTypeMipmappedArray = 1,
    cudaResourceTypeLinear = 2,
    cudaResourceTypePitch2D = 3,
}

public enum cudaResourceViewFormat : int
{
    cudaResViewFormatNone = 0,
    cudaResViewFormatUnsignedChar1 = 1,
    cudaResViewFormatUnsignedChar2 = 2,
    cudaResViewFormatUnsignedChar4 = 3,
    cudaResViewFormatSignedChar1 = 4,
    cudaResViewFormatSignedChar2 = 5,
    cudaResViewFormatSignedChar4 = 6,
    cudaResViewFormatUnsignedShort1 = 7,
    cudaResViewFormatUnsignedShort2 = 8,
    cudaResViewFormatUnsignedShort4 = 9,
    cudaResViewFormatSignedShort1 = 10,
    cudaResViewFormatSignedShort2 = 11,
    cudaResViewFormatSignedShort4 = 12,
    cudaResViewFormatUnsignedInt1 = 13,
    cudaResViewFormatUnsignedInt2 = 14,
    cudaResViewFormatUnsignedInt4 = 15,
    cudaResViewFormatSignedInt1 = 16,
    cudaResViewFormatSignedInt2 = 17,
    cudaResViewFormatSignedInt4 = 18,
    cudaResViewFormatHalf1 = 19,
    cudaResViewFormatHalf2 = 20,
    cudaResViewFormatHalf4 = 21,
    cudaResViewFormatFloat1 = 22,
    cudaResViewFormatFloat2 = 23,
    cudaResViewFormatFloat4 = 24,
    cudaResViewFormatUnsignedBlockCompressed1 = 25,
    cudaResViewFormatUnsignedBlockCompressed2 = 26,
    cudaResViewFormatUnsignedBlockCompressed3 = 27,
    cudaResViewFormatUnsignedBlockCompressed4 = 28,
    cudaResViewFormatSignedBlockCompressed4 = 29,
    cudaResViewFormatUnsignedBlockCompressed5 = 30,
    cudaResViewFormatSignedBlockCompressed5 = 31,
    cudaResViewFormatUnsignedBlockCompressed6H = 32,
    cudaResViewFormatSignedBlockCompressed6H = 33,
    cudaResViewFormatUnsignedBlockCompressed7 = 34,
}

public enum cudaFuncAttribute : int
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeClusterDimMustBeSet = 10,
    cudaFuncAttributeRequiredClusterWidth = 11,
    cudaFuncAttributeRequiredClusterHeight = 12,
    cudaFuncAttributeRequiredClusterDepth = 13,
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
    cudaFuncAttributeMax = 16,
}

public enum cudaFuncCache : int
{
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3,
}

public enum cudaSharedMemConfig : int
{
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2,
}

public enum cudaLimit : int
{
    cudaLimitStackSize = 0,
    cudaLimitPrintfFifoSize = 1,
    cudaLimitMallocHeapSize = 2,
    cudaLimitDevRuntimeSyncDepth = 3,
    cudaLimitDevRuntimePendingLaunchCount = 4,
    cudaLimitMaxL2FetchGranularity = 5,
    cudaLimitPersistingL2CacheSize = 6,
}

public enum cudaMemoryAdvise : int
{
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6,
}

public enum cudaMemRangeAttribute : int
{
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4,
    cudaMemRangeAttributePreferredLocationType = 5,
    cudaMemRangeAttributePreferredLocationId = 6,
    cudaMemRangeAttributeLastPrefetchLocationType = 7,
    cudaMemRangeAttributeLastPrefetchLocationId = 8,
}

public enum cudaFlushGPUDirectRDMAWritesScope : int
{
    cudaFlushGPUDirectRDMAWritesToOwner = 100,
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200,
}

public enum cudaFlushGPUDirectRDMAWritesTarget : int
{
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice = 0,
}

public enum cudaDeviceAttr : int
{
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrMax = 136,
}

public enum cudaMemPoolAttr : int
{
    cudaMemPoolReuseFollowEventDependencies = 1,
    cudaMemPoolReuseAllowOpportunistic = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold = 4,
    cudaMemPoolAttrReservedMemCurrent = 5,
    cudaMemPoolAttrReservedMemHigh = 6,
    cudaMemPoolAttrUsedMemCurrent = 7,
    cudaMemPoolAttrUsedMemHigh = 8,
}

public enum cudaMemLocationType : int
{
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
    cudaMemLocationTypeHostNuma = 3,
    cudaMemLocationTypeHostNumaCurrent = 4,
}

public enum cudaMemAccessFlags : int
{
    cudaMemAccessFlagsProtNone = 0,
    cudaMemAccessFlagsProtRead = 1,
    cudaMemAccessFlagsProtReadWrite = 3,
}

public enum cudaMemAllocationType : int
{
    cudaMemAllocationTypeInvalid = 0,
    cudaMemAllocationTypePinned = 1,
    cudaMemAllocationTypeMax = 2147483647,
}

public enum cudaMemAllocationHandleType : int
{
    cudaMemHandleTypeNone = 0,
    cudaMemHandleTypePosixFileDescriptor = 1,
    cudaMemHandleTypeWin32 = 2,
    cudaMemHandleTypeWin32Kmt = 4,
    cudaMemHandleTypeFabric = 8,
}

public enum cudaGraphMemAttributeType : int
{
    cudaGraphMemAttrUsedMemCurrent = 0,
    cudaGraphMemAttrUsedMemHigh = 1,
    cudaGraphMemAttrReservedMemCurrent = 2,
    cudaGraphMemAttrReservedMemHigh = 3,
}

public enum cudaDeviceP2PAttr : int
{
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3,
    cudaDevP2PAttrCudaArrayAccessSupported = 4,
}

public enum cudaExternalMemoryHandleType : int
{
    cudaExternalMemoryHandleTypeOpaqueFd = 1,
    cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalMemoryHandleTypeD3D12Heap = 4,
    cudaExternalMemoryHandleTypeD3D12Resource = 5,
    cudaExternalMemoryHandleTypeD3D11Resource = 6,
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    cudaExternalMemoryHandleTypeNvSciBuf = 8,
}

public enum cudaExternalSemaphoreHandleType : int
{
    cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalSemaphoreHandleTypeD3D12Fence = 4,
    cudaExternalSemaphoreHandleTypeD3D11Fence = 5,
    cudaExternalSemaphoreHandleTypeNvSciSync = 6,
    cudaExternalSemaphoreHandleTypeKeyedMutex = 7,
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10,
}

public enum cudaGraphConditionalNodeType : int
{
    cudaGraphCondTypeIf = 0,
    cudaGraphCondTypeWhile = 1,
}

public enum cudaGraphNodeType : int
{
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeConditional = 13,
    cudaGraphNodeTypeCount = 14,
}

public enum cudaGraphExecUpdateResult : int
{
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
    cudaGraphExecUpdateErrorAttributesChanged = 8,
}

public enum cudaGraphInstantiateResult : int
{
    cudaGraphInstantiateSuccess = 0,
    cudaGraphInstantiateError = 1,
    cudaGraphInstantiateInvalidStructure = 2,
    cudaGraphInstantiateNodeOperationNotSupported = 3,
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,
}

public enum cudaDriverEntryPointQueryResult : int
{
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1,
    cudaDriverEntryPointVersionNotSufficent = 2,
}

public enum cudaLaunchMemSyncDomain : int
{
    cudaLaunchMemSyncDomainDefault = 0,
    cudaLaunchMemSyncDomainRemote = 1,
}

public enum cudaLaunchAttributeID : int
{
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
}

public enum cudaAsyncNotificationType_enum : int
{
    cudaAsyncNotificationTypeOverBudget = 1,
}

public enum cudaTextureAddressMode : int
{
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3,
}

public enum cudaTextureFilterMode : int
{
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1,
}

public enum cudaTextureReadMode : int
{
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1,
}
