// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming

global using static Zen.CUDA.Interop.CudaRuntime;

using System.Runtime.InteropServices;

namespace Zen.CUDA.Interop;

internal static unsafe class CudaRuntime
{
    private const string __DllName = "cudart64_12.dll";

    [DllImport(__DllName, EntryPoint = "cudaDeviceReset", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceReset();

    [DllImport(__DllName, EntryPoint = "cudaDeviceSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceSynchronize();

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceSetLimit(cudaLimit limit, nuint value);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGetLimit(nuint* pValue, cudaLimit limit);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetStreamPriorityRange", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetSharedMemConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetSharedMemConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);

    [DllImport(__DllName, EntryPoint = "cudaThreadExit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadExit();

    [DllImport(__DllName, EntryPoint = "cudaThreadSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadSynchronize();

    [DllImport(__DllName, EntryPoint = "cudaThreadSetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadSetLimit(cudaLimit limit, nuint value);

    [DllImport(__DllName, EntryPoint = "cudaThreadGetLimit", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadGetLimit(nuint* pValue, cudaLimit limit);

    [DllImport(__DllName, EntryPoint = "cudaThreadGetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaThreadSetCacheConfig", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);

    [DllImport(__DllName, EntryPoint = "cudaGetLastError", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetLastError();

    [DllImport(__DllName, EntryPoint = "cudaPeekAtLastError", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaPeekAtLastError();

    [DllImport(__DllName, EntryPoint = "cudaGetErrorName", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern byte* cudaGetErrorName(cudaStatus status);

    [DllImport(__DllName, EntryPoint = "cudaGetErrorString", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern byte* cudaGetErrorString(cudaStatus status);

    [DllImport(__DllName, EntryPoint = "cudaGetDeviceCount", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetDeviceCount(int* count);

    [DllImport(__DllName, EntryPoint = "cudaGetDeviceProperties_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetDeviceProperties(cudaDeviceProp* prop, int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);

    [DllImport(__DllName, EntryPoint = "cudaChooseDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaChooseDevice(int* device, cudaDeviceProp* prop);

    [DllImport(__DllName, EntryPoint = "cudaInitDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaInitDevice(int device, uint deviceFlags, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaSetDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaSetDevice(int device);

    [DllImport(__DllName, EntryPoint = "cudaGetDevice", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetDevice(int* device);

    [DllImport(__DllName, EntryPoint = "cudaSetValidDevices", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaSetValidDevices(int* device_arr, int len);

    [DllImport(__DllName, EntryPoint = "cudaSetDeviceFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaSetDeviceFlags(uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGetDeviceFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetDeviceFlags(uint* flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamCreate(cudaStream** pStream);

    [DllImport(__DllName, EntryPoint = "cudaStreamCreateWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamCreateWithFlags(cudaStream** pStream, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamCreateWithPriority", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamCreateWithPriority(cudaStream** pStream, uint flags, int priority);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetPriority", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamGetPriority(cudaStream* hStream, int* priority);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamGetFlags(cudaStream* hStream, uint* flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamGetId", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamGetId(cudaStream* hStream, ulong* streamId);

    [DllImport(__DllName, EntryPoint = "cudaCtxResetPersistingL2Cache", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaCtxResetPersistingL2Cache();

    [DllImport(__DllName, EntryPoint = "cudaStreamDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamDestroy(cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaStreamWaitEvent", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamWaitEvent(cudaStream* stream, cudaEvent* @event, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamAddCallback", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamAddCallback(cudaStream* stream, delegate* unmanaged[Cdecl]<cudaStream*, cudaStatus, void*, void> callback, void* userData, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamSynchronize(cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaStreamQuery", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamQuery(cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaStreamAttachMemAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamAttachMemAsync(cudaStream* stream, void* devPtr, nuint length, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaStreamBeginCapture", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamBeginCapture(cudaStream* stream, cudaStreamCaptureMode mode);

    [DllImport(__DllName, EntryPoint = "cudaThreadExchangeStreamCaptureMode", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode);

    [DllImport(__DllName, EntryPoint = "cudaStreamEndCapture", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamEndCapture(cudaStream* stream, cudaGraph** pGraph);

    [DllImport(__DllName, EntryPoint = "cudaStreamIsCapturing", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaStreamIsCapturing(cudaStream* stream, cudaStreamCaptureStatus* pCaptureStatus);

    [DllImport(__DllName, EntryPoint = "cudaEventCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventCreate(cudaEvent** @event);

    [DllImport(__DllName, EntryPoint = "cudaEventCreateWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventCreateWithFlags(cudaEvent** @event, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaEventRecord", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventRecord(cudaEvent* @event, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaEventRecordWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventRecordWithFlags(cudaEvent* @event, cudaStream* stream, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaEventQuery", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventQuery(cudaEvent* @event);

    [DllImport(__DllName, EntryPoint = "cudaEventSynchronize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventSynchronize(cudaEvent* @event);

    [DllImport(__DllName, EntryPoint = "cudaEventDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventDestroy(cudaEvent* @event);

    [DllImport(__DllName, EntryPoint = "cudaEventElapsedTime", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaEventElapsedTime(float* ms, cudaEvent* start, cudaEvent* end);

    [DllImport(__DllName, EntryPoint = "cudaMallocManaged", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMallocManaged(void** devPtr, nuint size, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaMalloc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMalloc(void** devPtr, nuint size);

    [DllImport(__DllName, EntryPoint = "cudaMallocHost", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMallocHost(void** ptr, nuint size);

    [DllImport(__DllName, EntryPoint = "cudaMallocPitch", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMallocPitch(void** devPtr, nuint* pitch, nuint width, nuint height);

    [DllImport(__DllName, EntryPoint = "cudaFree", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaFree(void* devPtr);

    [DllImport(__DllName, EntryPoint = "cudaFreeHost", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaFreeHost(void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaHostAlloc", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaHostAlloc(void** pHost, nuint size, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaHostRegister", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaHostRegister(void* ptr, nuint size, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaHostUnregister", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaHostUnregister(void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaHostGetDevicePointer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaHostGetDevicePointer(void** pDevice, void* pHost, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaHostGetFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaHostGetFlags(uint* pFlags, void* pHost);

    [DllImport(__DllName, EntryPoint = "cudaMemGetInfo", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemGetInfo(nuint* free, nuint* total);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpy(void* dst, void* src, nuint count, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyPeer", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpyPeer(void* dst, int dstDevice, void* src, int srcDevice, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2D", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpy2D(void* dst, nuint dpitch, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyToSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpyToSymbol(void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyFromSymbol", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpyFromSymbol(void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpyAsync(void* dst, void* src, nuint count, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpy2DAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpy2DAsync(void* dst, nuint dpitch, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyToSymbolAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpyToSymbolAsync(void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemcpyFromSymbolAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemcpyFromSymbolAsync(void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGetSymbolAddress", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetSymbolAddress(void** devPtr, void* symbol);

    [DllImport(__DllName, EntryPoint = "cudaGetSymbolSize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetSymbolSize(nuint* size, void* symbol);

    [DllImport(__DllName, EntryPoint = "cudaMemPrefetchAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemPrefetchAsync(void* devPtr, nuint count, int dstDevice, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemPrefetchAsync_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemPrefetchAsync_v2(void* devPtr, nuint count, cudaMemLocation location, uint flags, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaMemAdvise", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemAdvise(void* devPtr, nuint count, cudaMemoryAdvise advice, int device);

    [DllImport(__DllName, EntryPoint = "cudaMemAdvise_v2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemAdvise_v2(void* devPtr, nuint count, cudaMemoryAdvise advice, cudaMemLocation location);

    [DllImport(__DllName, EntryPoint = "cudaMemRangeGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemRangeGetAttribute(void* data, nuint dataSize, cudaMemRangeAttribute attribute, void* devPtr, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMemRangeGetAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMemRangeGetAttributes(void** data, nuint* dataSizes, cudaMemRangeAttribute* attributes, nuint numAttributes, void* devPtr, nuint count);

    [DllImport(__DllName, EntryPoint = "cudaMallocAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaMallocAsync(void** devPtr, nuint size, cudaStream* hStream);

    [DllImport(__DllName, EntryPoint = "cudaFreeAsync", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaFreeAsync(void* devPtr, cudaStream* hStream);

    [DllImport(__DllName, EntryPoint = "cudaPointerGetAttributes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaPointerGetAttributes(cudaPointerAttributes* attributes, void* ptr);

    [DllImport(__DllName, EntryPoint = "cudaDriverGetVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDriverGetVersion(int* driverVersion);

    [DllImport(__DllName, EntryPoint = "cudaRuntimeGetVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaRuntimeGetVersion(int* runtimeVersion);

    [DllImport(__DllName, EntryPoint = "cudaGraphCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphCreate(cudaGraph** pGraph, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGraphMemTrim", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGraphMemTrim(int device);

    [DllImport(__DllName, EntryPoint = "cudaDeviceGetGraphMemAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);

    [DllImport(__DllName, EntryPoint = "cudaDeviceSetGraphMemAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);

    [DllImport(__DllName, EntryPoint = "cudaGraphClone", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphClone(cudaGraph** pGraphClone, cudaGraph* originalGraph);

    [DllImport(__DllName, EntryPoint = "cudaGraphInstantiate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphInstantiate(cudaGraphInstance** pGraphExec, cudaGraph* graph, ulong flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphInstantiateWithFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphInstantiateWithFlags(cudaGraphInstance** pGraphExec, cudaGraph* graph, ulong flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecGetFlags", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphExecGetFlags(cudaGraphInstance* graphExec, ulong* flags);

    [DllImport(__DllName, EntryPoint = "cudaGraphUpload", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphUpload(cudaGraphInstance* graphExec, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGraphLaunch", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphLaunch(cudaGraphInstance* graphExec, cudaStream* stream);

    [DllImport(__DllName, EntryPoint = "cudaGraphExecDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphExecDestroy(cudaGraphInstance* graphExec);

    [DllImport(__DllName, EntryPoint = "cudaGraphDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphDestroy(cudaGraph* graph);

    [DllImport(__DllName, EntryPoint = "cudaGraphDebugDotPrint", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGraphDebugDotPrint(cudaGraph* graph, byte* path, uint flags);

    [DllImport(__DllName, EntryPoint = "cudaGetDriverEntryPoint", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetDriverEntryPoint(byte* symbol, void** funcPtr, ulong flags, cudaDriverEntryPointQueryResult* driverStatus);

    [DllImport(__DllName, EntryPoint = "cudaGetDriverEntryPointByVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetDriverEntryPointByVersion(byte* symbol, void** funcPtr, uint cudaVersion, ulong flags, cudaDriverEntryPointQueryResult* driverStatus);

    [DllImport(__DllName, EntryPoint = "cudaGetExportTable", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cudaStatus cudaGetExportTable(void** ppExportTable, cudaGuid* pExportTableId);
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct cudaPointerAttributes
{
    public cudaMemoryType type_;
    public int device;
    public void* devicePointer;
    public void* hostPointer;
}

[StructLayout(LayoutKind.Sequential)]
internal struct cudaMemLocation
{
    public cudaMemLocationType type_;
    public int id;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct cudaGuid
{
    public fixed byte bytes[16];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct cudaDeviceProp
{
    public fixed byte name[256];
    public cudaGuid uuid;
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
internal struct cudaStream
{
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct cudaEvent
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct cudaGraph
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct cudaGraphInstance
{
    public fixed byte _unused[1];
}

internal enum cudaStatus
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

internal enum cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
}

internal enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

internal enum cudaStreamCaptureStatus
{
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
}

internal enum cudaStreamCaptureMode
{
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}

internal enum cudaFuncCache
{
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3,
}

internal enum cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2,
}

internal enum cudaLimit
{
    cudaLimitStackSize = 0,
    cudaLimitPrintfFifoSize = 1,
    cudaLimitMallocHeapSize = 2,
    cudaLimitDevRuntimeSyncDepth = 3,
    cudaLimitDevRuntimePendingLaunchCount = 4,
    cudaLimitMaxL2FetchGranularity = 5,
    cudaLimitPersistingL2CacheSize = 6,
}

internal enum cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6,
}

internal enum cudaMemRangeAttribute
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

internal enum cudaDeviceAttr
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

internal enum cudaMemLocationType
{
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
    cudaMemLocationTypeHostNuma = 3,
    cudaMemLocationTypeHostNumaCurrent = 4,
}

internal enum cudaGraphMemAttributeType
{
    cudaGraphMemAttrUsedMemCurrent = 0,
    cudaGraphMemAttrUsedMemHigh = 1,
    cudaGraphMemAttrReservedMemCurrent = 2,
    cudaGraphMemAttrReservedMemHigh = 3,
}

internal enum cudaDriverEntryPointQueryResult
{
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1,
    cudaDriverEntryPointVersionNotSufficent = 2,
}
