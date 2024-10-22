global using static Zen.CUDA.Interop.CudaRuntime;

using System.Runtime.InteropServices;

namespace Zen.CUDA.Interop;

internal static unsafe partial class CudaRuntime
{
    private const string LibraryName = "cudart64_12.dll";

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceReset")]
    public static partial cudaStatus cudaDeviceReset();

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceSynchronize")]
    public static partial cudaStatus cudaDeviceSynchronize();

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceSetLimit")]
    public static partial cudaStatus cudaDeviceSetLimit(cudaLimit limit, nuint value);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGetLimit")]
    public static partial cudaStatus cudaDeviceGetLimit(nuint* pValue, cudaLimit limit);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGetCacheConfig")]
    public static partial cudaStatus cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGetStreamPriorityRange")]
    public static partial cudaStatus cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceSetCacheConfig")]
    public static partial cudaStatus cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGetSharedMemConfig")]
    public static partial cudaStatus cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceSetSharedMemConfig")]
    public static partial cudaStatus cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadExit")]
    public static partial cudaStatus cudaThreadExit();

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadSynchronize")]
    public static partial cudaStatus cudaThreadSynchronize();

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadSetLimit")]
    public static partial cudaStatus cudaThreadSetLimit(cudaLimit limit, nuint value);

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadGetLimit")]
    public static partial cudaStatus cudaThreadGetLimit(nuint* pValue, cudaLimit limit);

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadGetCacheConfig")]
    public static partial cudaStatus cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig);

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadSetCacheConfig")]
    public static partial cudaStatus cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetLastError")]
    public static partial cudaStatus cudaGetLastError();

    [LibraryImport(LibraryName, EntryPoint = "cudaPeekAtLastError")]
    public static partial cudaStatus cudaPeekAtLastError();

    [LibraryImport(LibraryName, EntryPoint = "cudaGetErrorName")]
    public static partial byte* cudaGetErrorName(cudaStatus status);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetErrorString")]
    public static partial byte* cudaGetErrorString(cudaStatus status);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetDeviceCount")]
    public static partial cudaStatus cudaGetDeviceCount(int* count);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetDeviceProperties_v2")]
    public static partial cudaStatus cudaGetDeviceProperties(cudaDeviceProp* prop, int device);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGetAttribute")]
    public static partial cudaStatus cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);

    [LibraryImport(LibraryName, EntryPoint = "cudaChooseDevice")]
    public static partial cudaStatus cudaChooseDevice(int* device, cudaDeviceProp* prop);

    [LibraryImport(LibraryName, EntryPoint = "cudaInitDevice")]
    public static partial cudaStatus cudaInitDevice(int device, uint deviceFlags, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaSetDevice")]
    public static partial cudaStatus cudaSetDevice(int device);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetDevice")]
    public static partial cudaStatus cudaGetDevice(int* device);

    [LibraryImport(LibraryName, EntryPoint = "cudaSetValidDevices")]
    public static partial cudaStatus cudaSetValidDevices(int* device_arr, int len);

    [LibraryImport(LibraryName, EntryPoint = "cudaSetDeviceFlags")]
    public static partial cudaStatus cudaSetDeviceFlags(uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetDeviceFlags")]
    public static partial cudaStatus cudaGetDeviceFlags(uint* flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamCreate")]
    public static partial cudaStatus cudaStreamCreate(cudaStream** pStream);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamCreateWithFlags")]
    public static partial cudaStatus cudaStreamCreateWithFlags(cudaStream** pStream, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamCreateWithPriority")]
    public static partial cudaStatus cudaStreamCreateWithPriority(cudaStream** pStream, uint flags, int priority);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamGetPriority")]
    public static partial cudaStatus cudaStreamGetPriority(cudaStream* hStream, int* priority);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamGetFlags")]
    public static partial cudaStatus cudaStreamGetFlags(cudaStream* hStream, uint* flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamGetId")]
    public static partial cudaStatus cudaStreamGetId(cudaStream* hStream, ulong* streamId);

    [LibraryImport(LibraryName, EntryPoint = "cudaCtxResetPersistingL2Cache")]
    public static partial cudaStatus cudaCtxResetPersistingL2Cache();

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamDestroy")]
    public static partial cudaStatus cudaStreamDestroy(cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamWaitEvent")]
    public static partial cudaStatus cudaStreamWaitEvent(cudaStream* stream, cudaEvent* @event, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamAddCallback")]
    public static partial cudaStatus cudaStreamAddCallback(cudaStream* stream, delegate* unmanaged[Cdecl]<cudaStream*, cudaStatus, void*, void> callback, void* userData, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamSynchronize")]
    public static partial cudaStatus cudaStreamSynchronize(cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamQuery")]
    public static partial cudaStatus cudaStreamQuery(cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamAttachMemAsync")]
    public static partial cudaStatus cudaStreamAttachMemAsync(cudaStream* stream, void* devPtr, nuint length, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamBeginCapture")]
    public static partial cudaStatus cudaStreamBeginCapture(cudaStream* stream, cudaStreamCaptureMode mode);

    [LibraryImport(LibraryName, EntryPoint = "cudaThreadExchangeStreamCaptureMode")]
    public static partial cudaStatus cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamEndCapture")]
    public static partial cudaStatus cudaStreamEndCapture(cudaStream* stream, cudaGraph** pGraph);

    [LibraryImport(LibraryName, EntryPoint = "cudaStreamIsCapturing")]
    public static partial cudaStatus cudaStreamIsCapturing(cudaStream* stream, cudaStreamCaptureStatus* pCaptureStatus);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventCreate")]
    public static partial cudaStatus cudaEventCreate(cudaEvent** @event);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventCreateWithFlags")]
    public static partial cudaStatus cudaEventCreateWithFlags(cudaEvent** @event, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventRecord")]
    public static partial cudaStatus cudaEventRecord(cudaEvent* @event, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventRecordWithFlags")]
    public static partial cudaStatus cudaEventRecordWithFlags(cudaEvent* @event, cudaStream* stream, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventQuery")]
    public static partial cudaStatus cudaEventQuery(cudaEvent* @event);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventSynchronize")]
    public static partial cudaStatus cudaEventSynchronize(cudaEvent* @event);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventDestroy")]
    public static partial cudaStatus cudaEventDestroy(cudaEvent* @event);

    [LibraryImport(LibraryName, EntryPoint = "cudaEventElapsedTime")]
    public static partial cudaStatus cudaEventElapsedTime(float* ms, cudaEvent* start, cudaEvent* end);

    [LibraryImport(LibraryName, EntryPoint = "cudaMallocManaged")]
    public static partial cudaStatus cudaMallocManaged(void** devPtr, nuint size, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaMalloc")]
    public static partial cudaStatus cudaMalloc(void** devPtr, nuint size);

    [LibraryImport(LibraryName, EntryPoint = "cudaMallocHost")]
    public static partial cudaStatus cudaMallocHost(void** ptr, nuint size);

    [LibraryImport(LibraryName, EntryPoint = "cudaMallocPitch")]
    public static partial cudaStatus cudaMallocPitch(void** devPtr, nuint* pitch, nuint width, nuint height);

    [LibraryImport(LibraryName, EntryPoint = "cudaFree")]
    public static partial cudaStatus cudaFree(void* devPtr);

    [LibraryImport(LibraryName, EntryPoint = "cudaFreeHost")]
    public static partial cudaStatus cudaFreeHost(void* ptr);

    [LibraryImport(LibraryName, EntryPoint = "cudaHostAlloc")]
    public static partial cudaStatus cudaHostAlloc(void** pHost, nuint size, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaHostRegister")]
    public static partial cudaStatus cudaHostRegister(void* ptr, nuint size, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaHostUnregister")]
    public static partial cudaStatus cudaHostUnregister(void* ptr);

    [LibraryImport(LibraryName, EntryPoint = "cudaHostGetDevicePointer")]
    public static partial cudaStatus cudaHostGetDevicePointer(void** pDevice, void* pHost, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaHostGetFlags")]
    public static partial cudaStatus cudaHostGetFlags(uint* pFlags, void* pHost);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemGetInfo")]
    public static partial cudaStatus cudaMemGetInfo(nuint* free, nuint* total);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpy")]
    public static partial cudaStatus cudaMemcpy(void* dst, void* src, nuint count, cudaMemcpyKind kind);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpyPeer")]
    public static partial cudaStatus cudaMemcpyPeer(void* dst, int dstDevice, void* src, int srcDevice, nuint count);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpy2D")]
    public static partial cudaStatus cudaMemcpy2D(void* dst, nuint dpitch, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpyToSymbol")]
    public static partial cudaStatus cudaMemcpyToSymbol(void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpyFromSymbol")]
    public static partial cudaStatus cudaMemcpyFromSymbol(void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpyAsync")]
    public static partial cudaStatus cudaMemcpyAsync(void* dst, void* src, nuint count, cudaMemcpyKind kind, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpy2DAsync")]
    public static partial cudaStatus cudaMemcpy2DAsync(void* dst, nuint dpitch, void* src, nuint spitch, nuint width, nuint height, cudaMemcpyKind kind, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpyToSymbolAsync")]
    public static partial cudaStatus cudaMemcpyToSymbolAsync(void* symbol, void* src, nuint count, nuint offset, cudaMemcpyKind kind, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemcpyFromSymbolAsync")]
    public static partial cudaStatus cudaMemcpyFromSymbolAsync(void* dst, void* symbol, nuint count, nuint offset, cudaMemcpyKind kind, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetSymbolAddress")]
    public static partial cudaStatus cudaGetSymbolAddress(void** devPtr, void* symbol);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetSymbolSize")]
    public static partial cudaStatus cudaGetSymbolSize(nuint* size, void* symbol);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemPrefetchAsync")]
    public static partial cudaStatus cudaMemPrefetchAsync(void* devPtr, nuint count, int dstDevice, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemPrefetchAsync_v2")]
    public static partial cudaStatus cudaMemPrefetchAsync_v2(void* devPtr, nuint count, cudaMemLocation location, uint flags, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemAdvise")]
    public static partial cudaStatus cudaMemAdvise(void* devPtr, nuint count, cudaMemoryAdvise advice, int device);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemAdvise_v2")]
    public static partial cudaStatus cudaMemAdvise_v2(void* devPtr, nuint count, cudaMemoryAdvise advice, cudaMemLocation location);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemRangeGetAttribute")]
    public static partial cudaStatus cudaMemRangeGetAttribute(void* data, nuint dataSize, cudaMemRangeAttribute attribute, void* devPtr, nuint count);

    [LibraryImport(LibraryName, EntryPoint = "cudaMemRangeGetAttributes")]
    public static partial cudaStatus cudaMemRangeGetAttributes(void** data, nuint* dataSizes, cudaMemRangeAttribute* attributes, nuint numAttributes, void* devPtr, nuint count);

    [LibraryImport(LibraryName, EntryPoint = "cudaMallocAsync")]
    public static partial cudaStatus cudaMallocAsync(void** devPtr, nuint size, cudaStream* hStream);

    [LibraryImport(LibraryName, EntryPoint = "cudaFreeAsync")]
    public static partial cudaStatus cudaFreeAsync(void* devPtr, cudaStream* hStream);

    [LibraryImport(LibraryName, EntryPoint = "cudaPointerGetAttributes")]
    public static partial cudaStatus cudaPointerGetAttributes(cudaPointerAttributes* attributes, void* ptr);

    [LibraryImport(LibraryName, EntryPoint = "cudaDriverGetVersion")]
    public static partial cudaStatus cudaDriverGetVersion(int* driverVersion);

    [LibraryImport(LibraryName, EntryPoint = "cudaRuntimeGetVersion")]
    public static partial cudaStatus cudaRuntimeGetVersion(int* runtimeVersion);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphCreate")]
    public static partial cudaStatus cudaGraphCreate(cudaGraph** pGraph, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGraphMemTrim")]
    public static partial cudaStatus cudaDeviceGraphMemTrim(int device);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceGetGraphMemAttribute")]
    public static partial cudaStatus cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);

    [LibraryImport(LibraryName, EntryPoint = "cudaDeviceSetGraphMemAttribute")]
    public static partial cudaStatus cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphClone")]
    public static partial cudaStatus cudaGraphClone(cudaGraph** pGraphClone, cudaGraph* originalGraph);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphInstantiate")]
    public static partial cudaStatus cudaGraphInstantiate(cudaGraphInstance** pGraphExec, cudaGraph* graph, ulong flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphInstantiateWithFlags")]
    public static partial cudaStatus cudaGraphInstantiateWithFlags(cudaGraphInstance** pGraphExec, cudaGraph* graph, ulong flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphExecGetFlags")]
    public static partial cudaStatus cudaGraphExecGetFlags(cudaGraphInstance* graphExec, ulong* flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphUpload")]
    public static partial cudaStatus cudaGraphUpload(cudaGraphInstance* graphExec, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphLaunch")]
    public static partial cudaStatus cudaGraphLaunch(cudaGraphInstance* graphExec, cudaStream* stream);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphExecDestroy")]
    public static partial cudaStatus cudaGraphExecDestroy(cudaGraphInstance* graphExec);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphDestroy")]
    public static partial cudaStatus cudaGraphDestroy(cudaGraph* graph);

    [LibraryImport(LibraryName, EntryPoint = "cudaGraphDebugDotPrint")]
    public static partial cudaStatus cudaGraphDebugDotPrint(cudaGraph* graph, byte* path, uint flags);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetDriverEntryPoint")]
    public static partial cudaStatus cudaGetDriverEntryPoint(byte* symbol, void** funcPtr, ulong flags, cudaDriverEntryPointQueryResult* driverStatus);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetDriverEntryPointByVersion")]
    public static partial cudaStatus cudaGetDriverEntryPointByVersion(byte* symbol, void** funcPtr, uint cudaVersion, ulong flags, cudaDriverEntryPointQueryResult* driverStatus);

    [LibraryImport(LibraryName, EntryPoint = "cudaGetExportTable")]
    public static partial cudaStatus cudaGetExportTable(void** ppExportTable, cudaGuid* pExportTableId);
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
