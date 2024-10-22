using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

public static unsafe class DeviceArray
{
    public static DeviceArray<T> Allocate<T>(int size) where T : unmanaged
    {
        var bytes = (uint)(size * sizeof(T));
        var pointer = (T*)AllocateSpan(bytes);
        return new(pointer, size);
    }

    public static DeviceArray<T> AllocateAsync<T>(int size, CudaStream stream) where T : unmanaged
    {
        var bytes = (uint)(size * sizeof(T));
        var pointer = (T*)AllocateSpanAsync(bytes, stream.Pointer);
        return new(pointer, size);
    }
    
    private static void* AllocateSpan(uint bytes)
    {
        void* pointer;
        var error = cudaMalloc(&pointer, bytes);
        Status.EnsureIsSuccess(error);
        return pointer;
    }
    
    private static void* AllocateSpanAsync(uint bytes, cudaStream* stream)
    {
        void* pointer;
        var error = cudaMallocAsync(&pointer, bytes, stream);
        Status.EnsureIsSuccess(error);
        return pointer;
    }
}

public sealed unsafe class DeviceArray<T> : TensorRef<T>, IDisposable where T : unmanaged
{
    public readonly int Size;

    internal DeviceArray(T* pointer, int size) : base(pointer)
    {
        Size = size;
    }

    public void CopyTo(HostArray<T> array, CudaStream stream)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyDeviceToHost, stream.Pointer);
    }

    public void CopyTo(DeviceArray<T> array, CudaStream stream)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream.Pointer);
    }

    public void Dispose()
    {
        cudaFree(Pointer);
    }
}