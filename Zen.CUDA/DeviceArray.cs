using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

public static unsafe class DeviceArray
{
    public static DeviceArray<T> Allocate<T>(int size, cudaStream* stream = null) where T : unmanaged
    {
        var bytes = (uint)(size * sizeof(T));
        var pointer = (T*)AllocateSpan(bytes, stream);
        var array = new DeviceArray<T>(pointer, size);
        return array;
    }
    
    private static void* AllocateSpan(uint bytes, cudaStream* stream)
    {
        void* pointer;
        
        var error = cudaMallocAsync(&pointer, bytes, stream);
        if (error is cudaError.cudaSuccess) 
            return pointer;
        
        throw new CudaException(error);
    }
}

public sealed unsafe class DeviceArray<T> : IDisposable where T : unmanaged
{
    public readonly int ElementSize;
    public readonly int Size;
    public readonly T* Pointer;

    internal DeviceArray(T* pointer, int size)
    {
        ElementSize = sizeof(T);
        Size = size;
        Pointer = pointer;
    }

    public void CopyTo(HostArray<T> array, cudaStream* stream = null)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyDeviceToHost, stream);
    }

    public void CopyTo(DeviceArray<T> array, cudaStream* stream = null)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream);
    }

    public void Dispose()
    {
        cudaFree(Pointer);
    }
}
