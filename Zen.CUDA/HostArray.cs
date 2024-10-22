using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

public static unsafe class HostArray
{
    public static HostArray<T> Allocate<T>(int size) where T : unmanaged
    {
        var bytes = (uint)(size * sizeof(T));
        var pointer = (T*)AllocateSpan(bytes);
        var array = new HostArray<T>(pointer, size);
        return array;
    }
    
    private static void* AllocateSpan(uint bytes)
    {
        void* pointer;
        
        var error = cudaMallocHost(&pointer, bytes);
        if (error is cudaError.cudaSuccess) 
            return pointer;
        
        throw new CudaException(error);
    }
}

public sealed unsafe class HostArray<T> : IDisposable where T : unmanaged
{
    public readonly int ElementSize;
    public readonly int Size;
    public readonly T* Pointer;

    internal HostArray(T* pointer, int size)
    {
        ElementSize = sizeof(T);
        Size = size;
        Pointer = pointer;
    }

    public T this[int offset]
    {
        get => Pointer[offset];
        set => Pointer[offset] = value;
    }

    public Span<T> this[Range range] => AsSpan()[range];

    public Span<T> AsSpan() => new(Pointer, Size);

    public void CopyTo(HostArray<T> array, cudaStream* stream = null)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyHostToHost, stream);
    }

    public void CopyTo(DeviceArray<T> array, cudaStream* stream = null)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyHostToDevice, stream);
    }
    
    public void Sync()
    {
        cudaDeviceSynchronize();
    }
    
    public void Dispose()
    {
        cudaFreeHost(Pointer);
    }
}
