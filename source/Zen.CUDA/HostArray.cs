using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

public static unsafe class HostArray
{
    public static HostArray<T> Allocate<T>(int size) where T : unmanaged
    {
        var bytes = (uint)(size * sizeof(T));
        var pointer = (T*)AllocateSpan(bytes);
        return new(pointer, size);
    }
    
    private static void* AllocateSpan(uint bytes)
    {
        void* pointer;
        var error = cudaMallocHost(&pointer, bytes);
        Status.EnsureIsSuccess(error);
        return pointer;
    }
}

public sealed unsafe class HostArray<T> : TensorRef<T>, IDisposable where T : unmanaged
{
    public readonly int Size;

    internal HostArray(T* pointer, int size) : base(pointer)
    {
        Size = size;
    }

    public Span<T> this[Range range] => AsSpan()[range];

    public Span<T> AsSpan() => new(Pointer, Size);

    public void CopyTo(HostArray<T> array, CudaStream stream)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyHostToHost, stream.Pointer);
    }

    public void CopyTo(DeviceArray<T> array, CudaStream stream)
    {
        cudaMemcpyAsync(array.Pointer, Pointer, (nuint)(Size * ElementSize), cudaMemcpyKind.cudaMemcpyHostToDevice, stream.Pointer);
    }
    
    public void Dispose()
    {
        cudaFreeHost(Pointer);
    }
}
