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

public sealed unsafe class HostArray<T> : IDisposable where T : unmanaged
{
    internal readonly T* Pointer;

    public readonly int ElementSize;
    public readonly int Size;

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

    public static HostArray<T> operator +(HostArray<T> array, int offset) => new(array.Pointer + offset, array.ElementSize);
    public static HostArray<T> operator -(HostArray<T> array, int offset) => new(array.Pointer - offset, array.ElementSize);

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
