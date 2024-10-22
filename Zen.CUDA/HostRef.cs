namespace Zen.CUDA;

public unsafe class HostRef<T> where T : unmanaged
{
    internal readonly T* Pointer;
    internal readonly int ElementSize;
    
    internal HostRef(T* pointer)
    {
        ElementSize = sizeof(T);
        Pointer = pointer;
    }
    
    public T this[int offset]
    {
        get => Pointer[offset];
        set => Pointer[offset] = value;
    }
    
    public static HostRef<T> operator +(HostRef<T> array, int offset) => new(array.Pointer + offset);
    public static HostRef<T> operator -(HostRef<T> array, int offset) => new(array.Pointer - offset);
}