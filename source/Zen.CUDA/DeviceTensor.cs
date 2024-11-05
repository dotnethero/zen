using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

public static class DeviceTensor
{
    public static DeviceTensor<T> Allocate<T>(Shape shape) where T : unmanaged
    {
        var array = DeviceArray.Allocate<T>(shape.Size);
        var tensor = new DeviceTensor<T>(shape, array);
        return tensor;
    }
}

public sealed class DeviceTensor<T> : Tensor<T>, IDisposable where T : unmanaged
{
    public readonly DeviceArray<T> Array;

    public DeviceTensor(Shape shape, DeviceArray<T> array) : base(shape, array)
    {
        Array = array;
    }

    public void CopyTo(HostTensor<T> tensor, CudaStream? stream = null) // TODO: check shapes
    {
        Array.CopyTo(tensor.Array, stream ?? CudaStream.Default);
    }

    public void CopyTo(DeviceTensor<T> tensor, CudaStream? stream = null)
    {
        Array.CopyTo(tensor.Array, stream ?? CudaStream.Default);
    }

    public void Dispose()
    {
        Array.Dispose();
    }
}
