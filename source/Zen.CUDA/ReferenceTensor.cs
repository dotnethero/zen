namespace Zen.CUDA;

public static class ReferenceTensor
{
    public static ReferenceTensor<T> Allocate<T>(Shape shape) where T : unmanaged
    {
        var host = HostTensor.Allocate<T>(shape);
        var device = DeviceTensor.Allocate<T>(shape);
        return new ReferenceTensor<T>(host, device);
    }
}

public sealed class ReferenceTensor<T> : IDisposable where T : unmanaged
{
    public readonly HostTensor<T> Host;
    public readonly DeviceTensor<T> Device;

    public Shape Shape => Host.Shape;
    public int Size => Host.Size;
    public int Cosize => Host.Cosize;
    
    public ReferenceTensor(HostTensor<T> host, DeviceTensor<T> device)
    {
        Host = host;
        Device = device;
    }

    public void SyncToDevice()
    {
        Host.CopyTo(Device);
    }

    public void SyncToHost()
    {
        Device.CopyTo(Host);
    }

    public void Dispose()
    {
        Host.Dispose();
        Device.Dispose();
    }
}