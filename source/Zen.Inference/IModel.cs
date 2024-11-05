using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Inference;

public interface IModel : IDisposable
{
    DeviceTensor<float> Compose(DeviceTensor<float> inputs);
    void Execute(CudaStream stream);
}