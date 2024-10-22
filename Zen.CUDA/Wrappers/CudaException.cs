using Zen.CUDA.Interop;

namespace Zen.CUDA.Wrappers;

internal sealed class CudaException(cudaError status) : Exception($"Operation is not completed: {status}");