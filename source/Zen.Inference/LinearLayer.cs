using Zen.CUDA;
using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;
using static Zen.CUDA.Interop.CudaImports;

namespace Zen.Inference;

public unsafe class LinearLayer : IModel
{
    internal zenGemmHandle* Handle;

    public int InputSize { get; }
    public int OutputSize { get; }

    public DeviceTensor<float>? Weights { get; private set; }
    public DeviceTensor<float>? Outputs { get; private set; }

    public LinearLayer(int inputs, int outputs)
    {
        InputSize = inputs;
        OutputSize = outputs;
    }

    public DeviceTensor<float> Compose(DeviceTensor<float> inputs)
    {
        var batchSize = inputs.Shape.Extents[0];
        var inputSize = inputs.Shape.Extents[1];
        
        if (inputSize != InputSize)
            throw new InvalidOperationException("Contraction dimensions are not equal");

        Weights = DeviceTensor.Allocate<float>([OutputSize, InputSize]); // inputs is contracted dimension
        Outputs = DeviceTensor.Allocate<float>([batchSize, OutputSize]);

        var a = inputs;
        var b = Weights;
        var c = Outputs;
        
        zenGemmHandle* handle;
        zenCreateGemm(
            &handle,
            m: a.Shape.Extents[0],
            n: b.Shape.Extents[0],
            k: a.Shape.Extents[1],
            a.Pointer, lda: a.Shape.Strides[0],
            b.Pointer, ldb: b.Shape.Strides[0],
            c.Pointer, ldc: c.Shape.Strides[0]);

        Handle = handle;

        return Outputs;
    }

    public void Execute(CudaStream stream)
    {
        if (Handle is null)
            throw new InvalidOperationException("Operation is not ready");
            
        zenExecuteGemm(Handle, stream.Pointer);
    }

    public void Dispose()
    {
        zenDestroyGemm(Handle);
        Weights?.Dispose();
        Outputs?.Dispose();
    }
}