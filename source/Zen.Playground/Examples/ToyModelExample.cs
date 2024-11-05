using Zen.CUDA;
using Zen.CUDA.Wrappers;
using Zen.Inference;

namespace Zen.Playground.Examples;

public static class ToyModelExample
{
    public static void Run()
    {
        const int batchSize = 1;
        const int inputSize = 28 * 28;
        const int hiddenSize1 = 256;
        const int hiddenSize2 = 64;
        const int outputSize = 10;

        LinearLayer[] layers =
        [
            new(inputSize, hiddenSize1),
            new(hiddenSize1, hiddenSize2),
            new(hiddenSize2, outputSize),
        ];
        
        // allocate outputs and prepare kernels

        var inputs = DeviceTensor.Allocate<float>([batchSize, inputSize]);
        var outputs = inputs;

        foreach (var layer in layers)
        {
            outputs = layer.Compose(outputs);
        }

        // capture first execution

        using var stream = new CudaStream();
        stream.BeginCapture();
        
        foreach (var layer in layers)
        {
            layer.Execute(stream);
        }

        using var graph = stream.EndCapture();
        using var graphInstance = graph.CreateInstance();

        // execute graph for desired iterations count
        
        using var start = new CudaEvent();
        using var stop = new CudaEvent();
        
        start.Record(stream);
        
        for (var epoch = 0; epoch < 10000; epoch++)
        {
            graphInstance.Launch(stream);
        }
        
        stop.Record(stream);
        stop.Synchronize();
        
        // record elapsed time and copy results
        
        var elapsed = CudaEvent.Elapsed(start, stop);
        var results = HostTensor.Allocate<float>(outputs.Shape);
        
        outputs.CopyTo(results, stream);
        stream.Synchronize();
        
        Utils.WriteLine(results);
        Console.WriteLine($"Device execution: {elapsed}");
        
        // Device execution: 00:00:01.1092039 for 10'000 iterations
    }
}