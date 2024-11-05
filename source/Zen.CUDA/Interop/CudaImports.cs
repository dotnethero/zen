// ReSharper disable InconsistentNaming

using System.Runtime.InteropServices;

namespace Zen.CUDA.Interop;

internal static unsafe class CudaImports
{
    private const string LibraryName = "libzen.dll";

    [DllImport(LibraryName, EntryPoint = "zenCreateGemm")]
    public static extern void zenCreateGemm(
        zenGemmHandle** handle,
        int m,
        int n,
        int k,
        float* a, int lda,
        float* b, int ldb,
        float* c, int ldc);
    
    [DllImport(LibraryName, EntryPoint = "zenCreateConv2d")]
    public static extern void zenCreateConv2d(
        zenConv2dHandle** handle,
        zenConv2dParams* parameters,
        float* input,
        float* filter,
        float* bias,
        float* output);

    [DllImport(LibraryName, EntryPoint = "zenExecuteGemm")]
    public static extern void zenExecuteGemm(zenGemmHandle* handle, cudaStream* stream = null);
    
    [DllImport(LibraryName, EntryPoint = "zenExecuteConv2d")]
    public static extern void zenExecuteConv2d(zenConv2dHandle* handle, cudaStream* stream = null);

    [DllImport(LibraryName, EntryPoint = "zenDestroyGemm")]
    public static extern void zenDestroyGemm(zenGemmHandle* handle);
    
    [DllImport(LibraryName, EntryPoint = "zenDestroyConv2d")]
    public static extern void zenDestroyConv2d(zenConv2dHandle* handle);
}

[StructLayout(LayoutKind.Sequential)]
internal struct zenGemmHandle;

[StructLayout(LayoutKind.Sequential)]
internal struct zenConv2dHandle;

[StructLayout(LayoutKind.Sequential)]
internal struct zenConv2dParams
{
    public int input_n;
    public int input_h;
    public int input_w;
    public int input_c;
    public int filter_c;
    public int filter_h;
    public int filter_w;
    public int padding_h;
    public int padding_w;
    public int stride_h;
    public int stride_w;
    public int dilation_h;
    public int dilation_w;
    public int output_h;
    public int output_w;
}
