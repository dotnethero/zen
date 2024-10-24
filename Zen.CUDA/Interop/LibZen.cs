// ReSharper disable InconsistentNaming

using System.Runtime.InteropServices;

namespace Zen.CUDA.Interop;

public static unsafe partial class LibZen
{
    private const string LibraryName = "libzen.dll";
    
    [LibraryImport(LibraryName, EntryPoint = "zenCreateConv2dPlan")]
    public static partial void zenCreateConv2dPlan(
        zenConv2dPlan** plan,
        zenConv2dParams* parameters);

    [LibraryImport(LibraryName, EntryPoint = "zenGetConv2dWorkspaceSize")]
    public static partial void zenGetConv2dWorkspaceSize(
        zenConv2dPlan* plan,
        nuint* size);

    [LibraryImport(LibraryName, EntryPoint = "zenExecuteConv2d")]
    public static partial void zenExecuteConv2d(
        zenConv2dPlan* plan,
        float* input,
        float* filter,
        float* bias,
        float alpha,
        float beta,
        float* output,
        void* workspace,
        void* stream = null);
}

[StructLayout(LayoutKind.Sequential)]
public struct zenConv2dPlan
{
}

[StructLayout(LayoutKind.Sequential)]
public struct zenConv2dParams
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
