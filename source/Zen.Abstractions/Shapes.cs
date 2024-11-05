namespace Zen;

internal static class Shapes
{
    public static unsafe int GetSize(ReadOnlySpan<int> extents)
    {
        var dims = extents.Length;
        if (dims == 0)
            return 1;
        
        var result = 1;

        fixed (int* ex = extents)
        {
            for (var i = dims - 1; i >= 0; --i)
            {
                result *= ex[i];
            }
        }

        return result;
    }
    
    public static unsafe int[] GetStrides(ReadOnlySpan<int> extents, Layout layout)
    {
        var dims = extents.Length;
        if (dims == 0)
            return [];

        var strides = new int[dims];

        fixed (int* ex = extents, st = strides)
        {
            if (layout == Layout.Left)
            {
                st[0] = 1;
                
                for (var i = 1; i < dims; ++i)
                {
                    st[i] = st[i - 1] * ex[i - 1];
                }
            }
            
            if (layout == Layout.Right)
            {
                st[dims - 1] = 1;

                for (var i = dims - 2; i >= 0; --i)
                {
                    st[i] = st[i + 1] * ex[i + 1];
                }
            }
        }

        return strides;
    }
}
