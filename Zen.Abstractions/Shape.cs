using System.Collections;
using System.Runtime.CompilerServices;

namespace Zen;

[CollectionBuilder(typeof(Shape), nameof(Create))]
public readonly struct Shape : IEnumerable<int>
{
    public static readonly Shape Scalar = [];

    public readonly int Rank = 0;
    public readonly int Size = 1;
    public readonly int[] Extents = [];
    public readonly int[] Strides = [];

    public static Shape Create(ReadOnlySpan<int> extents) => new(extents, Layout.Right);
    public static Shape Create(ReadOnlySpan<int> extents, Layout layout) => new(extents, layout);
    public static Shape Create(ReadOnlySpan<int> extents, ReadOnlySpan<int> strides) => new(extents, strides);

    private Shape(ReadOnlySpan<int> extents, Layout layout)
    {
        Rank = extents.Length;
        Extents = extents.ToArray();
        Strides = Shapes.GetStrides(extents, layout);
        Size = Shapes.GetSize(extents);
    }

    private Shape(ReadOnlySpan<int> extents, ReadOnlySpan<int> strides)
    {
        Rank = extents.Length;
        Extents = extents.ToArray();
        Strides = strides.ToArray();
        Size = Shapes.GetSize(extents);
    }

    IEnumerator<int> IEnumerable<int>.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();
}
