using System.Collections;
using System.Runtime.CompilerServices;

namespace Zen;

[CollectionBuilder(typeof(Shape), nameof(Create))]
public readonly unsafe struct Shape : IEnumerable<int>
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

    public Shape this[Range range] => new(Extents[range], Strides[range]);

    public Shape Permute(Index[] axis) // TODO: Create permutation type
    {
        Span<int> extents = stackalloc int[Rank];
        Span<int> strides = stackalloc int[Rank];
        for (var i = 0; i < Rank; ++i)
        {
            extents[i] = Extents[axis[i]];
            strides[i] = Strides[axis[i]];
        }
        return Create(extents, strides);
    }

    IEnumerator<int> IEnumerable<int>.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();
    
    public override string ToString() => $"({string.Join(",", Extents)})";
}
