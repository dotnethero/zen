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

    public Shape this[Range range] => new(
        Extents[range], 
        Strides[range]);

    public Shape Slice(ReadOnlySpan<Coord> coords, out int offset)
    {
        offset = 0;

        Span<int> extents = stackalloc int[Rank];
        Span<int> strides = stackalloc int[Rank];

        var rank = 0;
        
        for (var i = 0; i < Rank; ++i)
        {
            if (i < coords.Length)
            {
                offset += Strides[i] * coords[i].Start.GetOffset(Rank);     

                var coord = coords[i];
                if (coord.IsIndex)
                    continue;
                
                var total = Extents[i];
                var start = coord.Start.GetOffset(total);
                var end   = coord.End.GetOffset(total);
                extents[rank] = end - start;
            }
            else
            {
                extents[rank] = Extents[i];
            }

            strides[rank] = Strides[i];
            rank++;
        }
        return new(
            extents[..rank], 
            strides[..rank]);
    }

    public Shape Permute(ReadOnlySpan<Axis> axis)
    {
        Span<int> extents = stackalloc int[Rank];
        Span<int> strides = stackalloc int[Rank];
        for (var i = 0; i < Rank; ++i)
        {
            extents[i] = Extents[axis[i].Index];
            strides[i] = Strides[axis[i].Index];
        }
        return new(extents, strides);
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
