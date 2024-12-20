﻿using System.Collections;
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

    private Shape(int extent, int stride)
    {
        Rank = 1;
        Extents = [extent];
        Strides = [stride];
        Size = extent;
    }
    
    public Shape this[Index index] =>
        new(Extents[index], 
            Strides[index]);
    
    public Shape this[Range range] =>
        new(Extents[range].AsSpan(), 
            Strides[range].AsSpan());

    public Shape Prepend(int extent, int stride) => 
        new([extent, ..Extents.AsSpan()],
            [stride, ..Strides.AsSpan()]);

    public Shape Append(int extent, int stride) => 
        new([..Extents.AsSpan(), extent],
            [..Strides.AsSpan(), stride]);

    public static Shape Concat(Shape[] shapes)
    {
        var rank = shapes.Sum(x => x.Rank);
        
        Span<int> extents = stackalloc int[rank];
        Span<int> strides = stackalloc int[rank];

        var k = 0;
        
        for (var i = 0; i < shapes.Length; ++i)
        for (var j = 0; j < shapes[i].Rank; ++j)
        {
            extents[k] = shapes[i].Extents[j];
            strides[k] = shapes[i].Strides[j];
            ++k;
        }

        return new Shape(extents, strides);
    }

    public static Shape[] Split(Shape shape)
    {
        var shapes = new Shape[shape.Rank];
        
        for (var i = 0; i < shape.Rank; ++i)
        {
            shapes[i] = new(
                shape.Extents[i],
                shape.Strides[i]);
        }
        
        return shapes;
    }
    
    public Shape Replace(Axis axis, int extent, int stride)
    {
        ReadOnlySpan<int> extents = Extents.AsSpan();
        ReadOnlySpan<int> strides = Strides.AsSpan();
        return new(
            [..extents[..axis], extent, ..extents[(axis + 1)..]],
            [..strides[..axis], stride, ..strides[(axis + 1)..]]);
    }

    public Shape Reduce(Axis axis) => Replace(axis, 1, 0);

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

    public Shape Transpose(Axis axis1, Axis axis2)
    {
        Span<int> extents = [..Extents.AsSpan()];
        Span<int> strides = [..Strides.AsSpan()];
        extents[axis1] = Extents[axis2];
        extents[axis2] = Extents[axis1];
        strides[axis1] = Strides[axis2];
        strides[axis2] = Strides[axis1];
        return new(extents, strides);
    }

    private static void EnsureCoordinateIsValid(ReadOnlySpan<LogicalCoord> coords, Shape shape)
    {
        if (coords.Length != shape.Rank)
            throw new InvalidOperationException($"Can not apply {coords.Length} rank coordinate to {shape.Rank} rank shape");
    }
    
    public int GetOffset(params ReadOnlySpan<LogicalCoord> coords)
    {
        EnsureCoordinateIsValid(coords, this);
        
        ReadOnlySpan<int> extents = Extents.AsSpan();
        ReadOnlySpan<int> strides = Strides.AsSpan();
        
        var offset = 0;
        
        for (var i = 0; i < Rank; ++i)
        {
            var extent = extents[i];
            var stride = strides[i];
            offset += coords[i].GetOffset(extent, stride);
        }
        
        return offset;
    }

    public Shape Slice(ReadOnlySpan<LogicalRange> slice, out int offset)
    {
        ReadOnlySpan<int> originalExtents = Extents.AsSpan();
        ReadOnlySpan<int> originalStrides = Strides.AsSpan();

        offset = 0;

        Span<int> extents = stackalloc int[Rank];
        Span<int> strides = stackalloc int[Rank];

        var rank = 0;
        
        for (var i = 0; i < Rank; ++i)
        {
            var extent = originalExtents[i];
            var stride = originalStrides[i];

            if (i < slice.Length)
            {
                var range = slice[i];
                
                offset += range.GetOffset(extent, stride);     

                if (range.IsIndex)
                    continue;
                
                extents[rank] = range.GetSize(extent);
            }
            else
            {
                extents[rank] = extent;
            }

            strides[rank] = stride;
            rank++;
        }
        
        return new(
            extents[..rank], 
            strides[..rank]);
    }

    IEnumerator<int> IEnumerable<int>.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();

    public override string ToString() => $"({string.Join(",", Extents)}):({string.Join(",", Strides)})";
}