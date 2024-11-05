namespace Zen;

public readonly struct LogicalRange
{
    public readonly Index Start;
    public readonly Index End;
    public readonly bool IsIndex;

    public LogicalRange(Index index)
    {
        Start = index;
        End = index;
        IsIndex = true;
    }

    public LogicalRange(Range range)
    {
        Start = range.Start;
        End = range.End;
        IsIndex = false;
    }

    public static implicit operator LogicalRange(int index) => new(index);
    public static implicit operator LogicalRange(Index index) => new(index);
    public static implicit operator LogicalRange(Range index) => new(index);

    public int GetSize(int extent) => End.GetOffset(extent) - Start.GetOffset(extent);
    
    public int GetOffset(int extent, int stride) => Start.GetOffset(extent) * stride;
}