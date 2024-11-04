namespace Zen;

public readonly struct LogicalCoord
{
    public readonly Index Index;

    public LogicalCoord(Index index)
    {
        Index = index;
    }
    
    public static implicit operator LogicalCoord(int index) => new(index);
    public static implicit operator LogicalCoord(Index index) => new(index);

    public int GetOffset(int extent, int stride) => Index.GetOffset(extent) * stride;
}