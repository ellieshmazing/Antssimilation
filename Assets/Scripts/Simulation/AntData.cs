using Unity.Mathematics;

public struct AntData
{
    public float2 position;
    public float angle;
    public float speed;
    public int state; // 0 = SEARCHING, 1 = CARRYING
    public Random rng;
}
