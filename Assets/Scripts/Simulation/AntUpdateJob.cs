using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

[BurstCompile]
public struct AntUpdateJob : IJobParallelFor
{
    public NativeArray<AntData> antData;

    [ReadOnly] public NativeArray<Color32> cpuMirror;
    [ReadOnly] public NativeArray<byte> obstacleMask;
    [ReadOnly] public NativeArray<float2> foodPositions;

    public float2 colonyPos;
    public float sensorAngle;
    public float sensorDist;
    public float maxTurnSpeed;
    public float randomPerturbation;
    public float deltaTime;
    public int textureSize;
    public float2 worldOrigin;
    public float2 worldSize;
    public float foodDetectionRadius;
    public float colonyDetectionRadius;

    public void Execute(int index)
    {
        var ant = antData[index];
        int ch = ant.state == 0 ? 0 : 1;

        float2 posL = SensorWorldPos(in ant, +sensorAngle, sensorDist);
        float2 posF = SensorWorldPos(in ant, 0f, sensorDist);
        float2 posR = SensorWorldPos(in ant, -sensorAngle, sensorDist);

        float sL = SamplePheromone(posL, ch);
        float sF = SamplePheromone(posF, ch);
        float sR = SamplePheromone(posR, ch);

        if (ObstacleAt(posL)) sL = -1f;
        if (ObstacleAt(posF)) sF = -1f;
        if (ObstacleAt(posR)) sR = -1f;

        float desiredTurn;
        if (sF >= sL && sF >= sR)
            desiredTurn = 0f;
        else if (sL > sR)
            desiredTurn = +maxTurnSpeed * deltaTime;
        else if (sR > sL)
            desiredTurn = -maxTurnSpeed * deltaTime;
        else
            desiredTurn = ant.rng.NextFloat(-1f, 1f) * randomPerturbation;

        float jitter = ant.rng.NextFloat(-1f, 1f) * randomPerturbation;
        float maxDelta = maxTurnSpeed * deltaTime;
        ant.angle += math.clamp(desiredTurn + jitter, -maxDelta, +maxDelta);

        ant.position += new float2(math.cos(ant.angle), math.sin(ant.angle)) * ant.speed * deltaTime;

        CheckStateTransitions(ref ant);

        antData[index] = ant;
    }

    void CheckStateTransitions(ref AntData ant)
    {
        if (ant.state == 0)
        {
            for (int i = 0; i < foodPositions.Length; i++)
            {
                if (math.distance(ant.position, foodPositions[i]) < foodDetectionRadius)
                {
                    ant.state = 1;
                    break;
                }
            }
        }
        else
        {
            if (math.distance(ant.position, colonyPos) < colonyDetectionRadius)
                ant.state = 0;
        }
    }

    float2 SensorWorldPos(in AntData ant, float angleOffset, float distance)
    {
        float a = ant.angle + angleOffset;
        return ant.position + new float2(math.cos(a), math.sin(a)) * distance;
    }

    int2 WorldToTexel(float2 worldPos)
    {
        float2 uv = (worldPos - worldOrigin) / worldSize;
        return (int2)math.floor(uv * textureSize);
    }

    float SamplePheromone(float2 worldPos, int channel)
    {
        int2 texel = WorldToTexel(worldPos);
        if (texel.x < 0 || texel.x >= textureSize || texel.y < 0 || texel.y >= textureSize)
            return 0f;

        Color32 c = cpuMirror[texel.y * textureSize + texel.x];
        return channel == 0 ? c.r / 255f : channel == 1 ? c.g / 255f : c.b / 255f;
    }

    bool ObstacleAt(float2 worldPos)
    {
        int2 texel = WorldToTexel(worldPos);
        if (texel.x < 0 || texel.x >= textureSize || texel.y < 0 || texel.y >= textureSize)
            return true;
        return obstacleMask[texel.y * textureSize + texel.x] < 128;
    }
}
