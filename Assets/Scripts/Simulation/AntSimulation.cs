using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(100)]
public class AntSimulation : MonoBehaviour
{
    [Header("Config")]
    [SerializeField] PheromoneConfig config;
    [SerializeField] ComputeShader pheromoneCompute;

    [Header("World")]
    [SerializeField] float worldSize = 50f;
    [SerializeField] int antCount = 10000;
    [SerializeField] Transform colonyTransform;
    [SerializeField] Transform[] foodTransforms;

    [Header("Rendering")]
    [SerializeField] Mesh antMesh;
    [SerializeField] Material antMaterial;
    [SerializeField] float antScale = 0.1f;

    [Header("Obstacle")]
    [SerializeField] Texture2D obstacleMaskTexture;

    NativeArray<AntData> antData;
    NativeArray<Color32> cpuMirror;
    NativeArray<byte> obstacleData;
    NativeArray<float2> foodPositions;

    RenderTexture pheromoneRT;
    RenderTexture tempRT;

    ComputeBuffer antGpuBuffer;
    ComputeBuffer gaussianWeightsBuffer;

    JobHandle antJobHandle;
    bool cpuMirrorReady;

    int depositKernel;
    int diffuseHKernel;
    int diffuseVKernel;

    Matrix4x4[] matrices;
    Vector4[] antDataVectors;
    Matrix4x4[] batchMatrices;
    Vector4[] batchVectors;
    MaterialPropertyBlock mpb;

    struct AntGpuData
    {
        public Vector2 position;
        public int state;
    }

    AntGpuData[] antGpuArray;

    void Start()
    {
        int texSize = config.PheromoneTextureSize;

        pheromoneRT = CreatePheromoneRT(texSize);
        tempRT = CreatePheromoneRT(texSize);

        cpuMirror = new NativeArray<Color32>(texSize * texSize, Allocator.Persistent);

        InitializeObstacleMask(texSize);
        InitializeAnts();
        RebuildFoodPositions();

        antGpuArray = new AntGpuData[antCount];
        antGpuBuffer = new ComputeBuffer(antCount, 12); // float2 + int = 12 bytes

        float[] weights = ComputeGaussianWeights(config.DiffusionKernelRadius);
        gaussianWeightsBuffer = new ComputeBuffer(weights.Length, sizeof(float));
        gaussianWeightsBuffer.SetData(weights);

        depositKernel = pheromoneCompute.FindKernel("DepositPass");
        diffuseHKernel = pheromoneCompute.FindKernel("DiffuseHorizontal");
        diffuseVKernel = pheromoneCompute.FindKernel("DiffuseVerticalAndEvaporate");

        BindComputeResources(texSize);

        matrices = new Matrix4x4[antCount];
        antDataVectors = new Vector4[antCount];
        batchMatrices = new Matrix4x4[1023];
        batchVectors = new Vector4[1023];
        mpb = new MaterialPropertyBlock();
    }

    void Update()
    {
        int texSize = config.PheromoneTextureSize;
        float2 worldOrigin = new float2(-worldSize * 0.5f, -worldSize * 0.5f);
        float2 worldSizeVec = new float2(worldSize, worldSize);
        float2 colonyPos = new float2(colonyTransform.position.x, colonyTransform.position.y);

        // 1. Complete previous frame's job
        antJobHandle.Complete();

        // 2. Schedule ant update if CPU mirror is ready
        if (cpuMirrorReady)
        {
            RebuildFoodPositions();

            var job = new AntUpdateJob
            {
                antData = antData,
                cpuMirror = cpuMirror,
                obstacleMask = obstacleData,
                foodPositions = foodPositions,
                colonyPos = colonyPos,
                sensorAngle = config.SensorAngle,
                sensorDist = config.SensorDist,
                maxTurnSpeed = config.MaxTurnSpeed,
                randomPerturbation = config.RandomPerturbation,
                deltaTime = Time.deltaTime,
                textureSize = texSize,
                worldOrigin = worldOrigin,
                worldSize = worldSizeVec,
                foodDetectionRadius = config.FoodDetectionRadius,
                colonyDetectionRadius = config.ColonyDetectionRadius
            };

            antJobHandle = job.Schedule(antCount, 64);
        }

        // 3. Must finish before deposit pass
        antJobHandle.Complete();

        // 4. Upload ant data to GPU and dispatch deposit
        UploadAntDataToGpu();
        pheromoneCompute.SetFloat("_DeltaTime", Time.deltaTime);
        pheromoneCompute.SetFloat("_EvaporationRate", config.EvaporationRate);
        pheromoneCompute.SetFloat("_DepositAmount", config.DepositAmount);
        pheromoneCompute.Dispatch(depositKernel, Mathf.CeilToInt(antCount / 64f), 1, 1);

        // 5. Diffuse and evaporate
        int threadGroups = Mathf.CeilToInt(texSize / 8f);
        pheromoneCompute.Dispatch(diffuseHKernel, threadGroups, threadGroups, 1);
        pheromoneCompute.Dispatch(diffuseVKernel, threadGroups, threadGroups, 1);

        // 6. Request async readback
        AsyncGPUReadback.Request(pheromoneRT, 0, TextureFormat.RGBA32, OnReadbackComplete);

        // 7. Render ants
        RenderAnts();
    }

    void OnReadbackComplete(AsyncGPUReadbackRequest request)
    {
        if (request.hasError) return;
        cpuMirror.CopyFrom(request.GetData<Color32>());
        cpuMirrorReady = true;
    }

    void OnDestroy()
    {
        antJobHandle.Complete();

        if (antData.IsCreated) antData.Dispose();
        if (cpuMirror.IsCreated) cpuMirror.Dispose();
        if (obstacleData.IsCreated) obstacleData.Dispose();
        if (foodPositions.IsCreated) foodPositions.Dispose();

        if (pheromoneRT != null) pheromoneRT.Release();
        if (tempRT != null) tempRT.Release();

        antGpuBuffer?.Release();
        gaussianWeightsBuffer?.Release();
    }

    // --- Initialization ---

    RenderTexture CreatePheromoneRT(int size)
    {
        var rt = new RenderTexture(size, size, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        rt.Create();
        return rt;
    }

    void InitializeObstacleMask(int texSize)
    {
        if (obstacleMaskTexture != null)
        {
            Color32[] pixels = obstacleMaskTexture.GetPixels32();
            obstacleData = new NativeArray<byte>(pixels.Length, Allocator.Persistent);
            for (int i = 0; i < pixels.Length; i++)
                obstacleData[i] = pixels[i].r;
        }
        else
        {
            // No obstacle mask — all passable
            obstacleData = new NativeArray<byte>(texSize * texSize, Allocator.Persistent);
            for (int i = 0; i < obstacleData.Length; i++)
                obstacleData[i] = 255;
        }
    }

    void InitializeAnts()
    {
        antData = new NativeArray<AntData>(antCount, Allocator.Persistent);
        float2 colonyPos = new float2(colonyTransform.position.x, colonyTransform.position.y);

        for (int i = 0; i < antCount; i++)
        {
            var rng = new Unity.Mathematics.Random((uint)(i * 1234567u + 1u));
            antData[i] = new AntData
            {
                position = colonyPos,
                angle = rng.NextFloat(0f, math.PI * 2f),
                speed = config.AntSpeed,
                state = 0,
                rng = rng
            };
        }
    }

    void RebuildFoodPositions()
    {
        int count = foodTransforms != null ? foodTransforms.Length : 0;
        if (!foodPositions.IsCreated || foodPositions.Length != count)
        {
            if (foodPositions.IsCreated) foodPositions.Dispose();
            foodPositions = new NativeArray<float2>(count, Allocator.Persistent);
        }
        for (int i = 0; i < count; i++)
        {
            Vector3 pos = foodTransforms[i].position;
            foodPositions[i] = new float2(pos.x, pos.y);
        }
    }

    // --- GPU ---

    void BindComputeResources(int texSize)
    {
        float2 worldOrigin = new float2(-worldSize * 0.5f, -worldSize * 0.5f);
        float2 worldSizeVec = new float2(worldSize, worldSize);

        pheromoneCompute.SetInt("_TextureSize", texSize);
        pheromoneCompute.SetInt("_KernelRadius", config.DiffusionKernelRadius);
        pheromoneCompute.SetInt("_AntCount", antCount);
        pheromoneCompute.SetFloats("_WorldOrigin", worldOrigin.x, worldOrigin.y);
        pheromoneCompute.SetFloats("_WorldSize", worldSizeVec.x, worldSizeVec.y);

        // Deposit kernel
        pheromoneCompute.SetTexture(depositKernel, "_PheromoneTexture", pheromoneRT);
        pheromoneCompute.SetBuffer(depositKernel, "_AntBuffer", antGpuBuffer);

        // Horizontal blur kernel
        pheromoneCompute.SetTexture(diffuseHKernel, "_PheromoneTexture", pheromoneRT);
        pheromoneCompute.SetTexture(diffuseHKernel, "_TempTexture", tempRT);
        pheromoneCompute.SetBuffer(diffuseHKernel, "_GaussianWeights", gaussianWeightsBuffer);

        // Vertical blur + evaporate kernel
        pheromoneCompute.SetTexture(diffuseVKernel, "_TempTexture", tempRT);
        pheromoneCompute.SetTexture(diffuseVKernel, "_PheromoneTexture", pheromoneRT);
        pheromoneCompute.SetBuffer(diffuseVKernel, "_GaussianWeights", gaussianWeightsBuffer);

        // Obstacle mask (shared across blur kernels)
        if (obstacleMaskTexture != null)
        {
            pheromoneCompute.SetTexture(diffuseHKernel, "_ObstacleMask", obstacleMaskTexture);
            pheromoneCompute.SetTexture(diffuseVKernel, "_ObstacleMask", obstacleMaskTexture);
        }
        else
        {
            // Create a 1x1 white texture as fallback
            var fallback = new Texture2D(1, 1, TextureFormat.R8, false);
            fallback.SetPixel(0, 0, Color.white);
            fallback.Apply();
            pheromoneCompute.SetTexture(diffuseHKernel, "_ObstacleMask", fallback);
            pheromoneCompute.SetTexture(diffuseVKernel, "_ObstacleMask", fallback);
        }
    }

    void UploadAntDataToGpu()
    {
        for (int i = 0; i < antCount; i++)
        {
            AntData ant = antData[i];
            antGpuArray[i] = new AntGpuData
            {
                position = new Vector2(ant.position.x, ant.position.y),
                state = ant.state
            };
        }
        antGpuBuffer.SetData(antGpuArray);
    }

    // --- Rendering ---

    void RenderAnts()
    {
        if (antMesh == null || antMaterial == null) return;

        for (int i = 0; i < antCount; i++)
        {
            AntData ant = antData[i];
            matrices[i] = Matrix4x4.TRS(
                new Vector3(ant.position.x, ant.position.y, 0f),
                Quaternion.Euler(0f, 0f, ant.angle * Mathf.Rad2Deg),
                Vector3.one * antScale
            );
            antDataVectors[i] = new Vector4(ant.position.x, ant.position.y, ant.angle, ant.state);
        }

        int batchStart = 0;
        while (batchStart < antCount)
        {
            int batchSize = Mathf.Min(1023, antCount - batchStart);
            Array.Copy(matrices, batchStart, batchMatrices, 0, batchSize);
            Array.Copy(antDataVectors, batchStart, batchVectors, 0, batchSize);

            mpb.SetVectorArray("_AntData", batchVectors);
            Graphics.DrawMeshInstanced(antMesh, 0, antMaterial, batchMatrices, batchSize, mpb);

            batchStart += 1023;
        }
    }

    // --- Utility ---

    static float[] ComputeGaussianWeights(int radius)
    {
        int size = 2 * radius + 1;
        float[] weights = new float[size];
        float sigma = Mathf.Max(radius * 0.5f, 0.5f);
        float sum = 0f;

        for (int i = 0; i < size; i++)
        {
            float x = i - radius;
            weights[i] = Mathf.Exp(-0.5f * x * x / (sigma * sigma));
            sum += weights[i];
        }

        for (int i = 0; i < size; i++)
            weights[i] /= sum;

        return weights;
    }
}
