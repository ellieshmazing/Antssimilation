using UnityEngine;

[CreateAssetMenu(fileName = "PheromoneConfig", menuName = "Simulation/Pheromone Config")]
public class PheromoneConfig : ScriptableObject
{
    [Header("Ant Sensors")]
    [SerializeField] float sensorAngle = 45f;
    [SerializeField] float sensorDist = 2f;
    [SerializeField] float maxTurnSpeed = 180f;
    [SerializeField] float randomPerturbation = 0.1f;

    [Header("Movement")]
    [SerializeField] float antSpeed = 2f;

    [Header("Pheromone")]
    [SerializeField] float depositAmount = 1f;
    [SerializeField] float evaporationRate = 0.985f;
    [SerializeField] int diffusionKernelRadius = 1;
    [SerializeField] int pheromoneTextureSize = 512;

    [Header("Detection")]
    [SerializeField] float foodDetectionRadius = 0.5f;
    [SerializeField] float colonyDetectionRadius = 1f;

    public float SensorAngle => sensorAngle * Mathf.Deg2Rad;
    public float SensorDist => sensorDist;
    public float MaxTurnSpeed => maxTurnSpeed * Mathf.Deg2Rad;
    public float RandomPerturbation => randomPerturbation;
    public float AntSpeed => antSpeed;
    public float DepositAmount => depositAmount;
    public float EvaporationRate => evaporationRate;
    public int DiffusionKernelRadius => diffusionKernelRadius;
    public int PheromoneTextureSize => pheromoneTextureSize;
    public float FoodDetectionRadius => foodDetectionRadius;
    public float ColonyDetectionRadius => colonyDetectionRadius;
}
