# Variables Reference

| Variable | Type | Location | Description | Behavior | Affects |
|---|---|---|---|---|---|
| `sensorAngle` | float (degrees) | PheromoneConfig | Angular offset of left/right sensors from forward | Lower = tighter path following; higher = wider lateral correction. 20–70° range. | AntUpdateJob steering |
| `sensorDist` | float | PheromoneConfig | World-space distance from ant to each sensor | Higher = smoother trails; lower = sharper responsiveness. 1.0–5.0 range. | AntUpdateJob steering |
| `maxTurnSpeed` | float (deg/sec) | PheromoneConfig | Maximum rotation rate per second | Lower = organic arcs; higher = snappy mechanical turns. 90–360 range. | AntUpdateJob steering |
| `randomPerturbation` | float (radians) | PheromoneConfig | Random jitter added per frame | Prevents freezing when sensors are equal. 0.0–0.5 range; higher = more chaotic exploration. | AntUpdateJob steering |
| `antSpeed` | float | PheromoneConfig | Base movement speed in world units/second | Directly scales ant velocity. 1.0–5.0 range. | AntUpdateJob movement, AntData.speed |
| `depositAmount` | float | PheromoneConfig | Pheromone deposited per ant per frame | Higher = stronger trails, faster colony consensus. 0.5–5.0 range. | PheromoneCompute DepositPass |
| `evaporationRate` | float | PheromoneConfig | Per-second decay multiplier applied as pow(rate, dt) | Lower = shorter trail memory; higher = persistent trails. 0.95–0.999 range. | PheromoneCompute DiffuseVerticalAndEvaporate |
| `diffusionKernelRadius` | int | PheromoneConfig | Pixel radius of Gaussian blur kernel | Higher = wider smell radius, blurrier trails. 1–4 range. | PheromoneCompute DiffuseHorizontal/Vertical |
| `pheromoneTextureSize` | int | PheromoneConfig | RenderTexture resolution (power of 2) | Higher = more precise trails at greater GPU cost. 256/512/1024. | AntSimulation pheromone field, AntUpdateJob sampling |
| `foodDetectionRadius` | float | PheromoneConfig | World-space proximity that triggers SEARCHING→CARRYING | Smaller = ants must get closer to food. 0.2–2.0 range. | AntUpdateJob state transitions |
| `colonyDetectionRadius` | float | PheromoneConfig | World-space proximity that triggers CARRYING→SEARCHING | Smaller = ants must get closer to colony. 0.5–3.0 range. | AntUpdateJob state transitions |
| `worldSize` | float | AntSimulation | Side length of the simulation world in world units | Defines world-to-texel mapping. Larger = lower pheromone resolution per unit. | AntSimulation, AntUpdateJob, PheromoneCompute |
| `antCount` | int | AntSimulation | Total number of simulated ants | More ants = denser trails but higher CPU/GPU cost. | AntSimulation, AntUpdateJob, PheromoneCompute |
| `antScale` | float | AntSimulation | Visual scale of each ant mesh instance | Purely cosmetic — scales the rendered quad. | AntSimulation rendering |
| `position` | float2 | AntData | Current world position of the ant | Updated each frame by steering + movement. | AntUpdateJob, PheromoneCompute deposit |
| `angle` | float | AntData | Current heading in radians | Updated each frame by steering logic. | AntUpdateJob, rendering |
| `speed` | float | AntData | Per-ant movement speed | Initialized from config; allows per-ant variation. | AntUpdateJob movement |
| `state` | int | AntData | 0 = SEARCHING, 1 = CARRYING | Determines which pheromone channel is followed and deposited. | AntUpdateJob steering + state transitions, PheromoneCompute deposit |
| `rng` | Random | AntData | Per-ant Burst-compatible RNG | Seeded uniquely per ant; provides jitter and tie-breaking. | AntUpdateJob steering |
| `foodTrailColor` | Color | AntSimulation | Additive color for the food-return pheromone (R channel) | Orange by default; determines how food trails look in vis mode. | PheromoneVis shader |
| `homeTrailColor` | Color | AntSimulation | Additive color for the home-search pheromone (G channel) | Cyan by default; determines how search trails look in vis mode. | PheromoneVis shader |
| `visBrightness` | float | AntSimulation | Multiplier applied to pheromone values before saturate in vis shader | Higher = trails visible at lower pheromone concentrations. 1.0–10.0 range. | PheromoneVis shader |
| `showPheromoneVis` | bool | AntSimulation | Toggles pheromone gradient overlay on/off (P key) | True = render the trail heatmap behind ants. | AntSimulation rendering |
