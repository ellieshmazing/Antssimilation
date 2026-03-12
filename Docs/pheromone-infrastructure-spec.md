# Pheromone Infrastructure Spec

## Goal

This system implements the foundational simulation loop for ~10,000 ants: a GPU-resident multi-channel pheromone field, a Burst-parallelized ant steering update, and a GPU instancing renderer. Each frame, a compute shader deposits ant pheromone, diffuses it via a separable Gaussian blur, then exponentially decays it. A `AsyncGPUReadback` bridge copies the result to a CPU `NativeArray<Color32>` mirror so Burst jobs can sample sensor positions without crossing the GPU boundary on the hot path. Ants steer toward the pheromone channel appropriate to their state (searching follows food trail R; carrying follows home trail G), with three forward sensors approximating a gradient sample. State transitions (searching↔carrying) are checked inside the same job via proximity to food sources and the colony. This system has no MonoBehaviour per ant; ant state lives entirely in `NativeArray<AntData>`.

---

## Architecture Overview

```
Every Update():
  1. antJob.Complete()                          // finish previous frame's jobs

  2. if cpuMirrorReady:
       Schedule AntUpdateJob(antData, cpuMirror, foodPositions, colonyPos)
       → IJobParallelFor over all ants:
           - sample 3 sensors from cpuMirror
           - compute turn direction, clamp to maxTurnSpeed
           - check obstacle mask at sensors
           - update ant.position, ant.angle
           - check state transitions (food/colony proximity)
       store JobHandle

  3. antJob.Complete()                          // must finish before deposit pass

  4. DispatchDepositPass(antData)               // GPU: ants write pheromone at positions
  5. DispatchDiffuseEvaporatePass()             // GPU: horizontal blur → vertical blur → decay
  6. AsyncGPUReadback.Request(pheromoneRT,      // request next mirror update
                               OnReadbackComplete)
  7. AntRenderer.DrawInstanced(antData)         // batched DrawMeshInstanced

On OnReadbackComplete(request):
  if !request.hasError:
    cpuMirror ← request.GetData<Color32>()
    cpuMirrorReady = true
```

Pheromone channels: **R = food trail** (deposited by carrying ants), **G = home trail** (deposited by searching ants), **B = alarm** (reserved).

---

## Behaviors

### Three-Sensor Pheromone Steering

**Concept:** Three point samples at fixed angular offsets approximate a directional gradient without computing a true spatial derivative — O(3) per ant regardless of texture resolution.

**Role:** Biases each ant's heading toward the pheromone channel it's currently tracking, producing trail-following and trail-reinforcement behavior.

**Logic:**
```
channel = (ant.state == SEARCHING) ? R : G

sensorAngle_L = ant.angle + sensorAngle
sensorAngle_F = ant.angle
sensorAngle_R = ant.angle - sensorAngle

sL = SamplePheromone(cpuMirror, SensorWorldPos(ant, +sensorAngle, sensorDist), channel)
sF = SamplePheromone(cpuMirror, SensorWorldPos(ant, 0,            sensorDist), channel)
sR = SamplePheromone(cpuMirror, SensorWorldPos(ant, -sensorAngle, sensorDist), channel)

// Zero out sensors that land on obstacles
if ObstacleAt(SensorWorldPos(ant, +sensorAngle, sensorDist)): sL = -1
if ObstacleAt(SensorWorldPos(ant,  0,           sensorDist)): sF = -1
if ObstacleAt(SensorWorldPos(ant, -sensorAngle, sensorDist)): sR = -1

if sF >= sL and sF >= sR:
    desiredTurn = 0                                         // forward wins — hold course
elif sL > sR:
    desiredTurn = +maxTurnSpeed * deltaTime
elif sR > sL:
    desiredTurn = -maxTurnSpeed * deltaTime
else:
    desiredTurn = ant.rng.NextFloat(-1, 1) * randomPerturbation  // symmetric — randomize

jitter = ant.rng.NextFloat(-1, 1) * randomPerturbation
ant.angle += clamp(desiredTurn + jitter, -maxTurnSpeed * deltaTime, +maxTurnSpeed * deltaTime)

ant.position += float2(cos(ant.angle), sin(ant.angle)) * ant.speed * deltaTime
```

---

### Boundary and Obstacle Avoidance

**Concept:** The obstacle mask texture is sampled at each sensor position; obstacle hits set sensor value to -1, making those directions strongly undesirable without a separate steering pass.

**Role:** Prevents ants from walking through walls or off the map edge.

**Logic:**
```
// Obstacle mask is a CPU Texture2D (static, uploaded once)
// 1.0 = passable, 0.0 = obstacle
// Map borders are pre-set to 0.0 in the asset

ObstacleAt(worldPos):
    texel = WorldToTexel(worldPos)
    if texel out of bounds: return true
    return obstacleMask.GetPixel(texel.x, texel.y).r < 0.5

// In diffusion pass (compute shader), after each blur sub-pass:
//   pheromoneTexture[id] *= obstacleMaskTexture[id]
// This prevents pheromone from accumulating inside walls.
```

---

### Pheromone Deposit

**Concept:** Each ant scatters a fixed-point write into `RWTexture2D` at its texel position; multiple ants writing the same texel race non-atomically, which is acceptable since deposit drops are imperceptible at 10k ants.

**Role:** Encodes ant paths into the shared chemical field so other ants can follow them.

**Logic:**
```
// Compute shader kernel: DepositPass
// Dispatched with one thread per ant

[numthreads(64, 1, 1)]
void DepositPass(uint id):
    ant = antBuffer[id]
    texel = WorldToTexel(ant.position)
    if texel out of bounds: return

    if ant.state == SEARCHING:
        pheromoneTexture[texel].g += depositAmount   // deposit home trail
    else:  // CARRYING
        pheromoneTexture[texel].r += depositAmount   // deposit food trail
```

---

### Diffusion and Evaporation

**Concept:** A separable Gaussian blur (two 1D passes) spreads pheromone in O(w·h·k) rather than O(w·h·k²); evaporation is a per-pixel multiply applied as `pow(evaporationRate, deltaTime)` to stay framerate-independent.

**Role:** Gives pheromone trails spatial extent (ants can sense trails they don't intersect exactly) and finite lifetime (old trails fade, preventing the map from saturating).

**Logic:**
```
// Compute shader kernel: DiffuseHorizontal
// Pass 1: for each pixel, accumulate weighted horizontal neighbors
[numthreads(8, 8, 1)]
void DiffuseHorizontal(uint2 id):
    sum = 0
    for k in [-kernelRadius .. +kernelRadius]:
        sum += pheromoneTexture[id + int2(k, 0)] * gaussianWeight[k + kernelRadius]
    tempTexture[id] = sum * obstacleMask[id]      // mask after blur

// Compute shader kernel: DiffuseVertical + Evaporate
// Pass 2: vertical blur then decay in one pass
[numthreads(8, 8, 1)]
void DiffuseVerticalAndEvaporate(uint2 id):
    float decayThisFrame = pow(evaporationRate, deltaTime)
    sum = 0
    for k in [-kernelRadius .. +kernelRadius]:
        sum += tempTexture[id + int2(0, k)] * gaussianWeight[k + kernelRadius]
    pheromoneTexture[id] = sum * obstacleMask[id] * decayThisFrame
```

---

### State Transitions

**Concept:** Proximity checks inside the Burst job — comparing `ant.position` against known food source positions and colony center — trigger state changes that flip which pheromone channel the ant follows and deposits.

**Role:** Turns the steering system into a complete foraging loop: find food, carry it home, repeat.

**Logic:**
```
// Runs at end of AntUpdateJob.Execute(), after position update

if ant.state == SEARCHING:
    for each pos in foodPositions:
        if distance(ant.position, pos) < foodDetectionRadius:
            ant.state = CARRYING
            break

else:  // CARRYING
    if distance(ant.position, colonyPos) < colonyDetectionRadius:
        ant.state = SEARCHING
        // (food delivery logic will be handled by a higher-level system)
```

---

### GPU Instanced Rendering

**Concept:** `Graphics.DrawMeshInstanced` submits all ant transforms in a single GPU command per 1023-ant batch, bypassing Unity's per-object render overhead entirely.

**Role:** Renders 10,000 ants without 10,000 draw calls.

**Logic:**
```
// Runs after antJob.Complete()

positionAngles = Vector4[antCount]    // x,y = position; z = angle; w = state (for shader)
for i in [0..antCount):
    positionAngles[i] = float4(antData[i].position, antData[i].angle, antData[i].state)

mpb.SetVectorArray("_AntData", positionAngles)

batchStart = 0
while batchStart < antCount:
    batchSize = min(1023, antCount - batchStart)
    Graphics.DrawMeshInstanced(antMesh, 0, antMaterial, matrices[batchStart..], batchSize, mpb)
    batchStart += batchSize
```

---

## Function Designs

### `SamplePheromone(mirror: NativeArray<Color32>, worldPos: float2, channel: int) → float`
Samples a single channel value from the CPU texture mirror at a world position.

**Parameters:**
- `mirror` — flattened row-major Color32 array from the last AsyncGPUReadback
- `worldPos` — world-space position to sample
- `channel` — 0=R (food), 1=G (home), 2=B (alarm)

**Returns:** Pheromone concentration in [0, 1]; returns 0.0 if worldPos is out of bounds.

```
texel = WorldToTexel(worldPos)
if texel.x < 0 or texel.x >= textureSize or texel.y < 0 or texel.y >= textureSize:
    return 0.0
c = mirror[texel.y * textureSize + texel.x]
return channel == 0 ? c.r/255f : channel == 1 ? c.g/255f : c.b/255f
```

---

### `SensorWorldPos(ant: AntData, angleOffset: float, distance: float) → float2`
Computes the world position of a single sensor.

**Parameters:**
- `ant` — the ant whose sensors are being computed
- `angleOffset` — signed radians offset from `ant.angle` (+left, −right)
- `distance` — world-space distance from `ant.position` to sensor

**Returns:** World position of the sensor.

```
a = ant.angle + angleOffset
return ant.position + float2(cos(a), sin(a)) * distance
```

---

### `WorldToTexel(worldPos: float2) → int2`
Converts a world-space position to an integer texel coordinate.

**Parameters:**
- `worldPos` — position in world space

**Returns:** Texel coordinate; may be out of `[0, textureSize-1]` — callers must clamp or bounds-check.

```
// worldOrigin and worldSize are uniform values set once (e.g., world is [-worldSize/2, +worldSize/2])
uv = (worldPos - worldOrigin) / worldSize       // [0, 1]
return int2(floor(uv * textureSize))
```

---

### `AntUpdateJob.Execute(index: int) → void`
Core IJobParallelFor body — updates one ant's angle, position, and state for the current frame.

**Side effects:** Mutates `antData[index]` (angle, position, state, rng).

**Must run after:** `antJob.Complete()` from the previous frame. Must complete before `DispatchDepositPass`.

```
ant = antData[index]

// Sensor sample
sL = SamplePheromone(cpuMirror, SensorWorldPos(ant, +sensorAngle, sensorDist), StateChannel(ant.state))
sF = SamplePheromone(cpuMirror, SensorWorldPos(ant,  0,           sensorDist), StateChannel(ant.state))
sR = SamplePheromone(cpuMirror, SensorWorldPos(ant, -sensorAngle, sensorDist), StateChannel(ant.state))

if ObstacleAt(SensorWorldPos(ant, +sensorAngle, sensorDist)): sL = -1
if ObstacleAt(SensorWorldPos(ant,  0,           sensorDist)): sF = -1
if ObstacleAt(SensorWorldPos(ant, -sensorAngle, sensorDist)): sR = -1

// Steering
if sF >= sL and sF >= sR:   desiredTurn = 0
elif sL > sR:               desiredTurn = +maxTurnSpeed * deltaTime
elif sR > sL:               desiredTurn = -maxTurnSpeed * deltaTime
else:                       desiredTurn = ant.rng.NextFloat(-1,1) * randomPerturbation

jitter = ant.rng.NextFloat(-1, 1) * randomPerturbation
ant.angle += clamp(desiredTurn + jitter, -maxTurnSpeed * deltaTime, +maxTurnSpeed * deltaTime)
ant.position += float2(cos(ant.angle), sin(ant.angle)) * ant.speed * deltaTime

// State transitions
CheckStateTransitions(ref ant)    // see State Transitions behavior

antData[index] = ant
```

---

### `OnReadbackComplete(request: AsyncGPUReadbackRequest) → void`
Callback invoked on the main thread when the GPU readback finishes.

**Side effects:** Replaces `cpuMirror` contents; sets `cpuMirrorReady = true`.

```
if request.hasError:
    return   // keep using stale mirror; do not set ready flag

cpuMirror.CopyFrom(request.GetData<Color32>())
cpuMirrorReady = true
```

> `cpuMirror` must be a persistent `NativeArray<Color32>` allocated at startup (size = textureSize²). Do not allocate inside this callback.

---

## Modifiable Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `sensorAngle` | float (degrees) | 45 | Angular offset of left/right sensors from forward. Controls how wide the ant "looks". Try 20–70°; lower = tighter path following, higher = wider lateral correction. |
| `sensorDist` | float | 2.0 | World-space distance from ant to each sensor. Controls how far ahead the ant looks. Try 1.0–5.0; higher = smoother trails, lower = sharper responsiveness. |
| `maxTurnSpeed` | float (deg/sec) | 180 | Maximum rotation rate. Clamps both deliberate steering and jitter. Try 90–360; lower = organic arcs, higher = snappy but mechanical. |
| `randomPerturbation` | float (radians) | 0.1 | Random jitter added per frame. Prevents freezing when sensors are equal. Try 0.0–0.5; higher = more chaotic exploration. |
| `antSpeed` | float | 2.0 | Base movement speed in world units/second. Try 1.0–5.0. |
| `depositAmount` | float | 1.0 | Pheromone deposited per ant per frame. Try 0.5–5.0; higher = stronger trails, faster colony consensus. |
| `evaporationRate` | float | 0.985 | Per-second decay multiplier applied as `pow(rate, deltaTime)`. Try 0.95–0.999; lower = shorter trail memory. |
| `diffusionKernelRadius` | int | 1 | Pixel radius of Gaussian blur (1 = 3×3, 2 = 5×5, 3 = 7×7). Try 1–4; higher = wider smell radius, blurrier trails. |
| `pheromoneTextureSize` | int | 512 | RenderTexture resolution (power of 2). Higher = more precise trails at greater GPU cost. 512 is the practical default for 10k ants. |
| `foodDetectionRadius` | float | 0.5 | World-space proximity to food that triggers SEARCHING→CARRYING. Try 0.2–2.0. |
| `colonyDetectionRadius` | float | 1.0 | World-space proximity to colony that triggers CARRYING→SEARCHING. Try 0.5–3.0. |

---

## Implementation Notes

**Readback latency (1–2 frames):** Ants always steer on the previous frame's pheromone field. This is physically realistic and not a bug. Do not try to synchronize it — the `AsyncGPUReadback` pipeline exists precisely to avoid stalling the GPU.

**NativeArray layout for readback:** Declare `cpuMirror` as `NativeArray<Color32>(textureSize * textureSize, Allocator.Persistent)`. Index as `mirror[y * textureSize + x]`. Request readback with `AsyncGPUReadback.Request(pheromoneRT, 0, TextureFormat.RGBA32, OnReadbackComplete)`.

**Burst-compatible RNG:** `UnityEngine.Random` and `System.Random` are not Burst-compatible. Store `Unity.Mathematics.Random rng` inside `AntData`. Seed each ant at init with `new Random((uint)(index * 1234567u + 1u))` — the `+1` prevents the illegal seed value of 0.

**Framerate-independent evaporation:** Use `pow(evaporationRate, deltaTime)` in the compute shader, not a fixed multiplier. Pass `deltaTime` as a shader uniform each frame. A fixed `0.985` multiplier will evaporate ~5× faster at 300 fps than at 60 fps.

**Deposit race condition:** In the deposit compute shader, multiple ants writing the same texel race non-atomically on `RWTexture2D<float4>`. This is intentional — the visual artifact (a random subset of deposits is silently dropped in rare collisions) is imperceptible at 10k ants. If precision becomes a problem later, switch to `RWTexture2D<int4>` with fixed-point encoding (e.g., `InterlockedAdd(tex[id].r, (int)(amount * 1000))`).

**DrawMeshInstanced 1023 limit:** Unity's limit is 1023 instances per call. For 10k ants: `for (int i = 0; i < antCount; i += 1023) DrawMeshInstanced(mesh, 0, mat, matrices, min(1023, antCount-i), mpb)`. Pack position + angle into `matrices` as `Matrix4x4.TRS` or pass raw data via `mpb.SetVectorArray`.

**JobHandle lifecycle:** The safe pattern: at the top of `Update()`, call `antJobHandle.Complete()`. Then read `antData` for deposit and render. Then schedule the next job and store the handle. Never schedule while the previous handle is incomplete.

**Obstacle mask upload:** The obstacle mask is a static `Texture2D` (authored in editor or generated at runtime). Upload it to the compute shader once via `computeShader.SetTexture(kernel, "_ObstacleMask", obstacleMask)`. Re-upload only if the map changes.

**Compute shader texture format:** Use `RenderTextureFormat.ARGBFloat` during development (full 32-bit precision). Switch to `ARGBHalf` before shipping — halves GPU memory and bandwidth with negligible quality loss for pheromone values.

**`StateChannel` helper:** Searching ants (state=0) sample R channel (food trail); carrying ants (state=1) sample G channel (home trail). Inline this as a ternary in the job rather than a function call to keep Burst happy: `int ch = ant.state == 0 ? 0 : 1`.
