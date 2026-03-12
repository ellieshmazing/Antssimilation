
---
Date: 2026-03-11
Topic: Ant colony simulation architecture
Concepts:
  - **Stigmergy**: Indirect coordination through environment modification — ants don't communicate directly, they write to and read from a shared pheromone field. The emergent colony intelligence (optimal pathfinding, load balancing) arises purely from local rules interacting with this shared state. It's a master class in keeping individual agent logic minimal while achieving complex global behavior.
  - **Data-Oriented Design**: For thousands of simulated agents, the memory layout of your data matters more than the logic. Packing agent state into flat struct arrays (NativeArray) instead of class objects eliminates cache misses and enables SIMD parallelism via the Burst compiler — transforming what would be a 500-ant ceiling into a 50,000-ant simulation.

---
Date: 2026-03-11
Topic: Multi-agent coordination paradigms
Concepts:
  - **Emergence vs. Coordination**: Most large-scale simulations deliberately deny agents global knowledge — the interesting behavior only appears when agents can only sense locally. When agents *can* communicate globally (blackboard systems, flow fields), you get efficient routing but lose emergent surprise. The design choice of what an agent can and cannot perceive is often more important than the rules themselves.
  - **Stigmergy vs. Direct Interaction**: Two fundamental classes of swarm coordination. In stigmergy, agents communicate indirectly by modifying a shared medium (pheromones, physical construction, resource depletion). In direct interaction (Boids, social forces), agents sense each other. Stigmergy tends to produce persistent, spatial patterns; direct interaction tends to produce fluid, dynamic collective motion.

---
Date: 2026-03-11
Topic: Machine learning applied to mass agent simulation
Concepts:
  - **Hierarchical Control**: Splitting agent behavior into fast local rules and slow global policy — a "colony brain" modulating rule-based ants — is a form of hierarchical reinforcement learning. The key insight is that you don't need ML everywhere; putting it at the right level of abstraction (strategic decisions, not motor control) dramatically reduces the problem complexity and keeps the simulation performant.
  - **Autocurricula**: In competitive multi-agent training (like OpenAI Hide and Seek), difficulty self-generates — as one side learns an exploit, the other adapts, producing an open-ended training curriculum without human design. This is the ML equivalent of evolutionary arms races, and it's why self-play tends to produce richer emergent behaviors than training against fixed opponents.

---
Date: 2026-03-11
Topic: Pheromone infrastructure — steering, GPU field, Burst job architecture
Concepts:
  - **Stigmergy**: Indirect coordination through environment modification — ants don't communicate directly, they write chemical signals into a shared field and react to what they find. The pheromone texture is the stigmergic medium; the three-sensor steering model is the reaction. This is why 10,000 individually simple agents can produce colony-level intelligence.
  - **Data-Oriented Design (DOD)**: Organizing simulation state as flat arrays of blittable value types rather than object graphs, so the CPU cache is maximally utilized and SIMD auto-vectorization is possible. The `NativeArray<AntData>` struct layout is a direct application: no references, no inheritance, no virtual dispatch — just contiguous memory the Burst compiler can reason about and vectorize.

---
Date: 2026-03-11
Topic: Pheromone infrastructure implementation — GPU/CPU bridge, separable convolution
Concepts:
  - **Async GPU Readback**: The one-to-two-frame latency of `AsyncGPUReadback` is not a bug — it's a deliberate trade: ants steer on slightly stale pheromone data, but the GPU never stalls waiting for the CPU to finish reading. This latency is physically plausible (real chemical diffusion has propagation delay) and keeps the GPU pipeline fully saturated. Sebastian Lague's ant simulation and GPU Gems 3 both use this pattern.
  - **Separable Convolution**: A 2D Gaussian blur decomposes into two 1D passes (horizontal then vertical) because the Gaussian kernel is separable — `G(x,y) = G(x) * G(y)`. This reduces O(k²) per pixel to O(2k), making larger diffusion radii practical on the GPU. The key constraint is that intermediate results (the horizontal pass output) must be fully written before the vertical pass reads them, which GPU command ordering guarantees within a single frame's dispatches.
