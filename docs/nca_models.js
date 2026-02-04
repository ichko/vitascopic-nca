// Model registry and JS implementations, plus JSON-backed NCA models.

import { createEmptyState, cloneState, GRID_SIZE } from "./nca_core.js";

/**
 * Shape of a JSON-exported NeuralCA model (see Python export utility).
 * We keep this as a JSDoc typedef-style comment for documentation only.
 *
 * model = {
 *   model_type: "NeuralCA",
 *   model_name: string,
 *   channels: number,
 *   hidden_channels: number,
 *   alive_threshold: number,
 *   fire_rate: number,
 *   padding_type: "circular",
 *   rule: {
 *     conv1: { weight: number[][], bias: number[] },
 *     conv2: { weight: number[][] }
 *   }
 * }
 */

/**
 * Simple built-in demo model that does a diffusive blur on the first channel,
 * and pushes a bit of activation into the remaining channels.
 */
function makeBuiltInDemoModel() {
  const id = "demo-js";
  const channels = 4;

  const initState = () => {
    const state = createEmptyState(channels);
    const center = Math.floor(GRID_SIZE / 2);
    const radius = 3;
    for (let y = -radius; y <= radius; y++) {
      for (let x = -radius; x <= radius; x++) {
        const yy = center + y;
        const xx = center + x;
        if (yy < 0 || yy >= GRID_SIZE || xx < 0 || xx >= GRID_SIZE) continue;
        const base = (yy * GRID_SIZE + xx) * channels;
        const dist = Math.sqrt(x * x + y * y);
        const amp = Math.max(0, 1 - dist / radius);
        state[base + 0] = 0.2 + 0.6 * amp;
        state[base + 1] = 0.1 * amp;
      }
    }
    return state;
  };

  const step = (state) => {
    const next = cloneState(state);

    const kernel = [
      [0.05, 0.1, 0.05],
      [0.1, 0.4, 0.1],
      [0.05, 0.1, 0.05],
    ];

    const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        let acc = 0.0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const yy = (y + ky + GRID_SIZE) % GRID_SIZE;
            const xx = (x + kx + GRID_SIZE) % GRID_SIZE;
            const kVal = kernel[ky + 1][kx + 1];
            const base = (yy * GRID_SIZE + xx) * channels;
            acc += state[base + 0] * kVal;
          }
        }

        const base = (y * GRID_SIZE + x) * channels;
        const center = state[base + 0];
        const delta = acc - center;

        next[base + 0] = clamp01(center + 0.25 * delta);
        next[base + 1] = clamp01(next[base + 1] + 0.03 * delta);
        next[base + 2] = clamp01(next[base + 2] + 0.015 * delta);
      }
    }

    return next;
  };

  return {
    id,
    name: "Built-in demo (JS)",
    channels,
    initState,
    step,
  };
}

/**
 * Build a JS step function from a JSON NeuralCA definition.
 */
function buildNeuralCAStepFromJson(modelJson) {
  const C = modelJson.channels;
  const hiddenC = modelJson.hidden_channels;
  const aliveThreshold = modelJson.alive_threshold ?? 0.0;

  const conv1W = modelJson.rule.conv1.weight; // [hiddenC][3*C]
  const conv1B = modelJson.rule.conv1.bias; // [hiddenC]
  const conv2W = modelJson.rule.conv2.weight; // [C][hiddenC]

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  const aliveMaskFromState = (state) => {
    const mask = new Uint8Array(GRID_SIZE * GRID_SIZE);
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        let maxVal = -Infinity;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const yy = (y + ky + GRID_SIZE) % GRID_SIZE;
            const xx = (x + kx + GRID_SIZE) % GRID_SIZE;
            const base = (yy * GRID_SIZE + xx) * C;
            const v = state[base + 0];
            if (v > maxVal) maxVal = v;
          }
        }
        mask[y * GRID_SIZE + x] = maxVal > aliveThreshold ? 1 : 0;
      }
    }
    return mask;
  };

  const percept = new Float32Array(GRID_SIZE * GRID_SIZE * 3 * C);

  const step = (state) => {
    const size = GRID_SIZE;
    const preAlive = aliveMaskFromState(state);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        for (let c = 0; c < C; c++) {
          let idSum = 0;
          let gx = 0;
          let gy = 0;

          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const yy = (y + ky + size) % size;
              const xx = (x + kx + size) % size;
              const base = (yy * size + xx) * C + c;

              const v = state[base];
              const sx =
                (ky === -1 && kx === -1
                  ? -1
                  : ky === -1 && kx === 1
                  ? 1
                  : ky === 0 && kx === -1
                  ? -2
                  : ky === 0 && kx === 1
                  ? 2
                  : ky === 1 && kx === -1
                  ? -1
                  : ky === 1 && kx === 1
                  ? 1
                  : 0) / 8;
              const sy =
                (ky === -1 && kx === -1
                  ? 1
                  : ky === -1 && kx === 0
                  ? 2
                  : ky === -1 && kx === 1
                  ? 1
                  : ky === 1 && kx === -1
                  ? -1
                  : ky === 1 && kx === 0
                  ? -2
                  : ky === 1 && kx === 1
                  ? -1
                  : 0) / 8;

              idSum += v * (ky === 0 && kx === 0 ? 1 : 0);
              gx += v * sx;
              gy += v * sy;
            }
          }

          const cellIndex = (y * size + x) * (3 * C) + c * 3;
          percept[cellIndex + 0] = idSum;
          percept[cellIndex + 1] = gx;
          percept[cellIndex + 2] = gy;
        }
      }
    }

    const hidden = new Float32Array(GRID_SIZE * GRID_SIZE * hiddenC);
    const delta = new Float32Array(GRID_SIZE * GRID_SIZE * C);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pBase = (y * size + x) * (3 * C);
        const hBase = (y * size + x) * hiddenC;

        for (let h = 0; h < hiddenC; h++) {
          let acc = conv1B[h] || 0;
          const wRow = conv1W[h];
          for (let i = 0; i < 3 * C; i++) {
            acc += wRow[i] * percept[pBase + i];
          }
          if (acc < 0) acc = 0;
          hidden[hBase + h] = acc;
        }

        const dBase = (y * size + x) * C;
        for (let c = 0; c < C; c++) {
          const wRow = conv2W[c];
          let acc = 0;
          for (let h = 0; h < hiddenC; h++) {
            acc += wRow[h] * hidden[hBase + h];
          }
          delta[dBase + c] = acc;
        }
      }
    }

    const next = new Float32Array(state.length);
    for (let i = 0; i < state.length; i++) {
      next[i] = state[i] + delta[i];
    }

    const postAlive = aliveMaskFromState(next);
    if (aliveThreshold > 0) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = y * size + x;
          const keep = preAlive[idx] && postAlive[idx];
          if (!keep) {
            const base = (y * size + x) * C;
            for (let c = 0; c < C; c++) {
              next[base + c] = 0;
            }
          } else {
            const base = (y * size + x) * C;
            for (let c = 0; c < C; c++) {
              next[base + c] = clamp01(next[base + c]);
            }
          }
        }
      }
    } else {
      for (let i = 0; i < next.length; i++) {
        next[i] = clamp01(next[i]);
      }
    }

    return next;
  };

  const initState = () => {
    const state = createEmptyState(C);
    const center = Math.floor(GRID_SIZE / 2);
    const base = (center * GRID_SIZE + center) * C;
    for (let c = 0; c < C; c++) {
      state[base + c] = c === 0 ? 1.0 : 0.0;
    }
    return state;
  };

  return { step, initState, channels: C };
}

export async function loadJsonModels() {
  const descriptors = [
    {
      id: "sample-json-nca",
      name: "Sample JSON NCA",
      path: "./models/sample_nca.json",
    },
  ];

  const loaded = [];

  for (const desc of descriptors) {
    try {
      const res = await fetch(desc.path);
      if (!res.ok) continue;
      const json = await res.json();
      const built = buildNeuralCAStepFromJson(json);
      loaded.push({
        id: desc.id,
        name: desc.name,
        channels: built.channels,
        initState: built.initState,
        step: built.step,
      });
    } catch (err) {
      console.warn("Failed to load JSON NCA model", desc, err);
    }
  }

  return loaded;
}

export function getBuiltInModels() {
  return [makeBuiltInDemoModel()];
}


