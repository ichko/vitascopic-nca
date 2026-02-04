// Core NCA data structures and stepping logic.

const GRID_SIZE = 32;

/**
 * Create a zero-initialized state for a grid of size GRID_SIZE x GRID_SIZE
 * with the given number of channels.
 */
export function createEmptyState(channels) {
  return new Float32Array(GRID_SIZE * GRID_SIZE * channels);
}

/**
 * Utility to clone a state Float32Array.
 */
export function cloneState(state) {
  return new Float32Array(state);
}

/**
 * Apply a simple brush at integer coordinates (x, y) in grid space.
 * The brush adds a small activation vector to the cell and clamps to [0, 1].
 */
export function applyBrush(state, channels, x, y, options = {}) {
  const brushVector =
    options.brushVector || [0.0, 0.8, 0.2, 0.0]; // default RG-ish activation
  const radius = options.radius || 0;

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  const applyAt = (ix, iy, strength = 1.0) => {
    if (ix < 0 || ix >= GRID_SIZE || iy < 0 || iy >= GRID_SIZE) return;
    const base = (iy * GRID_SIZE + ix) * channels;
    for (let c = 0; c < channels && c < brushVector.length; c++) {
      const idx = base + c;
      const delta = brushVector[c] * strength;
      state[idx] = clamp01(state[idx] + delta);
    }
  };

  if (radius <= 0) {
    applyAt(x, y, 1.0);
    return;
  }

  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= radius + 1e-6) {
        const falloff = 1 - dist / (radius || 1);
        applyAt(x + dx, y + dy, falloff);
      }
    }
  }
}

/**
 * Render NCA state to a canvas 2D context.
 * Maps channels 0..2 to RGB; if fewer channels, falls back gracefully.
 */
export function renderStateToCanvas(state, channels, ctx) {
  const size = GRID_SIZE;
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;

  const clamp255 = (v) => {
    if (Number.isNaN(v)) return 0;
    if (v < 0) return 0;
    if (v > 1) return 255;
    return Math.round(v * 255);
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const base = (y * size + x) * channels;
      const outIdx = (y * size + x) * 4;

      const r = channels > 0 ? state[base + 0] : 0;
      const g = channels > 1 ? state[base + 1] : r;
      const b = channels > 2 ? state[base + 2] : r;

      data[outIdx + 0] = clamp255(r);
      data[outIdx + 1] = clamp255(g);
      data[outIdx + 2] = clamp255(b);
      data[outIdx + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Convert mouse event coordinates to integer grid coordinates in [0, GRID_SIZE-1].
 */
export function eventToCellCoords(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = GRID_SIZE / rect.width;
  const scaleY = GRID_SIZE / rect.height;

  const x = Math.floor((event.clientX - rect.left) * scaleX);
  const y = Math.floor((event.clientY - rect.top) * scaleY);

  const clampedX = x < 0 ? 0 : x >= GRID_SIZE ? GRID_SIZE - 1 : x;
  const clampedY = y < 0 ? 0 : y >= GRID_SIZE ? GRID_SIZE - 1 : y;

  return { x: clampedX, y: clampedY };
}

export { GRID_SIZE };


