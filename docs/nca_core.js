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
 * Apply a brush at integer coordinates (x, y) in grid space.
 * This writes a 1.0 value into the FIRST channel only (binary on/off).
 */
export function applyBrush(state, channels, x, y) {
  if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) return;
  const base = (y * GRID_SIZE + x) * channels;
  state[base + 0] = 1.0;
}

/**
 * Render NCA state to a canvas 2D context.
 * Only the FIRST channel is visualized as a binary image (black/white).
 */
export function renderStateToCanvas(state, channels, ctx) {
  const size = GRID_SIZE;
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;


  let max = Math.max(...state);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const base = (y * size + x) * channels;
      const outIdx = (y * size + x) * 4;

      const v = channels > 0 ? state[base + 0] : 0;
      // const on = v > 0.5 ? 255 : 0;
      const normalized = v / max;
      data[outIdx + 0] = normalized * 255;
      data[outIdx + 1] = normalized * 255;
      data[outIdx + 2] = normalized * 255;
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


