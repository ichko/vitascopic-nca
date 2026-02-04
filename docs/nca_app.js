import {
  createEmptyState,
  renderStateToCanvas,
  applyBrush,
  eventToCellCoords,
  GRID_SIZE,
} from "./nca_core.js";
import { getBuiltInModels, loadJsonModels } from "./nca_models.js";

let models = [];
let currentModel = null;
let state = null;
let stepCount = 0;

let isRunning = true;
let isBrushMode = true;
let isMouseDown = false;

let lastFrameTime = 0;
const TARGET_FPS = 24;

function $(id) {
  return document.getElementById(id);
}

async function initModels() {
  const builtIns = getBuiltInModels();
  const jsonModels = await loadJsonModels();
  models = [...builtIns, ...jsonModels];

  const select = $("model-select");
  select.innerHTML = "";

  models.forEach((m, idx) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    if (idx === 0) opt.selected = true;
    select.appendChild(opt);
  });

  currentModel = models[0];
}

function resetState() {
  if (!currentModel) return;
  state =
    typeof currentModel.initState === "function"
      ? currentModel.initState()
      : createEmptyState(currentModel.channels);
  stepCount = 0;
  updateMetrics();
}

function updateMetrics() {
  const resEl = $("metric-res");
  const chEl = $("metric-channels");
  const stepsEl = $("metric-steps");
  if (resEl) resEl.textContent = `${GRID_SIZE}×${GRID_SIZE}`;
  if (chEl)
    chEl.textContent = currentModel ? String(currentModel.channels) : "–";
  if (stepsEl) stepsEl.textContent = String(stepCount);
}

function togglePlayPause() {
  isRunning = !isRunning;
  const btn = $("play-pause-btn");
  if (btn) {
    btn.textContent = isRunning ? "Pause" : "Play";
    if (isRunning) {
      btn.classList.add("toggled");
    } else {
      btn.classList.remove("toggled");
    }
  }
}

function toggleBrush() {
  isBrushMode = !isBrushMode;
  const btn = $("brush-toggle-btn");
  if (btn) {
    btn.textContent = isBrushMode ? "Brush: On" : "Brush: Off";
    if (isBrushMode) {
      btn.classList.add("toggled");
    } else {
      btn.classList.remove("toggled");
    }
  }
}

function attachUIHandlers(canvas, ctx) {
  const modelSelect = $("model-select");
  const resetBtn = $("reset-btn");
  const playPauseBtn = $("play-pause-btn");
  const brushBtn = $("brush-toggle-btn");

  modelSelect.addEventListener("change", () => {
    const id = modelSelect.value;
    const found = models.find((m) => m.id === id);
    if (found) {
      currentModel = found;
      resetState();
      renderStateToCanvas(state, currentModel.channels, ctx);
    }
  });

  resetBtn.addEventListener("click", () => {
    resetState();
    renderStateToCanvas(state, currentModel.channels, ctx);
  });

  playPauseBtn.addEventListener("click", () => {
    togglePlayPause();
  });

  brushBtn.addEventListener("click", () => {
    toggleBrush();
  });

  const handlePaint = (event) => {
    if (!isBrushMode || !currentModel || !state) return;
    const { x, y } = eventToCellCoords(canvas, event);
    applyBrush(state, currentModel.channels, x, y, {
      brushVector: [0.0, 0.9, 0.3, 0.0],
      radius: 0,
    });
  };

  canvas.addEventListener("mousedown", (e) => {
    isMouseDown = true;
    handlePaint(e);
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!isMouseDown) return;
    handlePaint(e);
  });

  window.addEventListener("mouseup", () => {
    isMouseDown = false;
  });
  canvas.addEventListener("mouseleave", () => {
    isMouseDown = false;
  });
}

function stepSimulation() {
  if (!currentModel || !state) return;
  if (typeof currentModel.step !== "function") return;

  state = currentModel.step(state);
  stepCount += 1;
  updateMetrics();
}

function animationLoop(ctx) {
  const loop = (timestamp) => {
    const dt = timestamp - lastFrameTime;
    const minFrameTime = 1000 / TARGET_FPS;
    if (dt >= minFrameTime) {
      lastFrameTime = timestamp;
      if (isRunning) {
        stepSimulation();
      }
      if (currentModel && state) {
        renderStateToCanvas(state, currentModel.channels, ctx);
      }
    }

    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

async function main() {
  const canvas = $("nca-canvas");
  const ctx = canvas.getContext("2d");

  await initModels();
  resetState();
  attachUIHandlers(canvas, ctx);
  renderStateToCanvas(state, currentModel.channels, ctx);
  animationLoop(ctx);
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch((err) => {
    console.error("Failed to initialize NCA app", err);
  });
});


