const S = {
  hwidth: 100,
  hheight: 100,
  initialSpeed: 25,
  maxKESteps: 60 * 60, // about 1 minute at 60fps
  physicsDt: 1 / 200,
  maxDt: 1 / 15,
  collisionBucketSize: 4,
};
const CONTROLS = [
  {
    name: "num-balls",
    default: 2000,
    min: 100,
    max: 30000,
    step: 500,
    title: "Num Balls",
  },
  {
    name: "radius",
    default: 1.4,
    min: 0.2,
    max: 4,
    step: 0.2,
    title: "Radius",
  },
  {
    name: "restitution",
    default: 1,
    min: 0.95,
    max: 1.01,
    step: 0.001,
    title: "Bounce, Ball",
  },
  {
    name: "temp-top",
    default: 1,
    min: 0.1,
    max: 2,
    step: 0.1,
    title: "Bounce, Top",
  },
  {
    name: "temp-bottom",
    default: 1,
    min: 0.1,
    max: 2,
    step: 0.1,
    title: "Bounce, Bottom",
  },
];

type WallFactors = { top: number; bottom: number; left: number; right: number };

function shuffle<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

class Balls {
  px: number[] = [];
  py: number[] = [];
  vx: number[] = [];
  vy: number[] = [];
  r: number[] = [];
  buckets: number[][];
  bucketNx: number;
  bucketNy: number;
  bucketOrder: number[];
  // Settings
  wallFactors: WallFactors = { top: 1, bottom: 1, left: 1, right: 1 };
  restitution: number = 1;

  constructor() {
    this.bucketNx = Math.ceil((S.hwidth * 2) / S.collisionBucketSize);
    this.bucketNy = Math.ceil((S.hheight * 2) / S.collisionBucketSize);
    this.buckets = new Array(this.bucketNx * this.bucketNy);
    this.bucketOrder = new Array(this.buckets.length);
    for (let i = 0; i < this.buckets.length; i++) {
      this.buckets[i] = [];
      this.bucketOrder[i] = i;
    }
  }

  get length(): number {
    return this.px.length;
  }

  spawn(px: number, py: number, vx: number, vy: number, r: number) {
    this.px.push(px);
    this.py.push(py);
    this.vx.push(vx);
    this.vy.push(vy);
    this.r.push(r);
  }

  remove(count: number) {
    this.px.splice(0, count);
    this.py.splice(0, count);
    this.vx.splice(0, count);
    this.vy.splice(0, count);
    this.r.splice(0, count);
  }

  foreach(callback: (i: number) => void) {
    for (let i = 0; i < this.px.length; i++) {
      callback(i);
    }
  }

  private updateCollisions() {
    const bs = S.collisionBucketSize;
    const nx = this.bucketNx;
    const ny = this.bucketNy;
    const buckets = this.buckets;

    // Clear buckets
    for (let i = 0; i < buckets.length; i++) {
      buckets[i].length = 0;
    }

    // Assign balls to buckets
    // (each ball goes into every bucket that the bounding box overlaps)
    this.foreach((i) => {
      const x = this.px[i] + S.hwidth;
      const y = this.py[i] + S.hheight;
      const r = this.r[i];
      const bx0 = Math.min(Math.max(Math.floor((x - r) / bs), 0), nx - 1);
      const bx1 = Math.min(Math.max(Math.floor((x + r) / bs), 0), nx - 1);
      const by0 = Math.min(Math.max(Math.floor((y - r) / bs), 0), ny - 1);
      const by1 = Math.min(Math.max(Math.floor((y + r) / bs), 0), ny - 1);
      for (let by = by0; by <= by1; by++) {
        for (let bx = bx0; bx <= bx1; bx++) {
          buckets[by * nx + bx].push(i);
        }
      }
    });

    // Check collisions in each bucket
    // (randomise order otherwise collisions in earlier buckets always get resolved first)
    shuffle(this.bucketOrder);
    for (const bidx of this.bucketOrder) {
      const bucket = buckets[bidx];
      for (const i of bucket) {
        for (const j of bucket) {
          if (i >= j) continue;
          const dx = this.px[j] - this.px[i];
          const dy = this.py[j] - this.py[i];
          const distSq = dx * dx + dy * dy;
          const rsum = this.r[i] + this.r[j];
          // if overlapping
          if (distSq < rsum * rsum) {
            const vx_i = this.vx[i];
            const vy_i = this.vy[i];
            const vx_j = this.vx[j];
            const vy_j = this.vy[j];
            // if moving towards each other
            if (vx_i * dx + vy_i * dy - vx_j * dx - vy_j * dy > 0) {
              // dot product with unscaled-normal
              const p = vx_i * dx + vy_i * dy - vx_j * dx - vy_j * dy;
              const r = (this.restitution * p) / distSq;
              this.vx[i] -= r * dx;
              this.vy[i] -= r * dy;
              this.vx[j] += r * dx;
              this.vy[j] += r * dy;
            }
          }
        }
      }
    }
  }

  update(dt: number) {
    this.updateCollisions();
    // Move and handle wall collisions
    this.foreach((i) => {
      let vx = this.vx[i];
      let vy = this.vy[i];
      let x = this.px[i] + this.vx[i] * dt;
      let y = this.py[i] + this.vy[i] * dt;
      const r = this.r[i];
      // Walls
      if (x - r < -S.hwidth) {
        x = -S.hwidth + r;
        vx *= -1 * this.wallFactors.left;
      }
      if (x + r > S.hwidth) {
        x = S.hwidth - r;
        vx *= -1 * this.wallFactors.right;
      }
      if (y - r < -S.hheight) {
        y = -S.hheight + r;
        vy *= -1 * this.wallFactors.top;
      }
      if (y + r > S.hheight) {
        y = S.hheight - r;
        vy *= -1 * this.wallFactors.bottom;
      }
      this.px[i] = x;
      this.py[i] = y;
      this.vx[i] = vx;
      this.vy[i] = vy;
    });
  }

  kineticEnergy(): number {
    let ke = 0;
    this.foreach((i) => {
      ke += 0.5 * (this.vx[i] * this.vx[i] + this.vy[i] * this.vy[i]);
    });
    return ke;
  }
}

function speedToColor(speed: number): string {
  const t = Math.min(speed / (2 * S.initialSpeed), 1);
  const r = Math.floor(255 * t);
  const g = 0;
  const b = (0.5 + 0.5 * t) * Math.floor(255 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

class Simulation {
  balls: Balls;
  lastTimeMs: number | null = null;
  keHistory: number[] = [];
  physicsDeltaT: number = 0;

  constructor(readonly controlValues: Map<string, number>) {
    this.balls = new Balls();
    this.updateControls();
  }

  private spawn() {
    const radius = this.controlValues.get("radius")!;
    const numBalls = this.controlValues.get("num-balls")!;
    if (this.balls.length >= numBalls) {
      this.balls.remove(this.balls.length - numBalls);
    }
    this.balls.foreach((i) => {
      this.balls.r[i] = radius;
    });
    while (this.balls.length < numBalls) {
      const x = Math.random() * S.hwidth * 2 - S.hwidth;
      const y = Math.random() * S.hheight * 2 - S.hheight;
      const va = Math.random() * Math.PI * 2;
      const vx = Math.cos(va) * S.initialSpeed;
      const vy = Math.sin(va) * S.initialSpeed;
      this.balls.spawn(x, y, vx, vy, radius);
    }
  }

  updateControls() {
    this.balls.wallFactors.top = this.controlValues.get("temp-top")!;
    this.balls.wallFactors.bottom = this.controlValues.get("temp-bottom")!;
    this.balls.restitution = this.controlValues.get("restitution")!;
    this.spawn();
  }

  update(timeMs: number) {
    if (this.lastTimeMs !== null) {
      const dt = Math.min((timeMs - this.lastTimeMs) / 1000, S.maxDt);
      this.physicsDeltaT += dt;
      while (this.physicsDeltaT >= S.physicsDt) {
        this.balls.update(S.physicsDt);
        this.physicsDeltaT -= S.physicsDt;
      }
    }
    this.lastTimeMs = timeMs;
    // Record KE
    this.keHistory.push(this.balls.kineticEnergy());
    while (this.keHistory.length > S.maxKESteps) {
      this.keHistory.shift();
    }
  }

  get ke(): number {
    return this.keHistory[this.keHistory.length - 1] || 0;
  }
}

function formatNumber(val: number, step: number): string {
  if (step < 1) {
    return val.toFixed(Math.ceil(Math.log10(1 / step)));
  }
  return val.toFixed(0);
}

function createFps(): (timeMs: number) => void {
  const fpsLabel = document.getElementById("fps-value") as HTMLSpanElement;
  let smoothFps = 0;
  let smoothFpsCount = 0;
  const maxSmoothCount = 10;
  let lastTimeMs: number | null = null;
  return (timeMs: number) => {
    if (lastTimeMs !== null) {
      const dt = (timeMs - lastTimeMs) / 1000;
      const fps = 1 / dt;
      smoothFps = (smoothFps * smoothFpsCount + fps) / (smoothFpsCount + 1);
      smoothFpsCount = Math.min(smoothFpsCount + 1, maxSmoothCount);
      fpsLabel.innerText = smoothFps.toFixed(0);
    }
    lastTimeMs = timeMs;
  };
}

window.onload = () => {
  const controlValues = new Map<string, number>();
  CONTROLS.forEach((control) => {
    controlValues.set(control.name, control.default);
  });
  let sim = new Simulation(controlValues);

  document.getElementById("btn-restart")!.onclick = () => {
    sim = new Simulation(controlValues);
  };
  CONTROLS.forEach((control) => {
    // Create a .control-group for this control
    const controlGroup = document.createElement("div");
    controlGroup.className = "control-group";
    const defaultStr = formatNumber(control.default, control.step);
    controlGroup.innerHTML = `
        <div class="control-header">
            <label>${control.title}</label>
            <button class="btn-${control.name} btn-reset">DEFAULT</button>
        </div>
        <div class="button-row">
            <button class="btn-${control.name} btn-decrease"></button>
            <span id="label-${control.name}">${defaultStr}</span>
            <button class="btn-${control.name} btn-increase"></button>
        </div>
    `;
    document.getElementById("panel-side")!.appendChild(controlGroup);

    // Wire up handlers
    const label = document.getElementById(`label-${control.name}`)!;
    const updateDisabled = () => {
      document.querySelectorAll(`.btn-${control.name}`).forEach((e) => {
        const b = e as HTMLButtonElement;
        if (b.classList.contains("btn-decrease")) {
          b.disabled = controlValues.get(control.name)! <= control.min;
        }
        if (b.classList.contains("btn-increase")) {
          b.disabled = controlValues.get(control.name)! >= control.max;
        }
      });
    };
    updateDisabled();
    document.querySelectorAll(`.btn-${control.name}`).forEach((b) => {
      if (b.classList.contains("btn-reset")) {
        b.addEventListener("click", () => {
          label.textContent = formatNumber(control.default, control.step);
          controlValues.set(control.name, control.default);
          updateDisabled();
          sim.updateControls();
        });
      } else {
        const step =
          control.step * (b.classList.contains("btn-decrease") ? -1 : 1);
        b.textContent = step > 0 ? `+${step}` : `${step}`;
        b.addEventListener("click", () => {
          let val = controlValues.get(control.name)!;
          val += step;
          val = Math.round(val / control.step) * control.step;
          val = Math.min(Math.max(val, control.min), control.max);
          label.textContent = formatNumber(val, control.step);
          controlValues.set(control.name, val);
          updateDisabled();
          sim.updateControls();
        });
      }
    });
  });

  const mainCanvas = document.getElementById(
    "canvas-main"
  ) as HTMLCanvasElement;
  const ctx = mainCanvas.getContext("2d")!;
  const keCanvas = document.getElementById("canvas-ke") as HTMLCanvasElement;
  const keCtx = keCanvas.getContext("2d")!;

  const updateFps = createFps();
  const animate = (timeMs: number) => {
    updateFps(timeMs);

    // Update physics
    sim.update(timeMs);

    // Draw balls
    const balls = sim.balls;
    ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 4;
    ctx.strokeRect(0, 0, mainCanvas.width, mainCanvas.height);
    balls.foreach((i) => {
      const screenX =
        ((balls.px[i] + S.hwidth) / (S.hwidth * 2)) * mainCanvas.width;
      const screenY =
        ((balls.py[i] + S.hheight) / (S.hheight * 2)) * mainCanvas.height;
      const screenR = (balls.r[i] / (S.hwidth * 2)) * mainCanvas.width;
      const speed = Math.sqrt(
        balls.vx[i] * balls.vx[i] + balls.vy[i] * balls.vy[i]
      );
      ctx.fillStyle = speedToColor(speed);
      ctx.beginPath();
      ctx.arc(screenX, screenY, screenR, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw KE
    const keLabel = document.getElementById("label-ke") as HTMLSpanElement;
    keLabel.innerText = `KE: ${(sim.ke / 1000).toFixed(0)} k`;

    keCtx.fillStyle = "#dddddd";
    keCtx.fillRect(0, 0, keCanvas.width, keCanvas.height);
    keCtx.beginPath();
    const keMax = sim.keHistory.reduce((a, b) => (a > b ? a : b), 0);
    const keMin = sim.keHistory.reduce((a, b) => (a < b ? a : b), keMax);
    const keRange = Math.max(keMax - keMin, 1e-4);
    const kePad = 0.1;
    sim.keHistory.forEach((ke, index) => {
      const x = index * (keCanvas.width / sim.keHistory.length);
      const y =
        keCanvas.height *
        (0.5 - (ke - (keMin + keMax) / 2) / (keRange * (1 + kePad)));
      if (index === 0) {
        keCtx.moveTo(x, y);
      } else {
        keCtx.lineTo(x, y);
      }
    });
    keCtx.strokeStyle = "#0000ff";
    keCtx.lineWidth = 2;
    keCtx.stroke();
    keCtx.strokeStyle = "#000000";
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
};
