/**
 * Demo code for the echo effect used in FAD (game jam game).
 */

///////////////////////////////////////////////////////////////////////////////
// Core

function traceRay(terrain, width, startX, startY, bearing) {
    // Calculate step direction, where max(abs(xStep), abs(yStep)) == 1
    let xStep, yStep;
    const cosBearing = Math.cos(bearing);
    const sinBearing = -Math.sin(bearing);
    if (Math.abs(cosBearing) > Math.abs(sinBearing)) {  // Y axis is major
        startX += 0.5;
        xStep = sinBearing / Math.abs(cosBearing);
        yStep = Math.sign(cosBearing);
    } else {  // X axis is major
        startY += 0.5;
        xStep = Math.sign(sinBearing);
        yStep = cosBearing / Math.abs(sinBearing);
    }
    // Line rasterisation loop
    const steps = [];
    for (let [x, y] = [startX + xStep, startY + yStep]; ; x += xStep, y += yStep) {
        const [ix, iy] = [Math.floor(x), Math.floor(y)];
        steps.push([ix, iy]);
        if (terrain[width * iy + ix]) {
            break;
        }
    }
    const stepLength = Math.sqrt(xStep * xStep + yStep * yStep);
    return { steps: steps, stepLength: stepLength, distance: stepLength * steps.length };
}

function traceRays(terrain, width, x, y, bearing, nRays) {
    return Array.from({ length: nRays }, (_, i) => {
        const ray = {
            x: Math.floor(x),
            y: Math.floor(y),
            bearing: bearing + i * 2 * Math.PI / nRays,
        };
        Object.assign(ray, traceRay(terrain, width, ray.x, ray.y, ray.bearing));
        return ray;
    });
}

function dbToGain(db) {
    return Math.pow(10, db / 20);
}

function playEcho(ctx, distances, attenuation, waveSpeed, duration, lateralDelay, useWorklet) {
    // Calculate stereo gains and delays
    const rawGain = distances.map(distance => dbToGain(-attenuation * distance));
    const totalGain = rawGain.reduce((total, x) => total + x, 0);
    const echoes = distances.map((distance, i) => {
        const gain = rawGain[i] / Math.max(totalGain, 1);
        const delay = distance / waveSpeed;
        const pan = Math.sin(i * 2 * Math.PI / distances.length);
        return {
            gainLeft: gain * (1 - pan) / 2,
            gainRight: gain * (1 + pan) / 2,
            delayLeft: delay + lateralDelay * Math.max(pan, 0),
            delayRight: delay + lateralDelay * Math.max(-pan, 0),
        };
    });

    // Build the audio graph
    const startTime = ctx.currentTime + 0.1;
    const oscillator = new OscillatorNode(ctx, { type: "sine", frequency: 700 });
    const decay = new GainNode(ctx, { gain: 0 });
    const halfSine = new Float32Array(32).map((_, idx) => Math.sin(idx * Math.PI / 31));
    decay.gain.setValueCurveAtTime(halfSine, startTime, duration);
    const ping = oscillator.connect(decay);

    ping.connect(new GainNode(ctx, { gain: 0.1 })).connect(ctx.destination);

    if (useWorklet) {
        ping.connect(new AudioWorkletNode(ctx, "echo-processor", {
            outputChannelCount: [2],
            processorOptions: { echoes: echoes },
        })).connect(ctx.destination);
    } else {
        for (const echo of echoes) {
            const merge = new ChannelMergerNode(ctx, { numberOfInputs: 2 });
            ping.connect(new DelayNode(ctx, { delayTime: echo.delayLeft, maxDelayTime: echo.delayLeft }))
                .connect(new GainNode(ctx, { gain: echo.gainLeft }))
                .connect(merge, 0, 0);
            ping.connect(new DelayNode(ctx, { delayTime: echo.delayRight, maxDelayTime: echo.delayRight }))
                .connect(new GainNode(ctx, { gain: echo.gainRight }))
                .connect(merge, 0, 1);
            merge.connect(ctx.destination);
        }
    }

    oscillator.start(startTime);
    oscillator.stop(startTime + duration);
    return echoes;
}

///////////////////////////////////////////////////////////////////////////////
// Common

class Game {
    constructor(rootElement, terrain, width, height, boat, createHandler) {
        this.rootElement = rootElement;
        [this.canvas] = rootElement.getElementsByClassName("demo-game-canvas");
        this.scale = 8;
        this.canvas.width = this.scale * width;
        this.canvas.height = this.scale * height;
        this.terrain = terrain;
        this.width = width;
        this.height = height;
        this.initBoat = boat;
        this.boat = Object.assign({}, boat);

        this.dt = 1 / 60;
        this.watchKeys = new Set([" ", "w", "s", "a", "d", "ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"]);
        this.keys = new Set();
        this.ticker = null;
        this.rootElement.addEventListener("keydown", e => this.handleKey(e));
        this.rootElement.addEventListener("keyup", e => this.handleKey(e));
        this.rootElement.addEventListener("focus", e => this.handleFocus(e));
        this.rootElement.addEventListener("focusout", e => this.handleFocus(e));
        for (const span of this.rootElement.getElementsByClassName("demo-reset")) {
            span.addEventListener("click", () => this.reset());
        }

        this.buttons = new Set();
        for (const kind of ["up", "left", "down", "right"]) {
            for (const button of this.rootElement.getElementsByClassName(`demo-game-control-${kind}`)) {
                button.addEventListener("mousedown", () => { this.buttons.add(kind); });
                button.addEventListener("mouseup", () => { this.buttons.delete(kind); });
                button.addEventListener("mouseout", () => { this.buttons.delete(kind); });
                button.addEventListener("touchstart", () => { this.buttons.add(kind); });
                button.addEventListener("touchend", () => { this.buttons.delete(kind); });
            }
        }
        for (const button of this.rootElement.getElementsByClassName("demo-game-control-ping")) {
            button.addEventListener("mousedown", () => this.handler.ping());
        }

        this.handler = createHandler(this);
        this.draw(false);
    }

    static load(rootElement, createHandler) {
        return new Promise((resolve, _) => {
            const image = new Image();
            image.onload = () => {
                const ctx = document.createElement("canvas").getContext("2d", { colorSpace: "srgb" });
                ctx.canvas.width = image.width;
                ctx.canvas.height = image.height;
                ctx.drawImage(image, 0, 0);
                const imageData = ctx.getImageData(0, 0, image.width, image.height);
                const terrain = Array.from(
                    { length: image.width * image.height },
                    (_, idx) => 1 - Boolean(imageData.data[4 * idx])
                );
                const game = new Game(rootElement, terrain, image.width, image.height,
                    { x: 13, y: 18, bearing: Math.PI * 0.75 }, createHandler);
                resolve(game);
            };
            image.src = "demo/map.png";
        });
    }

    handleKey(e) {
        if (this.watchKeys.has(e.key)) {
            if (e.type === "keyup") this.keys.delete(e.key);
            if (e.type === "keydown") {
                this.keys.add(e.key);
                if (e.key === " ") {
                    this.handler.ping();
                }
            }
            e.preventDefault();
        }
    }

    handleFocus(e) {
        if (this.ticker !== null) {
            window.clearInterval(this.ticker);
            this.ticker = null;
        }
        if (e.type === "focus") {
            this.ticker = window.setInterval(() => this.tick(), 1000 * this.dt);
        }
        if (e.type === "focusout") {
            this.draw(false);
        }
    }

    reset() {
        Object.assign(this.boat, this.initBoat);
        this.handler.reset();
    }

    draw(focus) {
        const ctx = this.canvas.getContext("2d");
        ctx.setTransform();

        ctx.fillStyle = focus ? "#fff" : "#888";
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.fillStyle = "#000";
        ctx.scale(this.scale, this.scale);
        for (let y = 0; y < this.height; ++y) {
            for (let x = 0; x < this.width; ++x) {
                if (this.terrain[this.width * y + x]) {
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }

        this.handler.draw(ctx, focus);

        ctx.setTransform();
        ctx.scale(this.scale, this.scale);
        ctx.translate(this.boat.x, this.boat.y);
        ctx.rotate(this.boat.bearing);
        ctx.beginPath();
        ctx.moveTo(-1, -1);
        ctx.bezierCurveTo(-0.25, +2, +0.25, +2, +1, -1);
        ctx.bezierCurveTo(0, -0.5, 0, -0.5, -1, -1);
        ctx.fillStyle = "#00f";
        ctx.fill();
    }

    tick() {
        const thrust = 3 * (
            +(this.keys.has("ArrowUp") || this.keys.has("w") || this.buttons.has("up"))
            - +(this.keys.has("ArrowDown") || this.keys.has("s") || this.buttons.has("down")));
        const rotation = 1.5 * (
            +(this.keys.has("ArrowRight") || this.keys.has("d") || this.buttons.has("right"))
            - +(this.keys.has("ArrowLeft") || this.keys.has("a") || this.buttons.has("left")));
        this.boat.bearing += this.dt * rotation;
        const newX = this.boat.x + thrust * this.dt * -Math.sin(this.boat.bearing);
        const newY = this.boat.y + thrust * this.dt * Math.cos(this.boat.bearing);
        if (!this.terrain[this.width * Math.floor(newY) + Math.floor(newX)]) {
            this.boat.x = newX;
            this.boat.y = newY;
        }
        this.handler.tick();
        this.draw(true);
    }
}

/**
 * Setup standard responsive behaviour for input sliders and reset button.
 *
 * @param {HTMLElement} root container of inputs and reset button
 * @param {Function} action run when clicked
 */
function setupConfigurableAction(root, action) {
    const [reset] = root.getElementsByClassName("demo-reset");
    for (const input of root.getElementsByTagName("input")) {
        const label = root.querySelector(`label[for=${input.id}]`);
        const valueField = input.type === "checkbox" ? "checked" : "value";
        const defaultValue = input[valueField];

        const setLabel = () => {
            const value = input[valueField];
            if (input.type === "checkbox") {
                label.textContent = label.getAttribute(`data-format-${value}`);
            } else {
                label.textContent = label.getAttribute("data-format").replace("{}", value);
            }
        };
        setLabel();
        input.addEventListener("change", () => {
            setLabel();
            action();
        });
        reset.addEventListener("mousedown", () => {
            input[valueField] = defaultValue;
            setLabel();
        });
    }
    root.addEventListener("mousedown", action);
    root.addEventListener("keydown", e => {
        if (e.key === " " || e.key === "Enter") {
            action();
            e.preventDefault();
        }
    });
}

///////////////////////////////////////////////////////////////////////////////
// Demo main

class MainDemoHandler {
    constructor(game, audioContext) {
        this.nRays = 16;
        this.game = game;
        this.audioContext = audioContext;
        this.reset();
    }

    reset() {
        this.rays = [];
        this.rayTimeToLive = 0;
    }

    ping() {
        this.rays = traceRays(this.game.terrain, this.game.width,
            this.game.boat.x, this.game.boat.y, this.game.boat.bearing, this.nRays);
        const duration = 0.04;
        const echoes = playEcho(this.audioContext, this.rays.map(ray => ray.distance),
            5, 20, duration, 0.01, "audioWorklet" in this.audioContext);
        this.rayTimeToLive = duration + echoes.reduce((max, echo) => Math.max(max, echo.delayLeft, echo.delayRight), 0);
    }

    tick() {
        this.rayTimeToLive -= this.game.dt;
    }

    draw(ctx) {
        if (this.rayTimeToLive > 0) {
            for (const ray of this.rays) {
                ctx.fillStyle = "#ccc";
                ctx.fillRect(ray.x, ray.y, 1, 1);
                for (let j = 0; j < ray.steps.length; ++j) {
                    ctx.fillStyle = (j < ray.steps.length - 1 ? "#ccc" : "#0f0");
                    const [x, y] = ray.steps[j];
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Demo physics

class PhysicsHandler {
    constructor(game) {
        this.game = game;
        this.nRays = 4;
        this.animationDt = 1 / 20;
        this.statsScale = 6;
        this.statsYmul = 4;
        this.reset();
        this.statsCanvas = document.getElementById("demo-stats-canvas");
        this.statsCanvas.width = this.statsScale * (this.statsYmul + Math.SQRT2 * this.game.width);
        this.statsCanvas.height = this.statsYmul * this.statsScale * this.nRays;
    }

    reset() {
        this.rays = [];
        this.animationTime = 0;
    }

    ping() {
        this.rays = traceRays(this.game.terrain, this.game.width,
            this.game.boat.x, this.game.boat.y, this.game.boat.bearing, this.nRays);
        this.animationTime = 0;
    }

    tick() {
        this.animationTime += this.game.dt / this.animationDt;
    }

    drawStats(focus) {
        const bgColor = focus ? "#fff" : "#888";
        const ctx = this.statsCanvas.getContext("2d");
        ctx.setTransform();
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, this.statsCanvas.width, this.statsCanvas.height);

        // Draw animated bars
        ctx.setTransform();
        ctx.scale(this.statsScale, this.statsScale);
        ctx.fillStyle = "#080";
        let nAnimated = 0;
        for (let i = 0; i < this.rays.length; ++i) {
            const ray = this.rays[i];
            for (let j = 0; j < ray.steps.length; ++j) {
                if (nAnimated++ < Math.floor(this.animationTime)) {
                    const x = this.statsYmul + j * ray.stepLength;
                    const y = this.statsYmul * i;
                    ctx.fillRect(x, y, ray.stepLength + 1 / this.statsScale, this.statsYmul);
                }
            }
        }

        // Draw arrows
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, this.statsYmul, this.statsYmul * this.rays.length);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 0.25;
        for (let i = 0; i < this.rays.length; ++i) {
            ctx.setTransform();
            ctx.scale(this.statsScale, this.statsScale);
            ctx.translate(0.5 * this.statsYmul, this.statsYmul * (i + 0.5));
            ctx.rotate(this.rays[i].bearing);
            ctx.scale(0.75, 0.75);
            ctx.beginPath();
            ctx.moveTo(0, -1);
            ctx.lineTo(0, 1);
            ctx.lineTo(1, 0);
            ctx.lineTo(0, 1);
            ctx.lineTo(-1, 0);
            ctx.stroke();
        }
    }

    draw(ctx, focus) {
        let nAnimated = 0;
        for (const ray of this.rays) {
            ctx.fillStyle = "#0c0";
            ctx.fillRect(ray.x, ray.y, 1, 1);
            for (let j = 0; j < ray.steps.length; ++j) {
                if (nAnimated++ < Math.floor(this.animationTime)) {
                    ctx.fillStyle = (j < ray.steps.length - 1 ? "#ccc" : "#0f0");
                    const [x, y] = ray.steps[j];
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }
        this.drawStats(focus);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Demo ping

function playPing(ctx, frequency, duration) {
    const startTime = ctx.currentTime + 0.1;
    const oscillator = new OscillatorNode(ctx, { type: "sine", frequency: frequency });
    const decay = new GainNode(ctx, { gain: 0 });
    const halfSine = new Float32Array(32).map((_, idx) => Math.sin(idx * Math.PI / 31));
    decay.gain.setValueCurveAtTime(halfSine, startTime, duration);
    oscillator.connect(decay)
        .connect(new GainNode(ctx, { gain: 0.25 }))
        .connect(ctx.destination);

    oscillator.start(startTime);
    oscillator.stop(startTime + duration);
}

function setupDemoPing(audioContext) {
    setupConfigurableAction(document.getElementById("demo-ping"), () => {
        playPing(audioContext,
            Number(document.getElementById("demo-ping-frequency").value),
            Number(document.getElementById("demo-ping-duration").value));
    });
}

///////////////////////////////////////////////////////////////////////////////
// Demo echo

function plotDemoEcho(echoes, duration, canvas) {
    const pingGain = 0.1;
    const maxTime = duration + echoes.reduce((max, echo) => Math.max(max, echo.delayLeft, echo.delayRight), 0);
    const maxGain = echoes.reduce((max, echo) => Math.max(max, echo.gainLeft, echo.gainRight), pingGain);

    const ctx = canvas.getContext("2d");
    ctx.setTransform();
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const padPx = 10;
    ctx.translate(padPx, padPx + (canvas.height - 2 * padPx) / 2);
    ctx.scale((canvas.width - 2 * padPx) / maxTime, (canvas.height - 2 * padPx) / maxGain / 2);

    // Axes
    ctx.strokeStyle = "#000";
    ctx.lineWidth = maxGain / canvas.height;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(maxTime, 0);
    ctx.stroke();
    ctx.lineWidth = maxTime / canvas.width;
    ctx.beginPath();
    ctx.moveTo(0, -maxGain);
    ctx.lineTo(0, maxGain);
    ctx.stroke();

    // Content
    const halfSine = Array.from({ length: 32 }, (_, idx) => Math.sin(idx * Math.PI / 31));
    function drawWave(time, gain, left) {
        ctx.fillStyle = left ? "#a00" : "#080";
        const maxY = left ? -gain : +gain;
        ctx.beginPath();
        ctx.moveTo(time, 0);
        halfSine.forEach((y, i) => {
            ctx.lineTo(time + i * duration / 31, maxY * y);
        });
        ctx.fill();
    }
    drawWave(0, pingGain, true);
    drawWave(0, pingGain, false);
    for (const echo of echoes) {
        drawWave(echo.delayLeft, echo.gainLeft, true);
        drawWave(echo.delayRight, echo.gainRight, false);
    }
}

function setupDemoEcho(audioContext) {
    const rootElement = document.getElementById("demo-echo");
    const canvas = document.getElementById("demo-wave-canvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setupConfigurableAction(rootElement, () => {
        const duration = 0.04;
        const echoes = playEcho(audioContext,
            [Number(document.getElementById("demo-echo-range1").value),
            Number(document.getElementById("demo-echo-range2").value),
            Number(document.getElementById("demo-echo-range3").value),
            Number(document.getElementById("demo-echo-range4").value)],
            Number(document.getElementById("demo-echo-attenuation").value),
            Number(document.getElementById("demo-echo-wavespeed").value),
            duration,
            0.01,
            document.getElementById("demo-echo-toggle-worklet").checked
        );
        plotDemoEcho(echoes, duration, canvas);
    });
}

///////////////////////////////////////////////////////////////////////////////
// Initialization

window.onload = () => {
    const audioContext = new AudioContext();
    if (audioContext.audioWorklet) {
        audioContext.audioWorklet.addModule("demo/demoechoworklet.js");
    } else {
        console.warn("AudioContext.audioWorklet is not available. Perhaps blocked over HTTP - try HTTPS.");
    }
    Game.load(document.getElementById("demo-main"), game => new MainDemoHandler(game, audioContext));
    Game.load(document.getElementById("demo-physics"), game => new PhysicsHandler(game));
    setupDemoPing(audioContext);
    setupDemoEcho(audioContext);
};
