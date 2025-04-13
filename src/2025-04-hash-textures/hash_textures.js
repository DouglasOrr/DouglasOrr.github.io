// Schema

const CONFIG_SCHEMA = {
  // Hash
  c_x: { type: "int", alias: "x", label: `$c_{x}$`, min: 0, max: "period" },
  c_y: { type: "int", alias: "y", label: `$c_{y}$`, min: 0, max: "period" },
  c_xy: {
    type: "int",
    alias: "xy",
    label: `$c_{xy}$`,
    min: 0,
    max: "period",
  },
  c_xx: {
    type: "int",
    alias: "xx",
    label: `$c_{x^2}$`,
    min: 0,
    max: "period",
  },
  c_yy: {
    type: "int",
    alias: "yy",
    label: `$c_{y^2}$`,
    min: 0,
    max: "period",
  },
  period: { type: "int", alias: "p", label: "mod", min: 0, step: 16 },
  threshold: {
    type: "float",
    alias: "t",
    label: `$\\tau$`,
    min: 0,
    max: 1,
    step: 0.1,
    animateStep: 0.05,
  },
  // Canvas & shape
  width: {
    type: "int",
    alias: "w",
    label: `w`,
    min: 0,
    step: 32,
    shader: false,
  },
  height: {
    type: "int",
    alias: "h",
    label: `h`,
    min: 0,
    step: 32,
    shader: false,
  },
  shape_power: {
    type: "int",
    alias: "s",
    label: `$\\rho_s$`,
    min: 0,
    default: 0,
  },
  shape_gamma: {
    type: "float",
    alias: "g",
    label: `$\\gamma_s$`,
    default: 0,
    min: 0,
    step: 0.125,
  },
  color0: { type: "color", label: `color 0`, alias: "k" },
  color1: { type: "color", label: `color 1`, alias: "j" },
  // Animation
  animate: { type: "option", default: null, shader: false },
  fps: { type: "float", alias: "r", min: 0, max: 30, shader: false },
};

// Parsing

function parseValue(s, spec) {
  if (spec.type === "int" || spec.type === "float") {
    const v = spec.type === "int" ? Number.parseInt(s) : Number.parseFloat(s);
    // Can't apply {max: "period"} here, as period may not be parsed yet
    return Math.min(
      Math.max(v, spec.min ?? -Infinity),
      spec.max === undefined || spec.max === "period" ? Infinity : spec.max
    );
  }
  if (spec.type === "color") {
    const i = parseInt(s.replace("#", ""), 16);
    return [(i >> 16) & 0xff, (i >> 8) & 0xff, i & 0xff];
  }
  if (spec.type == "option") {
    return s;
  }
  throw Error(`Unknown spec.type ${spec.type}`);
}

function printValue(v, spec, colorPrefix) {
  if (spec.type === "color") {
    return (
      (colorPrefix || "") +
      v.map((x) => x.toString(16).padStart(2, "0")).join("")
    );
  }
  if (spec.type === "float") {
    return v.toFixed(3);
  }
  return String(v);
}

function parseConfigStr(s) {
  const aliasToKey = Object.fromEntries(
    Object.entries(CONFIG_SCHEMA).map(([k, v]) => [v.alias, k])
  );
  const result = {};
  for (let [key, spec] of Object.entries(CONFIG_SCHEMA)) {
    if ("default" in spec) {
      result[key] = spec.default;
    }
  }
  for (let [_, k, v] of s.matchAll(
    new RegExp("([g-z]+)([0-9a-f.#]+|\\*)", "g")
  )) {
    const key = aliasToKey[k];
    if (v === "*") {
      result[key] = 0;
      result["animate"] = key;
    } else {
      result[key] = parseValue(v, CONFIG_SCHEMA[key]);
    }
  }
  return result;
}

function printConfigStr(config) {
  let s = "";
  for (let [key, value] of Object.entries(config)) {
    const spec = CONFIG_SCHEMA[key];
    if (spec.alias !== undefined) {
      const v = key == config.animate ? "*" : printValue(value, spec);
      s += `${spec.alias}${v}`;
    }
  }
  return s;
}

function getWindowLocationHash() {
  const h = new URL(window.location.href).searchParams.get("h");
  return h ? decodeURIComponent(h) : null;
}

function setWindowLocationHash(config) {
  const url = new URL(window.location.href);
  url.searchParams.set("h", encodeURIComponent(printConfigStr(config)));
  window.history.replaceState(null, "", url.toString());
}

// Rendering

function hashFnEquation(c) {
  let parts = [];
  for (let [n, v] of [
    [c.c_x, "x"],
    [c.c_y, "y"],
    [c.c_xy, "x y"],
    [c.c_xx, "x^2"],
    [c.c_yy, "y^2"],
  ]) {
    if (n === 1) {
      parts.push(v);
    } else if (n !== 0) {
      parts.push(`${n} ${v}`);
    }
  }
  if (parts.length === 0) {
    parts.push(`0`);
  }
  return (
    `$$ (${parts.join(" + ")}) \\,\\mathrm{mod}\\, ${c.period}` +
    ` <  ${c.threshold.toFixed(2)} \\cdot ${c.period} $$`
  );
}

// Returns a function render(config)
function renderer(root, scrollbarWidth) {
  const eqn = root.querySelector(".hv-equation");
  const eqnCache = {};
  const canvas = root.querySelector(".hv-screen");
  const controlLabels = {};
  root.querySelectorAll(".hv-control").forEach((c) => {
    controlLabels[c.dataset.key] = c.querySelector(".hv-value");
  });
  const gl = canvas.getContext("webgl");
  if (gl === null) {
    throw new Error(
      "Couldn't set up WebGL - maybe unsupported by your browser."
    );
  }
  gl.clearColor(0.0, 0.0, 0.0, 0.0);

  // Shaders
  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const msg = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw Error(`Shader compile error: ${msg}`);
    }
    return shader;
  }
  const vertexShader = compileShader(
    gl.VERTEX_SHADER,
    `
    attribute vec2 position;
    void main() {
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `
  );
  const fragmentShader = compileShader(
    gl.FRAGMENT_SHADER,
    `
    precision highp float;
    precision highp int;

    uniform int c_x, c_y, c_xy, c_xx, c_yy;
    uniform int period;
    uniform float threshold;
    uniform vec4 color0, color1;
    uniform int shape_power;
    uniform float shape_gamma;

    uniform int scale, width, height;

    void main() {
      int x = int(gl_FragCoord.x) / scale;
      int y = int(gl_FragCoord.y) / scale;

      int h = c_x * x + c_y * y + c_xy * x*y + c_xx * x*x + c_yy * y*y;
      h -= period * (h / period);
      vec4 color = float(h) < threshold * float(period) ? color0 : color1;

      if (shape_power != 0) {
        float z = pow(
          pow(abs(float(2 * x) / float(width) - 1.0), float(shape_power))
          + pow(abs(float(2 * y) / float(height) - 1.0), float(shape_power)),
          1.0 / float(shape_power)
        );
        float a = max(1.0 - z, 0.0);
        color.a *= shape_gamma == 0.0 ? float(a > 0.0) : pow(a, shape_gamma);
      }
      gl_FragColor = color;
    }
  `
  );
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const msg = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw Error(`Program link error: ${msg}`);
  }
  gl.useProgram(program);

  // Shape
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW
  );
  const positionLocation = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

  // Draw
  return (config) => {
    // URL
    if (root.classList.contains("hv-url")) {
      setWindowLocationHash(config);
    }

    // Equation
    if (eqn !== null) {
      // When animating, calling typeset() for every frame causes a memory leak,
      // which typesetClear() doesn't seem to fix, so we cache previously typeset
      // equations here.
      const eqnText = hashFnEquation(config);
      if (eqnText in eqnCache) {
        eqn.replaceChildren(eqnCache[eqnText]);
      } else {
        const e = document.createElement("span");
        e.innerHTML = eqnText;
        eqn.replaceChildren(e);
        MathJax.typeset([e]);
        eqnCache[eqnText] = e;
      }
    }

    // Control labels
    for (let [key, e] of Object.entries(controlLabels)) {
      e.innerText = printValue(config[key], CONFIG_SCHEMA[key], "#");
    }

    // Implement hysteresis in the scaling, otherwise it can get into a loop
    // of showing/hiding the scrollbars
    const targetWidth =
      canvas.width < root.offsetWidth - scrollbarWidth
        ? root.offsetWidth - scrollbarWidth
        : root.offsetWidth;
    const scale = Math.max(1, Math.floor(targetWidth / config.width));

    // Texture
    canvas.width = scale * config.width;
    canvas.height = scale * config.height;
    gl.uniform1i(gl.getUniformLocation(program, "scale"), scale);
    gl.uniform1i(gl.getUniformLocation(program, "width"), config.width);
    gl.uniform1i(gl.getUniformLocation(program, "height"), config.height);
    for (let [k, spec] of Object.entries(CONFIG_SCHEMA)) {
      if (spec.shader === undefined || spec.shader) {
        let loc = gl.getUniformLocation(program, k);
        if (spec.type === "int") {
          gl.uniform1i(loc, config[k]);
        } else if (spec.type === "float") {
          gl.uniform1f(loc, config[k]);
        } else if (spec.type === "color") {
          const c = config[k];
          gl.uniform4f(loc, c[0] / 255, c[1] / 255, c[2] / 255, 1.0);
        } else {
          throw Error(`Unsupported type ${spec.type}`);
        }
      }
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };
}

// HTML Component

function createNode(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  return template.content.firstChild;
}

function createControls(root, config) {
  const controls = createNode(`<div class="hv-controls">`);

  // Top row controls, colors, buttons
  const topRow = createNode(`<div>`);
  for (let [key, spec] of Object.entries(CONFIG_SCHEMA)) {
    if (spec.label) {
      if (spec.type === "color") {
        const input = createNode(
          `<input class="hv-color" type="color" title="${spec.label}"/>`
        );
        input.value = printValue(config[key], spec, "#");
        input.dataset.key = key;
        topRow.appendChild(input);
      } else {
        const control = createNode(`<div class="hv-control" tabindex="0">`);
        control.appendChild(
          createNode(`<div class="hv-label">${spec.label}</div>`)
        );
        control.appendChild(
          createNode(`<div class="hv-value">${config[key]}</div>`)
        );
        control.dataset.key = key;
        topRow.appendChild(control);
      }
    }
  }
  topRow.appendChild(
    createNode(`<div class="hv-reset hv-button">\u21ba</div>`)
  );
  topRow.appendChild(
    createNode(`<div class="hv-download hv-button">\u2913</div>`)
  );
  if (root.classList.contains("hv-url") && navigator.clipboard) {
    topRow.appendChild(
      createNode(`<div class="hv-share hv-button">\u{260D}</div>`)
    );
  }
  controls.appendChild(topRow);

  // Common (shared) controls
  const common = createNode(`<div class="hv-controls-common">`);
  common.style.display = "none";
  common.appendChild(
    createNode(`<div class="hv-animate hv-button">\u25B6</div>`)
  );
  common.appendChild(
    createNode(`<span class="hv-animate-fps-label">&nbsp;fps&nbsp;</span>`)
  );
  common.appendChild(
    createNode(`<div class="hv-slider-dec hv-button">\u2212</div>`)
  );
  common.appendChild(createNode(`<input class="hv-slider" type="range"/>`));
  common.appendChild(
    createNode(`<div class="hv-slider-inc hv-button">+</div>`)
  );
  controls.appendChild(common);

  return controls;
}

function configureCommon(root, config, key) {
  const slider = root.querySelector(".hv-slider");
  const animate = root.querySelector(".hv-animate");
  const animateFpsLabel = root.querySelector(".hv-animate-fps-label");
  const isAnimating = config.animate === key;

  const skey = isAnimating ? "fps" : key;
  const sspec = CONFIG_SCHEMA[skey];
  slider.type =
    sspec.min !== undefined && sspec.max != undefined ? "range" : "number";
  slider.min = sspec.min;
  slider.max = typeof sspec.max === "string" ? config[sspec.max] : sspec.max;
  slider.step = sspec.step ?? 1;
  slider.value = config[skey];
  slider.dataset.key = skey;

  const spec = CONFIG_SCHEMA[key];
  animate.style.display =
    spec.min !== undefined && spec.max != undefined ? "inline-block" : "none";
  animate.dataset.key = key;
  animate.innerText = isAnimating ? `\u25A0` : `\u25B6`;
  animateFpsLabel.style.display = isAnimating ? "inline-block" : "none";
}

function hvInit(root, scrollbarWidth) {
  const config = parseConfigStr(getWindowLocationHash() ?? root.dataset.hvInit);

  if (root.classList.contains("hv-show-controls")) {
    root.appendChild(createControls(root, config));
    MathJax.typesetPromise();
  }
  if (root.classList.contains("hv-show-equation")) {
    root.appendChild(createNode(`<div class="hv-equation"></div>`));
  }
  root.appendChild(createNode(`<canvas class="hv-screen"></canvas>`));

  const render = renderer(root, scrollbarWidth);
  render(config);
  new ResizeObserver(() => {
    render(config);
  }).observe(root);

  // Shared controls
  root.querySelectorAll(".hv-slider").forEach((c) => {
    c.addEventListener("change", () => {
      config[c.dataset.key] = parseValue(c.value, CONFIG_SCHEMA[c.dataset.key]);
      render(config);
    });
  });
  root.querySelectorAll(".hv-control").forEach((c) => {
    c.addEventListener("click", (e) => {
      // Select/deselect this config parameter
      const toggleOn = c.dataset.selected !== "true";
      root.querySelectorAll(".hv-control").forEach((other) => {
        other.dataset.selected = false;
      });
      c.dataset.selected = toggleOn;

      // Configure the common controls panel
      const common = root.querySelector(".hv-controls-common");
      common.style.display = toggleOn ? "inline-block" : "none";
      if (toggleOn) {
        configureCommon(root, config, c.dataset.key);
      }
      e.stopPropagation();
    });
  });

  // Buttons
  root.querySelectorAll(".hv-slider-inc,.hv-slider-dec").forEach((c) => {
    c.addEventListener("click", () => {
      root.querySelectorAll(".hv-slider").forEach((slider) => {
        const delta =
          slider.step * (2 * +c.classList.contains("hv-slider-inc") - 1);
        slider.value = Math.min(
          Math.max(
            parseFloat(slider.value) + delta,
            slider.min === "undefined" ? -Infinity : parseFloat(slider.min)
          ),
          slider.max === "undefined" ? Infinity : parseFloat(slider.max)
        );
        slider.dispatchEvent(new Event("change"));
      });
    });
  });
  root.querySelectorAll(".hv-animate").forEach((c) => {
    c.addEventListener("click", () => {
      const slider = root.querySelector(".hv-slider");
      config["animate"] =
        config["animate"] !== c.dataset.key ? c.dataset.key : null;
      configureCommon(root, config, c.dataset.key);
    });
  });
  root.querySelectorAll(".hv-color").forEach((c) => {
    c.addEventListener("change", () => {
      config[c.dataset.key] = parseValue(c.value, CONFIG_SCHEMA[c.dataset.key]);
      render(config);
    });
  });
  root.querySelectorAll(".hv-download").forEach((download) => {
    download.addEventListener("click", () => {
      render(config);
      const a = document.createElement("a");
      a.download = "hash_texture.png";
      a.href = document.querySelector(".hv-screen").toDataURL("image/png");
      a.click();
    });
  });
  root.querySelectorAll(".hv-share").forEach((share) => {
    share.addEventListener("click", () => {
      navigator.clipboard
        .writeText(window.location.href)
        .then(() => {
          alert("Link copied to clipboard");
        })
        .catch((err) => {
          console.error("Failed to copy URL: ", err);
        });
    });
  });
  root.querySelectorAll(".hv-reset").forEach((reset) => {
    reset.addEventListener("click", () => {
      Object.assign(config, parseConfigStr(root.dataset.hvInit));
      root.querySelectorAll(".hv-color").forEach((color) => {
        color.value = printValue(
          config[color.dataset.key],
          CONFIG_SCHEMA[color.dataset.key],
          "#"
        );
      });
      root.querySelectorAll(".hv-control").forEach((c) => {
        c.dataset.selected = false;
      });
      root.querySelector(".hv-controls-common").style.display = "none";
      render(config);
    });
  });

  // Animation loop
  let lastUpdate = null;
  function onFrame(time) {
    if (
      config.animate &&
      (lastUpdate === null || time - lastUpdate >= 1000 / config.fps)
    ) {
      const spec = CONFIG_SCHEMA[config.animate];
      const specMax =
        typeof spec.max === "string" ? config[spec.max] : spec.max;
      let value = config[config.animate];
      value += spec.animateStep ?? 1;
      value = spec.min + ((value - spec.min) % (specMax - spec.min));
      config[config.animate] = value;
      render(config);
      lastUpdate = time;
    }
    requestAnimationFrame(onFrame);
  }
  requestAnimationFrame(onFrame);
}

// HTML Top-level

// Measure the default scroll bar width in pixels, empirically
function getScrollbarWidth() {
  const div = document.createElement("div");
  div.style.position = "absolute";
  div.style.top = "-9999px";
  div.style.width = "100px";
  div.style.height = "100px";
  div.style.overflow = "scroll";
  document.body.appendChild(div);
  try {
    return div.offsetWidth - div.clientWidth;
  } finally {
    document.body.removeChild(div);
  }
}

window.addEventListener("load", () => {
  const scrollbarWidth = getScrollbarWidth();
  for (const root of document.querySelectorAll(".hv")) {
    hvInit(root, scrollbarWidth);
  }
});
