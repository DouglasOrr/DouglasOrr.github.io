// Schema

const CONFIG_SCHEMA = {
  // Hash
  c_x: { type: "int", alias: "x" },
  c_y: { type: "int", alias: "y" },
  c_xy: { type: "int", alias: "xy" },
  c_xx: { type: "int", alias: "xx" },
  c_yy: { type: "int", alias: "yy" },
  period: { type: "int", alias: "p", min: 1 },
  threshold: { type: "float", alias: "t", min: 0, max: 1, step: 0.1 },
  // Shape
  shape_power: { type: "int", alias: "s", min: 0 },
  shape_gamma: { type: "float", alias: "g", min: 0, step: 0.125 },
  // Canvas
  width: { type: "int", alias: "w", min: 1 },
  height: { type: "int", alias: "h", min: 1 },
  color0: { type: "color", alias: "k" },
  color1: { type: "color", alias: "j" },
  // Animation
  animate: {
    type: "option",
    options: [null, "c_x", "c_y", "c_xy", "c_xx", "c_yy", "threshold"],
    default: null,
    shader: false,
  },
  fps: { type: "float", default: 5, shader: false },
};

const CONTROLS = [
  ["Hash", ["c_x", "c_y", "c_xy", "c_xx", "c_yy", "period", "threshold"]],
  [
    "Draw",
    ["shape_power", "shape_gamma", "width", "height", "color0", "color1"],
  ],
  ["Animate", ["animate", "fps"]],
];

// Parsing

function parseValue(s, spec) {
  if (spec.type === "int" || spec.type === "float") {
    const v = spec.type === "int" ? Number.parseInt(s) : Number.parseFloat(s);
    return Math.min(Math.max(v, spec.min ?? -Infinity), spec.max ?? Infinity);
  }
  if (spec.type === "color") {
    const i = parseInt(s.replace("#", ""), 16);
    return [(i >> 16) & 0xff, (i >> 8) & 0xff, i & 0xff];
  }
}

function printValue(v, spec) {
  if (spec.type == "color") {
    return "#" + v.map((x) => x.toString(16).padStart(2, "0")).join("");
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
  for (let [_, k, v] of s.matchAll(new RegExp("([g-z]+)([0-9a-f.]+)", "g"))) {
    const key = aliasToKey[k];
    result[key] = parseValue(v, CONFIG_SCHEMA[key]);
  }
  return result;
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
  return (
    `$$ (${parts.join(" + ")}) \\,\\mathrm{mod}\\, ${c.period}` +
    ` <  ${c.threshold.toFixed(2)} \\cdot ${c.period} $$`
  );
}

// Returns a function render(config)
function renderer(root) {
  const eqn = root.querySelector(".hv-equation");
  const canvas = root.querySelector(".hv-screen");
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
    precision mediump float;

    uniform int c_x, c_y, c_xy, c_xx, c_yy;
    uniform int period;
    uniform float threshold;
    uniform vec4 color0, color1;
    uniform int shape_power;
    uniform float shape_gamma;

    uniform int width, height;

    void main() {
      int x = int(gl_FragCoord.x);
      int y = int(gl_FragCoord.y);
      float h = float(c_x * x + c_y * y + c_xy * x*y + c_xx * x*x + c_yy * y*y);
      vec4 color = mod(h / float(period), 1.0) < threshold ? color0 : color1;

      if (shape_power != 0) {
        float z = pow(
          pow(abs(float(2 * x) / float(width) - 1.0), float(shape_power))
          + pow(abs(float(2 * y) / float(height) - 1.0), float(shape_power)),
          1.0 / float(shape_power)
        );
        color.a *= pow(max(1.0 - z, 0.0), shape_gamma);
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
    // Equation
    if (eqn !== null) {
      eqn.innerHTML = hashFnEquation(config);
      MathJax.typesetClear([eqn]);
      MathJax.typeset([eqn]);
    }
    // Texture
    canvas.width = config.width;
    canvas.height = config.height;
    gl.viewport(0, 0, canvas.width, canvas.height);
    // const scale = Math.max(
    //   1,
    //   Math.floor(
    //     Math.min(
    //       root.offsetWidth / canvas.width,
    //       root.offsetHeight / canvas.height
    //     )
    //   )
    // );
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
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };
}

// HTML

function createNode(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  return template.content.firstChild;
}

function hvInit(root) {
  const config = parseConfigStr(root.dataset.hvInit);

  root.appendChild(createNode(`<canvas class="hv-screen"></canvas>`));

  if (root.classList.contains("hv-show-equation")) {
    root.appendChild(createNode(`<div class="hv-equation"></div>`));
  }

  if (root.classList.contains("hv-show-controls")) {
    let controls = createNode(`<div class="hv-controls">`);
    for (let [groupTitle, keys] of CONTROLS) {
      const row = createNode("<p>");
      row.appendChild(createNode(groupTitle));
      for (let key of keys) {
        const spec = CONFIG_SCHEMA[key];
        if (spec.type === "option") {
          const select = createNode(
            `<select class="hv-control" title="${key}" name="${key}">`
          );
          for (const k2 of spec.options) {
            select.appendChild(
              createNode(`<option value="${k2}">${k2}</option>`)
            );
          }
          row.appendChild(select);
        } else {
          const input = createNode(
            `<input class="hv-control" title="${key}" name="${key}" />`
          );
          input.setAttribute("value", printValue(config[key], spec));
          input.setAttribute(
            "type",
            spec.type === "color" ? "color" : "number"
          );
          for (let attr of ["min", "max", "step"]) {
            if (attr in spec) {
              input.setAttribute(attr, spec[attr]);
            }
          }
          row.appendChild(input);
        }
      }
      controls.appendChild(row);
    }
    controls.appendChild(
      createNode(`<input class="hv-reset" type="button" value="Reset"/>`)
    );
    root.appendChild(controls);
  }

  const render = renderer(root);
  render(config);

  // Control wiring
  root.querySelectorAll(".hv-control").forEach((c) => {
    c.addEventListener("change", () => {
      config[c.name] = parseValue(c.value, CONFIG_SCHEMA[c.name]);
      render(config);
    });
  });
  root.querySelectorAll(".hv-reset").forEach((reset) => {
    reset.addEventListener("click", () => {
      Object.assign(config, parseConfigStr(root.dataset.hvInit));
      root.querySelectorAll(".hv-control").forEach((c) => {
        c.value = printValue(config[c.name], CONFIG_SCHEMA[c.name]);
      });
      render(config);
    });
  });
  if (root.classList.contains("hv-show-controls")) {
    let lastUpdate = null;
    const animate = root.querySelector(`.hv-control[name="animate"]`);
    const fps = root.querySelector(`.hv-control[name="fps"]`);
    function onFrame(time) {
      if (animate.value !== "null") {
        if (lastUpdate === null || time - lastUpdate >= 1000 / fps.value) {
          const input = root.querySelector(
            `.hv-control[name="${animate.value}"]`
          );
          if (animate.value === "threshold") {
            input.value = (Number(input.value) + 0.05) % 1;
          } else {
            const period = root.querySelector(`.hv-control[name="period"]`);
            input.value = (Number(input.value) + 1) % Number(period.value);
          }
          input.dispatchEvent(new Event("change"));
          lastUpdate = time;
        }
      }
      requestAnimationFrame(onFrame);
    }
    requestAnimationFrame(onFrame);
  }
}

window.addEventListener("load", () => {
  for (const root of document.querySelectorAll(".hv")) {
    hvInit(root);
  }
});
