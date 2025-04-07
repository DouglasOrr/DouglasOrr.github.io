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
  shape_power: { type: "int", alias: "s", min: 1 },
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
  },
  fps: { type: "float", default: 5 },
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

// Core

function hashFn(x, y, c) {
  const z =
    c.c_x * x + c.c_y * y + c.c_xy * x * y + c.c_xx * x * x + c.c_yy * y * y;
  return z % c.period < c.period * c.threshold;
}

function shapeFn(x, y, c) {
  const z =
    (Math.abs((2 * x) / c.width - 1) ** c.shape_power +
      Math.abs((2 * y) / c.height - 1) ** c.shape_power) **
    (1 / c.shape_power);
  return Math.max(1 - z, 0) ** c.shape_gamma;
}

function drawFn(hash, shape, c) {
  const color = hash ? c.color0 : c.color1;
  return [...color, shape * 255];
}

// Rendering

function hashFnEquation(c) {
  // return (
  //   `(${c.c_x}*x + ${c.c_y}*y + ${c.c_xy}*x*y + ${c.c_xx}*x*x + ${c.c_yy}*y*y)` +
  //   ` % ${c.period} < ${c.threshold.toFixed(2)} * ${c.period}`
  // );
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

function render(root, c) {
  // Equation
  const eqn = root.querySelector(".hv-equation");
  if (eqn !== null) {
    eqn.innerHTML = hashFnEquation(c);
    MathJax.typesetClear([eqn]);
    MathJax.typeset([eqn]);
    // MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
  }

  // Texture
  // const scale = Math.max(
  //   1,
  //   Math.floor(
  //     Math.min(
  //       root.offsetWidth / canvas.width,
  //       root.offsetHeight / canvas.height
  //     )
  //   )
  // );
  const canvas = root.querySelector(".hv-screen");
  canvas.width = c.width;
  canvas.height = c.height;
  const ctx = canvas.getContext("2d");
  const data = new ImageData(c.width, c.height, {
    colorSpace: "srgb",
  });
  for (let y = 0; y < c.height; ++y) {
    for (let x = 0; x < c.width; ++x) {
      const h = hashFn(x, y, c);
      const s = shapeFn(x, y, c);
      const value = drawFn(h, s, c);
      data.data.set(value, 4 * (y * c.width + x));
    }
  }
  ctx.putImageData(data, 0, 0);
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

  render(root, config);

  // Control wiring
  root.querySelectorAll(".hv-control").forEach((c) => {
    c.addEventListener("change", () => {
      config[c.name] = parseValue(c.value, CONFIG_SCHEMA[c.name]);
      render(root, config);
    });
  });
  root.querySelectorAll(".hv-reset").forEach((reset) => {
    reset.addEventListener("click", () => {
      Object.assign(config, parseConfigStr(root.dataset.hvInit));
      root.querySelectorAll(".hv-control").forEach((c) => {
        c.value = printValue(config[c.name], CONFIG_SCHEMA[c.name]);
      });
      render(root, config);
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
