  // ---- shared by the live panel (panel.html) and the recorded one (panel_static.html); included inside their script.
  // T holds the strings.  A "side" is an object with canvas, geo (geometry), traj (rows [pusher x, pusher y,
  // block x, block y, block theta], or null), success, limit (steps), pos (step being shown), raf.
  const $ = (id) => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  const STEPS_PER_SECOND = 60;      // replay speed: six times the 10 Hz of the task
  const TRAIL = 45;                 // steps of pusher path kept on screen
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  const fill = (template, values) => Object.keys(values).reduce((s, k) => s.split("{" + k + "}").join(values[k]), template);
  const pct = (v) => Math.round(v * 1000) / 10 + "%";         // one decimal where there is one: 20%, 84.5%

  // ------------------------------------------------------------------ stage (canvas)
  function colours() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    return { muted: v("--muted"), rule: v("--rule"), accent: v("--accent"), good: v("--good") };     // follows the theme
  }

  function outline(ctx, pts, pose, k) {       // body-frame polygon at pose (x, y, theta); screen y points down, as in the videos
    const c = Math.cos(pose[2]), sn = Math.sin(pose[2]);
    ctx.beginPath();
    pts.forEach(function (p, i) {
      const x = (pose[0] + c * p[0] - sn * p[1]) * k, y = (pose[1] + sn * p[0] + c * p[1]) * k;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.closePath();
  }

  function frameAt(traj, pos) {               // linear interpolation between steps (shortest way round for the angle)
    const i = Math.min(traj.length - 1, Math.floor(pos)), j = Math.min(traj.length - 1, i + 1), u = pos - i;
    const a = traj[i], b = traj[j];
    let d = b[4] - a[4];
    if (d > Math.PI) d -= 2 * Math.PI; else if (d < -Math.PI) d += 2 * Math.PI;
    return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u, a[4] + d * u];
  }

  function draw(s) {
    const dpr = window.devicePixelRatio || 1;
    const px = Math.max(1, Math.round(s.canvas.getBoundingClientRect().width * dpr));
    if (s.canvas.width !== px || s.canvas.height !== px) { s.canvas.width = px; s.canvas.height = px; }
    const ctx = s.canvas.getContext("2d"), geo = s.geo;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, px, px);
    if (!geo) return;
    const C = colours(), k = px / geo.workspace[0];      // the stage is the whole workspace
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    geo.block.forEach(function (pts) {        // goal pose
      outline(ctx, pts, geo.goal, k);
      ctx.globalAlpha = 0.14; ctx.fillStyle = C.good; ctx.fill();
      ctx.globalAlpha = 1; ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.strokeStyle = C.good; ctx.lineWidth = 1.5 * dpr; ctx.stroke();
      ctx.setLineDash([]);
    });
    if (!s.traj) return;
    const last = s.traj.length - 1, f = frameAt(s.traj, s.pos), ended = s.pos >= last;
    geo.block.forEach(function (pts) {        // block
      outline(ctx, pts, [f[2], f[3], f[4]], k);
      ctx.globalAlpha = 0.86; ctx.fillStyle = ended && s.success ? C.good : C.muted; ctx.fill();
      ctx.globalAlpha = 1;
    });
    const i1 = Math.floor(s.pos), i0 = Math.max(0, i1 - TRAIL);       // recent pusher path, in pieces of rising opacity
    ctx.strokeStyle = C.accent;
    ctx.lineWidth = 2 * dpr;
    ctx.lineCap = "butt";
    for (let a = i0; ; a += 5) {
      const b = Math.min(i1, a + 5);
      ctx.globalAlpha = 0.08 + 0.5 * (b - i0 + 1) / (i1 - i0 + 1);
      ctx.beginPath();
      ctx.moveTo(s.traj[a][0] * k, s.traj[a][1] * k);
      for (let i = a + 1; i <= b; i++) ctx.lineTo(s.traj[i][0] * k, s.traj[i][1] * k);
      if (b === i1) ctx.lineTo(f[0] * k, f[1] * k);
      ctx.stroke();
      if (b === i1) break;
    }
    ctx.globalAlpha = 1;
    ctx.beginPath();                          // pusher
    ctx.arc(f[0] * k, f[1] * k, geo.pusher_radius * k, 0, 2 * Math.PI);
    ctx.fillStyle = C.accent; ctx.fill();
    ctx.fillStyle = C.rule;                   // elapsed share of the step limit, along the bottom edge
    ctx.fillRect(0, px - 3 * dpr, px, 3 * dpr);
    ctx.fillStyle = ended && s.success ? C.good : C.muted;
    ctx.fillRect(0, px - 3 * dpr, px * Math.min(1, s.pos / s.limit), 3 * dpr);
  }

  function play(s) {
    cancelAnimationFrame(s.raf);
    if (!s.traj) return;
    const last = s.traj.length - 1;
    let before = 0;
    s.pos = 0;
    const tick = function (now) {
      const dt = before ? Math.min(0.5, (now - before) / 1000) : 0;       // a hidden tab resumes where it stopped
      before = now;
      s.pos = Math.min(last, s.pos + dt * STEPS_PER_SECOND);
      draw(s);
      if (s.pos < last) s.raf = requestAnimationFrame(tick);
    };
    s.raf = requestAnimationFrame(tick);
  }

  function show(s) {                          // a new trajectory: play it once, or show its last frame if motion is unwanted
    cancelAnimationFrame(s.raf);
    if (s.traj && !reduced.matches) play(s);
    else { s.pos = s.traj ? s.traj.length - 1 : 0; draw(s); }
  }

  function keepDrawn(sides) {                 // redraw when the stage is resized or the theme changes
    const redraw = function () { sides.forEach(draw); };
    if ("ResizeObserver" in window) sides.forEach(function (s) { new ResizeObserver(function () { draw(s); }).observe(s.canvas); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  // ------------------------------------------------------------------ chart frame (x: world model, y: environment)
  function node(name, attrs, parent) {
    const e = document.createElementNS(SVG, name);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    parent.appendChild(e);
    return e;
  }

  const PLOT = { x0: 52, y0: 12, size: 250 };
  const plotX = (v) => PLOT.x0 + v * PLOT.size, plotY = (v) => PLOT.y0 + (1 - v) * PLOT.size;
  const pointText = (version, world, real) => "v" + version + ": " + T.world_title + " " + pct(world) + ", " + T.real_title + " " + pct(real);
  const labelSide = (cx) => cx > PLOT.x0 + PLOT.size - 30 ? { dx: -9, anchor: "end" } : { dx: 9, anchor: "start" };

  function chartFrame(svg) {
    const X0 = PLOT.x0, Y0 = PLOT.y0, S = PLOT.size;
    svg.textContent = "";
    [0, 0.5, 1].forEach(function (v) {
      node("line", { class: "wm-line", x1: X0, y1: plotY(v), x2: X0 + S, y2: plotY(v) }, svg);
      node("line", { class: "wm-line", x1: plotX(v), y1: Y0, x2: plotX(v), y2: Y0 + S }, svg);
      node("text", { class: "wm-tk", x: X0 - 7, y: plotY(v) + 3.5, "text-anchor": "end" }, svg).textContent = pct(v);
      node("text", { class: "wm-tk", x: plotX(v), y: Y0 + S + 15, "text-anchor": "middle" }, svg).textContent = pct(v);
    });
    node("line", { class: "wm-diag", x1: plotX(0), y1: plotY(0), x2: plotX(1), y2: plotY(1) }, svg);
    node("text", { class: "wm-ax", x: X0 + S / 2, y: Y0 + S + 34, "text-anchor": "middle" }, svg).textContent = T.chart_x;
    node("text", { class: "wm-ax", x: 11, y: Y0 + S / 2, "text-anchor": "middle",
                   transform: "rotate(-90 11 " + (Y0 + S / 2) + ")" }, svg).textContent = T.chart_y;
  }

  // ------------------------------------------------------------------ task and program chips
  function chip(group, value, text, cls) {
    const l = document.createElement("label"), i = document.createElement("input"), t = document.createElement("span");
    l.className = "wm-chip" + (cls ? " " + cls : "");
    i.type = "radio"; i.name = group; i.value = value;
    t.textContent = text;
    l.appendChild(i); l.appendChild(t);
    return l;
  }

  function listTasks(tasks) {
    tasks.forEach(function (t) { $("wm-task").appendChild(chip("wm-task", t, T["task_" + t] || t, "")); });
  }

  function listVersions(programs) {         // the version chips of a task and the line naming its tagged versions
    const box = $("wm-version"), tags = $("wm-tags");
    box.textContent = "";
    tags.textContent = "";
    programs.forEach(function (p) {
      const c = chip("wm-version", String(p.version), "v" + p.version, "wm-v");
      const tag = p.tag ? T["tag_" + p.tag] : "";
      if (tag) {
        c.dataset.tag = p.tag;
        c.title = tag;
        c.firstChild.setAttribute("aria-label", "v" + p.version + ", " + tag);
        const item = document.createElement("span"), b = document.createElement("b");
        b.textContent = "v" + p.version;
        item.appendChild(b);
        item.appendChild(document.createTextNode(tag));
        tags.appendChild(item);
      }
      box.appendChild(c);
    });
  }

  function checkChips(task, version) {
    $("wm-version").querySelectorAll("input").forEach(function (i) { i.checked = Number(i.value) === version; });
    $("wm-task").querySelectorAll("input").forEach(function (i) { i.checked = i.value === task; });
  }

  const submitted = (programs) => (programs.find(function (p) { return p.tag === "submitted"; }) || programs[0]).version;
