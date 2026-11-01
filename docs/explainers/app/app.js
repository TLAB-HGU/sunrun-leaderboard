/* ===========================================================================
   Isometric walkthrough engine in the style of bbycroft.net/llm:
   fixed isometric projection (SVG polygons) + upright monospace labels,
   phase chips + vertical timeline + step commentary + camera pad.
   =========================================================================== */
(function () {
  'use strict';
  const C30 = Math.cos(Math.PI / 6), S30 = Math.sin(Math.PI / 6);
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const nf = (v, d) => (v == null || !isFinite(v) ? '—'
    : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));

  /* ---------- colour helpers ---------- */
  function hex2rgb(h) {
    const s = h.replace('#', '');
    return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
  }
  function shade(h, f) {
    const [r, g, b] = hex2rgb(h);
    const m = (c) => Math.max(0, Math.min(255, Math.round(f > 0 ? c + (255 - c) * f : c * (1 + f))));
    return `rgb(${m(r)},${m(g)},${m(b)})`;
  }

  /* ---------- isometric maths ---------- */
  const P = (x, y, z) => [(x - z) * C30, (x + z) * S30 - y];

  function cuboid(b) {
    const { x, y, z, w, h, d } = b;
    const x0 = x - w / 2, x1 = x + w / 2, y0 = y - h / 2, y1 = y + h / 2;
    const z0 = z - d / 2, z1 = z + d / 2;
    return {
      top: [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)],
      right: [P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1), P(x1, y0, z0)],
      front: [P(x0, y1, z1), P(x1, y1, z1), P(x1, y0, z1), P(x0, y0, z1)]
    };
  }

  /* ---------- box builders used by page specs ---------- */
  function row(n, o) {
    const out = [];
    const w = o.w || 9, gap = o.gap == null ? 3 : o.gap;
    const step = w + gap;
    const x0 = -(n - 1) * step / 2;
    for (let i = 0; i < n; i++) {
      out.push({
        x: x0 + i * step, y: o.y || 0, z: o.z || 0, w, h: o.h || 9, d: o.d || 9,
        color: o.color, id: o.id ? `${o.id}:${i}` : null,
        label: o.labels ? o.labels[i] : null,
        value: o.values ? o.values[i] : null,
        fmt: o.fmt == null ? 1 : o.fmt,
        tip: o.tip ? o.tip(i) : null,
        detail: o.detail ? o.detail(i) : null
      });
    }
    return out;
  }
  function grid(cols, rows, o) {
    const out = [];
    const w = o.w || 8, gap = o.gap == null ? 3 : o.gap;
    const sx = w + gap, sz = (o.d || 8) + gap;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        out.push({
          x: (c - (cols - 1) / 2) * sx, y: o.y || 0, z: (r - (rows - 1) / 2) * sz,
          w, h: o.h || 8, d: o.d || 8, color: o.color, id: o.id ? `${o.id}:${i}` : null,
          value: o.values ? o.values[i] : null, fmt: o.fmt == null ? 0 : o.fmt,
          tip: o.tip ? o.tip(i) : null, detail: o.detail ? o.detail(i) : null
        });
      }
    }
    return out;
  }
  function box(o) {
    return [{ x: o.x || 0, y: o.y || 0, z: o.z || 0, w: o.w || 40, h: o.h || 20, d: o.d || 20,
      color: o.color, id: o.id, label: o.label, value: o.value, fmt: o.fmt,
      tip: o.tip, detail: o.detail, big: true }];
  }

  /* ---------- app ---------- */
  function run(spec) {
    const svg = $('#scene'), lab = $('#labels'), cmt = $('#cmt'), toc = $('#toc'), tl = $('#tl');
    const stages = spec.stages;
    const camera = { zoom: 1, px: 0, py: 0, rx: 0 };
    let cur = -1, playing = false, timer = null, sel = null;

    /* -------- stage lookup -------- */
    const boxes = [];
    stages.forEach((st, si) => (st.boxes || []).forEach((b) => {
      b.stage = si; b.sid = st.id;
      boxes.push(b);
    }));

    /* -------- geometry: auto-frame the focused stage -------- */
    let frame = { s: 1, ox: 0, oy: 0 };
    function computeFrame(focus) {
      const r = svg.getBoundingClientRect();
      const bs = boxes.filter((b) => focus.has(b.sid));
      if (!bs.length || !r.width) return { s: 1, ox: r.width / 2, oy: r.height / 2 };
      let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      bs.forEach((b) => {
        const c = cuboid(b);
        [c.top, c.right, c.front].forEach((f) => f.forEach(([X, Y]) => {
          if (X < minX) minX = X; if (X > maxX) maxX = X;
          if (Y < minY) minY = Y; if (Y > maxY) maxY = Y;
        }));
      });
      // label headroom above the solid
      minY -= 26;
      /* keep the focused stage above the computation panel */
      const opEl = document.getElementById('op');
      const inset = (opEl && opEl.classList.contains('on') ? opEl.offsetHeight + 14 : 0);
      const availH = Math.max(120, r.height - inset);
      const s = Math.min(r.width * 0.90 / Math.max(maxX - minX, 1),
                         availH * 0.86 / Math.max(maxY - minY, 1)) * camera.zoom;
      return {
        s,
        ox: r.width / 2 - (minX + maxX) / 2 * s + camera.px,
        oy: availH / 2 - (minY + maxY) / 2 * s + camera.py
      };
    }
    function project(p) {
      return [frame.ox + p[0] * frame.s, frame.oy + p[1] * frame.s];
    }
    const pts = (arr) => arr.map((q) => project(q).map((v) => Math.round(v * 10) / 10).join(',')).join(' ');

    function render() {
      const step = spec.steps[cur] || spec.steps[0];
      const si = stages.findIndex((x) => x.id === step.stage);
      const st = stages[si] || stages[0];
      const focus = new Set(step.focus || [st.id]);
      frame = computeFrame(focus);
      /* draw the focused stage plus its neighbours only — the rest is depth */
      const near = new Set();
      [si - 1, si, si + 1].forEach((i) => { if (stages[i]) near.add(stages[i].id); });
      const solids = boxes.filter((b) => b.visible !== false && near.has(b.sid));
      const order = solids.slice().sort((a, b) => (a.z - b.z) || (a.x - b.x));
      let out = '';
      order.forEach((b) => {
        const c = cuboid(b);
        const inFocus = focus.has(b.sid);
        const op = inFocus ? 1 : 0.22;
        const col = b.color || '#c8d2e6';
        out += `<g class="bx" data-id="${b.id || ''}" data-stage="${b.sid}" opacity="${op}"
                 style="cursor:${b.detail ? 'pointer' : 'default'}">` +
          `<polygon points="${pts(c.front)}" fill="${shade(col, -0.16)}" stroke="#00000022"/>` +
          `<polygon points="${pts(c.right)}" fill="${shade(col, -0.04)}" stroke="#00000022"/>` +
          `<polygon points="${pts(c.top)}" fill="${shade(col, 0.22)}" stroke="#00000022"/>` +
          `</g>`;
      });
      svg.innerHTML = out;
      $$('g.bx[data-id]', svg).forEach((g) => {
        const b = boxes.find((x) => (x.id || '') === g.dataset.id && near.has(x.sid));
        if (!b || !b.detail) return;
        g.style.cursor = 'pointer';
        g.addEventListener('click', () => { sel = b; openCard(b); render(); });
      });

      /* labels (upright) */
      let L = '';
      const r = svg.getBoundingClientRect();
      const inside = (x, y) => x > -80 && x < r.width + 80 && y > -60 && y < r.height + 60;
      solids.forEach((b) => {
        const inFocus = focus.has(b.sid);
        if (!inFocus && !b.big) return;
        const [cx, cy] = project(P(b.x, b.y + (b.h || 9) / 2 + 3, b.z));
        if (!inside(cx, cy)) return;
        const bw = frame.s * (b.w || 9);
        if (!b.big && bw < 13) return;          // too small to read: no label
        if (b.big && b.label) {
          L += `<div class="lab title ${inFocus ? '' : 'dim'}" data-id="${b.id || ''}"
                  style="left:${cx}px;top:${cy - 14}px">${b.label}</div>`;
        }
        if (b.value != null) {
          const txt = typeof b.value === 'number'
            ? nf(b.value, b.fmt == null ? 1 : b.fmt) : String(b.value);
          L += `<div class="lab val ${inFocus ? '' : 'dim'}" data-id="${b.id || ''}"
                  style="left:${cx}px;top:${cy}px">${txt}</div>`;
        }
      });
      (st.labels || []).forEach((t) => {
        const [cx, cy] = project(P(t.x, t.y, t.z));
        L += `<div class="lab ${t.cls || ''}" style="left:${cx}px;top:${cy}px">${t.text}</div>`;
      });
      lab.innerHTML = L;
      if (sel) {
        $$('.lab', lab).forEach((e) => { if (e.dataset.id === sel.id) e.classList.add('hit'); });
      }
      $$('.lab.val[data-id]', lab).forEach((el) => {
        el.addEventListener('click', () => {
          const b = boxes.find((x) => (x.id || '') === el.dataset.id);
          if (b && b.detail) { sel = b; openCard(b); render(); }
        });
      });
      $('#hud').innerHTML = `<b>${spec.title}</b> · step ${cur + 1}/${spec.steps.length} · ` +
        `${step.title} · ${camera.zoom.toFixed(2)}×`;
    }

    /* -------- detail card -------- */
    function openCard(b) {
      const card = $('#card');
      card.classList.add('on');
      card.innerHTML = `<span class="x" id="cardX">×</span><h3>${b.tip || b.label || 'cell'}</h3>` +
        (b.detail.rows ? '<table>' + b.detail.rows.map(([k, v, hl]) =>
          `<tr><td>${k}</td><td class="${hl ? 'hl' : ''}">${v}</td></tr>`).join('') + '</table>' : '') +
        (b.detail.note ? `<div style="margin-top:6px;color:#666">${b.detail.note}</div>` : '');
      $('#cardX').addEventListener('click', () => { card.classList.remove('on'); sel = null; render(); });
    }

    /* -------- steps -------- */
    function go(k, fromUser) {
      k = Math.max(0, Math.min(spec.steps.length - 1, k));
      cur = k;
      const st = spec.steps[k];
      camera.zoom = st.zoom || 1;
      camera.px = 0;
      camera.py = 0;
      $$('.phase', toc).forEach((p, i) => p.classList.toggle('on', i === k));
      const caret = $('.caret', tl);
      if (caret) caret.style.top = `${(k + 0.5) / spec.steps.length * 100}%`;
      cmt.innerHTML = `<h1>${st.title}</h1>` + st.html;
      cmt.scrollTop = 0;
      $('#card').classList.remove('on');
      sel = null;
      if (window.APP.op) APP.op($('#op'), st.op || null);
      render();
      if (fromUser) stop();
    }
    function stop() { playing = false; clearInterval(timer); const b = $('#play'); if (b) b.textContent = '▶'; }
    function play() {
      playing = !playing;
      $('#play').textContent = playing ? '❚❚' : '▶';
      if (!playing) return clearInterval(timer);
      timer = setInterval(() => {
        if (cur >= spec.steps.length - 1) return stop();
        go(cur + 1);
      }, 4200);
    }

    /* -------- build chrome -------- */
    $$('.phase', toc).forEach((p, i) => p.addEventListener('click', () => go(i, true)));
    const total = spec.steps.length;
    tl.innerHTML = '<div class="caret"></div>' + spec.steps.map((s, i) => {
      const top = (i + 0.5) / total * 100;
      return `<div class="evt start" data-i="${i}" style="top:${top}%"></div>`;
    }).join('');
    $$('.evt', tl).forEach((e) => e.addEventListener('click', () => go(+e.dataset.i, true)));
    $('#prev').addEventListener('click', () => go(cur - 1, true));
    $('#next').addEventListener('click', () => go(cur + 1, true));
    $('#play').addEventListener('click', play);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') { go(cur + 1, true); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { go(cur - 1, true); e.preventDefault(); }
      else if (e.key === ' ') { play(); e.preventDefault(); }
    });

    /* -------- camera pad + mouse -------- */
    $$('.move div[data-mv]').forEach((d) => d.addEventListener('click', () => {
      const m = d.dataset.mv;
      if (m === 'l') camera.px -= 60;
      if (m === 'r') camera.px += 60;
      if (m === 'u') camera.py -= 60;
      if (m === 'd') camera.py += 60;
      if (m === 'in') camera.zoom = Math.min(2.6, camera.zoom * 1.15);
      if (m === 'out') camera.zoom = Math.max(0.55, camera.zoom / 1.15);
      if (m === 'reset') { camera.zoom = spec.steps[cur].zoom || 1; camera.px = 0; camera.py = 0; }
      render();
    }));
    let drag = null;
    svg.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, y: e.clientY, px: camera.px, py: camera.py };
      svg.classList.add('drag');
    });
    window.addEventListener('pointermove', (e) => {
      if (!drag) return;
      camera.px = drag.px + (e.clientX - drag.x);
      camera.py = drag.py + (e.clientY - drag.y);
      render();
    });
    window.addEventListener('pointerup', () => { drag = null; svg.classList.remove('drag'); });
    svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      camera.zoom = Math.max(0.55, Math.min(2.6, camera.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
      render();
    }, { passive: false });
    window.addEventListener('resize', render);

    go(0);
  }


  /* ---------- upstream band: instruments → files → ingest ----------
     Shared by both pages: shows where the hourly tables themselves come from. */
  function upstream(U) {
    if (!U) return { stages: [], steps: [] };
    const BLUE = '#7fa8d6', GREY = '#c8d2e6', PURPLE = '#9d8ef0';
    const put = (arr, xs, o) => xs.map((x, i) => ({
      x, y: o.y || 0, z: o.z, w: o.w, h: o.h || 16, d: o.d || 16, color: o.color,
      id: `${o.id}:${arr[i].id}`, label: arr[i].name, big: !!o.big,
      value: o.big ? (arr[i].value || '') : null,
      detail: o.detail ? o.detail(arr[i]) : null, tip: arr[i].name
    }));
    const tags = (arr, xs, z, y) => arr.map((a, i) => ({
      x: xs[i], y: y, z: z, text: a.what || a.note || '', cls: 'val'
    }));

    const ix = [-84, -28, 28, 84];
    const s1 = { id: 'sensor', name: '관측 · 모델', z: -240, focus: ['sensor'],
      labels: tags(U.instruments, ix, -240, -20),
      boxes: put(U.instruments, ix, { z: -240, w: 50, color: BLUE, id: 'sensor', big: true,
        detail: (a) => ({ rows: [['무엇', a.what], ['주기', a.cadence], ['규모', a.value]] }) }) };

    const fx = [-84, -28, 28, 84];
    const s2 = { id: 'files', name: '원본 파일', z: -160, focus: ['files'],
      labels: tags(U.files, fx, -160, -20),
      boxes: put(U.files, fx, { z: -160, w: 50, color: GREY, id: 'file', big: true,
        detail: (a) => ({ rows: [['경로', a.path], ['규모', a.note], ['파싱', a.value]] }) }) };

    const gx = [-70, 0, 70];
    const s3 = { id: 'ingest', name: '수집 · 1시간 격자', z: -80, focus: ['ingest'],
      labels: tags(U.ingest, gx, -80, -20),
      boxes: put(U.ingest, gx, { z: -80, w: 56, color: PURPLE, id: 'ing', big: true,
        detail: (a) => ({ rows: [['무엇', a.name], ['경로', a.note], ['결과', a.value]] }) }) };

    const S = U.suvi, G = U.gong, E = U.enlil;
    const steps = [
      { stage: 'sensor', zoom: 0.95, title: '관측 · 모델', html:
        `<p>모든 값은 네 곳에서 시작합니다. 위성의 분당 측정, 태양면 영상, 자기장 지도,
         그리고 수치모델 실행.</p>
         <table><tbody>` +
        U.instruments.map((a) => `<tr><td>${a.name}</td><td>${a.what}</td></tr>`).join('') +
        `</tbody></table>` },
      { stage: 'files', zoom: 0.95, title: '원본 파일', html:
        `<p>기관이 내놓는 형식 그대로 받습니다. CDF·netCDF·FITS·JSON — 주기와 스키마가 제각각입니다.</p>
         <table><tbody>` +
        U.files.map((a) => `<tr><td>${a.name}</td><td>${a.note}</td></tr>`).join('') +
        `</tbody></table>
         <div class="note">SUVI netCDF ${S.files.toLocaleString()}편 · ${S.gb} GB,
         GONG 지도 ${G.maps.toLocaleString()}장, ENLIL ${E.runs.toLocaleString()} 실행</div>` },
      { stage: 'ingest', zoom: 0.95, title: '수집 · 1시간 격자', html:
        `<p>collector가 내려받아 해시와 함께 RawDB에 남기고, 시간당 파케이로 정리합니다.
         결측은 ffill.</p>
         <table><thead><tr><th>계열</th><th>행</th></tr></thead><tbody>
         <tr><td>ACE 스냅숏</td><td>36,056 × 35</td></tr>
         <tr><td>CH 시간별</td><td>36,015 × 180</td></tr>
         <tr><td>SUVI 패치</td><td>36,015 × 148</td></tr>
         <tr><td>ENLIL 실행</td><td>${E.runs.toLocaleString()}건</td></tr>
         </tbody></table>
         <div class="note">SUVI 슬롯 ${S.slots.toLocaleString()}개 중 ok ${S.ok.toLocaleString()},
         missing ${S.missing.toLocaleString()}</div>` }
    ];
    return { stages: [s1, s2, s3], steps };
  }


  /* ---------------------------------------------------------------------------
     Computation panel: shows the actual arrays and the arithmetic of one step.
     types: 'rowop'  inputs → (weighted|mean|sum) → output, cell by cell
            'reduce' one array → a single number (Σ, mean)
            'curve'  31-point scan with the guard band and the chosen point
  --------------------------------------------------------------------------- */
  function op(host, spec) {
    if (!host) return;
    if (!spec) { host.classList.remove('on'); host.innerHTML = ''; return; }
    host.classList.add('on');
    const cell = (v, cls, i, fmt) =>
      `<div class="c ${cls}" data-i="${i}" title="${typeof v === 'number' ? nf(v, fmt == null ? 2 : fmt) : v}">${
        typeof v === 'number' ? nf(v, fmt == null ? 2 : fmt) : v}</div>`;
    const ell = '<span class="ell">…</span>';
    let h = `<div class="hd"><b>${spec.title}</b><span>${spec.sub || ''}</span></div>`;

    if (spec.type === 'rowop') {
      const rows = spec.rows;           // [{name, values:[…], color}]
      const out = spec.out;             // {name, values:[…]}
      const n = out.values.length;
      h += '<div class="rows">';
      rows.forEach((r) => {
        h += `<div class="r"><span class="nm">${r.name}</span><div class="cells">` +
          r.values.map((v, i) => cell(v, 'src', i, r.fmt)).join('') + '</div></div>';
      });
      h += `<div class="arrowrow"><span class="nm"></span><span>${spec.opLabel || '↓'}</span></div>`;
      h += `<div class="r"><span class="nm">${out.name}</span><div class="cells">` +
        out.values.map((v, i) => cell(v, 'out', i, out.fmt)).join('') + '</div></div>';
      h += '</div>';
      h += `<div class="fx" id="op-fx">${spec.hint || '셀에 마우스를 올리면 계산이 나옵니다'}</div>`;
      if (spec.note) h += `<div class="lg">${spec.note}</div>`;
    }

    if (spec.type === 'reduce') {
      const r0 = spec.row;              // {name, values, fmt}
      h += `<div class="r"><span class="nm">${r0.name}</span><div class="cells">` +
        r0.values.map((v, i) => cell(v, 'src', i, r0.fmt)).join('') +
        (spec.truncate ? ell : '') + '</div></div>' +
        `<div class="arrowrow"><span class="nm"></span><span>${spec.opLabel}</span></div>` +
        `<div class="r"><span class="nm">${spec.outName}</span><div class="cells">` +
        cell(spec.outValue, 'out', -1, spec.outFmt == null ? 2 : spec.outFmt) + '</div></div>' +
        `<div class="fx">${spec.formula}</div>`;
    }

    if (spec.type === 'curve') {
      const pts = spec.points;          // [{w, y, feasible}]
      const W = 560, H = 96, PAD = 18;
      const ys = pts.map((p) => p.y), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const X = (w) => PAD + (w - pts[0].w) / (pts[pts.length - 1].w - pts[0].w) * (W - 2 * PAD);
      const Y = (v) => H - PAD - (v - y0) / Math.max(y1 - y0, 1e-9) * (H - 2 * PAD);
      let g = `<svg class="curve" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
      g += `<line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="#ccc"/>`;
      pts.forEach((p) => {
        g += `<circle cx="${X(p.w).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="2.6"
          fill="${p.feasible ? '#7fb389' : '#d9a3a3'}" data-w="${p.w}" data-y="${p.y}"/>`;
      });
      const ch = pts.find((p) => Math.abs(p.w - spec.chosen) < 1e-9);
      if (ch) {
        g += `<line x1="${X(ch.w)}" y1="${PAD - 6}" x2="${X(ch.w)}" y2="${H - PAD}" stroke="#0c6fdf"
          stroke-dasharray="3 3"/><circle cx="${X(ch.w)}" cy="${Y(ch.y)}" r="4" fill="none"
          stroke="#0c6fdf" stroke-width="1.6"/>`;
      }
      g += '</svg>';
      h += g + `<div class="fx" id="op-fx">${spec.formula}</div>` +
        `<div class="lg"><span><i style="background:#7fb389"></i>가드 통과</span>
         <span><i style="background:#d9a3a3"></i>제외</span>
         <span><i style="background:#0c6fdf"></i>w* = ${nf(spec.chosen, 3)}</span></div>`;
    }

    host.innerHTML = h;

    /* hover arithmetic */
    const fx = host.querySelector('#op-fx');
    if (spec.type === 'rowop') {
      const cs = [...host.querySelectorAll('.c.out')];
      cs.forEach((c) => {
        c.addEventListener('mouseenter', () => {
          const i = +c.dataset.i;
          host.querySelectorAll('.c.src').forEach((x) => x.classList.toggle('pick', +x.dataset.i === i));
          if (fx) fx.innerHTML = spec.formula(i);
        });
        c.addEventListener('mouseleave', () => {
          host.querySelectorAll('.c.pick').forEach((x) => x.classList.remove('pick'));
          if (fx) fx.textContent = spec.hint || '';
        });
      });
    }
    if (spec.type === 'curve') {
      const cs = [...host.querySelectorAll('circle[data-w]')];
      const tf = (x) => ((x) / 560 * 100);
      cs.forEach((c) => {
        c.addEventListener('mouseenter', () => {
          const w = +c.dataset.w, y = +c.dataset.y;
          const p = spec.points.find((q) => Math.abs(q.w - w) < 1e-9);
          if (fx) fx.innerHTML = spec.hover(w, y, p);
        });
        c.addEventListener('mouseleave', () => { if (fx) fx.innerHTML = spec.formula; });
      });
    }
  }

  window.APP = { run, row, grid, box, nf, shade, upstream, op };
})();
