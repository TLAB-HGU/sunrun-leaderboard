/* ---------------------------------------------------------------------------
   Shared helpers for the two inference walkthroughs.
   DATA is injected by code/build.py as a global object.
--------------------------------------------------------------------------- */
(function () {
  'use strict';
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const SVG = 'http://www.w3.org/2000/svg';

  function get(path, obj) {
    return path.split('.').reduce((a, k) => (a == null ? a : a[k]), obj == null ? DATA : obj);
  }
  function nf(v, d) {
    if (v == null || v === '' || Number.isNaN(v)) return '—';
    const n = Number(v);
    if (!isFinite(n)) return '—';
    return n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })
      .replace('-', '\u2212');
  }
  const FMT = {
    '0': (v) => nf(v, 0), '1': (v) => nf(v, 1), '2': (v) => nf(v, 2), '3': (v) => nf(v, 3),
    '4': (v) => nf(v, 4),
    pct1: (v) => (v == null ? '—' : nf(v * 100, 1) + '%'),
    pct2: (v) => (v == null ? '—' : nf(v * 100, 2) + '%'),
    signed: (v) => (v == null ? '—' : (v > 0 ? '+' : '') + nf(v, 1)),
    signed2: (v) => (v == null ? '—' : (v > 0 ? '+' : '') + nf(v, 2)),
    k: (v) => (v == null ? '—' : nf(v / 1000, 1) + 'k'),
    raw: (v) => (v == null ? '—' : String(v)),
    us: (v) => (v == null ? '—' : nf(v, 1) + ' µs'),
    w: (v) => (v == null ? '—' : nf(v, 3))
  };
  function fmt(v, f) { return (FMT[f] || FMT.raw)(v); }
  function ts(s) {
    if (!s) return '—';
    const d = new Date(s);
    const p = (n) => String(n).padStart(2, '0');
    return d.getUTCFullYear() + '-' + p(d.getUTCMonth() + 1) + '-' + p(d.getUTCDate()) +
      ' ' + p(d.getUTCHours()) + ':' + p(d.getUTCMinutes());
  }
  function tsShort(s) {
    if (!s) return '—';
    const d = new Date(s);
    const p = (n) => String(n).padStart(2, '0');
    return p(d.getUTCMonth() + 1) + '/' + p(d.getUTCDate()) + ' ' + p(d.getUTCHours()) + 'h';
  }

  /* ---------- data binding: any element with data-num gets a formatted number ---------- */
  function bind(root) {
    $$('[data-num]', root).forEach((el) => {
      const v = get(el.dataset.num);
      const f = el.dataset.fmt || 'raw';
      let s = fmt(v, f);
      if (el.dataset.pre) s = el.dataset.pre + s;
      if (el.dataset.post) s = s + el.dataset.post;
      el.textContent = s;
    });
    $$('[data-ts]', root).forEach((el) => { el.textContent = ts(get(el.dataset.ts)); });
    $$('[data-tss]', root).forEach((el) => { el.textContent = tsShort(get(el.dataset.tss)); });
    $$('[data-text]', root).forEach((el) => { el.textContent = String(get(el.dataset.text)); });
  }

  /* ---------- tooltip ---------- */
  const tip = document.createElement('div');
  tip.className = 'tip';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(tip));
  function showTip(ev, html) {
    tip.innerHTML = html;
    tip.classList.add('on');
    const r = tip.getBoundingClientRect();
    let x = ev.clientX + 12, y = ev.clientY - r.height - 10;
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 12;
    if (y < 4) y = ev.clientY + 16;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function hideTip() { tip.classList.remove('on'); }

  /* ---------- generic cartesian chart ---------- */
  function chart(host, spec) {
    const W = spec.w || 900, H = spec.h || 300;
    const pad = Object.assign({ t: 18, r: 18, b: 38, l: 56 }, spec.pad || {});
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%');
    svg.style.display = 'block';
    svg.style.height = 'auto';
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const xs = (v) => pad.l + ((v - spec.x.min) / (spec.x.max - spec.x.min)) * iw;
    const ys = (v) => pad.t + ih - ((v - spec.y.min) / (spec.y.max - spec.y.min)) * ih;
    const add = (tag, attrs, parent) => {
      const e = document.createElementNS(SVG, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(e);
      return e;
    };

    (spec.bands || []).forEach((b) => {
      add('rect', { x: xs(b.x0), y: pad.t, width: Math.max(0, xs(b.x1) - xs(b.x0)), height: ih,
        fill: b.fill || '#f7f7fb', stroke: 'none' });
      if (b.label) {
        // bottom-left of the band: keeps clear of top-edge markers (w=0, first w>0, thresholds)
        const t = add('text', { x: xs(b.x0) + 6, y: pad.t + ih - 7, class: 'lbl',
          fill: b.color || '#76768a' });
        t.textContent = b.label;
      }
    });

    (spec.y.ticks || []).forEach((v) => {
      add('line', { x1: pad.l, x2: W - pad.r, y1: ys(v), y2: ys(v), class: 'ax',
        'stroke-dasharray': v === spec.y.min ? '' : '3 3' });
      const t = add('text', { x: pad.l - 8, y: ys(v) + 3.5, class: 'tick', 'text-anchor': 'end' });
      t.textContent = (spec.y.fmt || ((q) => nf(q, 0)))(v);
    });
    (spec.x.ticks || []).forEach((v) => {
      add('line', { x1: xs(v), x2: xs(v), y1: pad.t, y2: pad.t + ih, class: 'ax',
        'stroke-dasharray': '3 3' });
      const t = add('text', { x: xs(v), y: H - pad.b + 15, class: 'tick', 'text-anchor': 'middle' });
      t.textContent = (spec.x.fmt || ((q) => nf(q, 0)))(v);
    });
    if (spec.x.label) {
      const t = add('text', { x: pad.l + iw / 2, y: H - 4, class: 'lbl', 'text-anchor': 'middle' });
      t.textContent = spec.x.label;
    }
    if (spec.y.label) {
      const t = add('text', { x: 12, y: pad.t + ih / 2, class: 'lbl', 'text-anchor': 'middle',
        transform: `rotate(-90 12 ${pad.t + ih / 2})` });
      t.textContent = spec.y.label;
    }

    (spec.hlines || []).forEach((h) => {
      add('line', { x1: pad.l, x2: W - pad.r, y1: ys(h.y), y2: ys(h.y), stroke: h.color,
        'stroke-width': h.width || 1.2, 'stroke-dasharray': h.dash || '5 4' });
      if (h.label) {
        const t = add('text', { x: W - pad.r - 2, y: ys(h.y) - 5, class: 'tick',
          'text-anchor': 'end', fill: h.color });
        t.textContent = h.label;
      }
    });

    (spec.series || []).forEach((s) => {
      const pts = s.x.map((xv, i) => [xs(xv), ys(s.y[i])]).filter((p) => isFinite(p[1]));
      if (!pts.length) return;
      if (s.area) {
        const base = ys(s.area.at != null ? s.area.at : spec.y.min);
        let d = `M${pts[0][0]},${base}` + pts.map((p) => `L${p[0]},${p[1]}`).join('') +
          `L${pts[pts.length - 1][0]},${base}Z`;
        add('path', { d, fill: s.area.fill || s.color, opacity: s.area.opacity || 0.12, stroke: 'none' });
      }
      let d = `M${pts[0][0]},${pts[0][1]}`;
      for (let i = 1; i < pts.length; i++) {
        if (s.step) d += `L${pts[i][0]},${pts[i - 1][1]}`;
        d += `L${pts[i][0]},${pts[i][1]}`;
      }
      add('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2,
        'stroke-dasharray': s.dash || '', 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        opacity: s.opacity || 1 });
      if (s.points) {
        pts.forEach((p, i) => {
          const c = add('circle', { cx: p[0], cy: p[1], r: s.r || 2.4, fill: s.color,
            stroke: '#fff', 'stroke-width': 1 });
          if (s.tip) {
            c.style.cursor = 'crosshair';
            c.addEventListener('mousemove', (e) => showTip(e, s.tip(i)));
            c.addEventListener('mouseleave', hideTip);
          }
        });
      }
    });

    (spec.vlines || []).forEach((v) => {
      add('line', { x1: xs(v.x), x2: xs(v.x), y1: pad.t, y2: pad.t + ih, stroke: v.color || '#7b61ff',
        'stroke-width': v.width || 1.4, 'stroke-dasharray': v.dash || '4 4' });
      if (v.label) {
        const t = add('text', { x: xs(v.x) + 4, y: pad.t + (v.dy || 11), class: 'tick',
          fill: v.color || '#7b61ff' });
        t.textContent = v.label;
      }
    });

    (spec.marks || []).forEach((m) => {
      add('circle', { cx: xs(m.x), cy: ys(m.y), r: m.r || 4.5, fill: m.color || '#7b61ff',
        stroke: '#fff', 'stroke-width': 1.6 });
      if (m.label) {
        const t = add('text', { x: xs(m.x) + (m.dx || 8), y: ys(m.y) + (m.dy || -8), class: 'tick',
          fill: m.color || '#7b61ff', 'font-weight': 700 });
        t.textContent = m.label;
      }
    });
    host.appendChild(svg);
    return svg;
  }

  function legend(host, items) {
    const html = items.map((i) =>
      `<span class="sw"><i style="background:${i.color};${i.dash ? 'height:2px;border-radius:0' : ''}"></i>${i.label}</span>`
    ).join('');
    if (host.classList.contains('legend')) { host.insertAdjacentHTML('beforeend', html); return; }
    const d = document.createElement('div');
    d.className = 'legend';
    d.innerHTML = html;
    host.appendChild(d);
  }



  /* ---------------------------------------------------------------------------
     Interactive dataflow scene: one plate per pipeline layer, one step at a time.
     Inspired by the "walk through the computation" style of bbycroft.net/llm:
     real numbers on every block, click a plate (or press ← →) to advance.
  --------------------------------------------------------------------------- */
  function scene(cfg) {
    const host = $(cfg.host || '#scene'), side = $(cfg.panel || '#scene-side');
    if (!host || !side) return;
    const layers = cfg.layers;
    host.innerHTML = layers.map((L, i) =>
      `<div class="plate" data-layer="${L.id}" data-i="${i}">
         <div class="ph"><span class="pi">${String(i + 1).padStart(2, '0')}</span>
           <span class="pn">${L.name}</span><span class="ps">${L.sub || ''}</span></div>
         <div class="pb">${L.html || ''}</div>
       </div>`).join('');
    const plates = [...host.querySelectorAll('.plate')];
    const steps = cfg.steps;
    let cur = -1;

    const dots = steps.map((_, i) => `<i data-d="${i}"></i>`).join('');
    function show(k) {
      k = Math.max(0, Math.min(steps.length - 1, k));
      cur = k;
      const st = steps[k];
      plates.forEach((pl) => pl.classList.toggle('on', pl.dataset.layer === st.layer));
      side.innerHTML =
        `<div class="side-h"><span class="n">STEP ${String(k + 1).padStart(2, '0')}/${String(steps.length).padStart(2, '0')}</span>
           <span>${st.tag || ''}</span></div>` +
        `<div class="side-t">${st.title}</div>` +
        `<div class="side-b">${st.body}</div>` +
        (st.formula ? `<div class="side-f">${st.formula}</div>` : '') +
        (st.note ? `<div class="side-n">${st.note}</div>` : '') +
        `<div class="side-nav">
           <button class="btn" data-nav="prev">◀ 이전</button>
           <button class="btn" data-nav="next">다음 ▶</button>
           <div class="dots">${dots}</div>
         </div>
         <div class="kbdhint">← → 이동 · 판을 클릭하면 그 단계로 이동</div>`;
      $$('.dots i', side).forEach((d) => d.classList.toggle('on', +d.dataset.d === k));
      $$('[data-nav]', side).forEach((b) => b.addEventListener('click', () =>
        show(k + (b.dataset.nav === 'next' ? 1 : -1))));
      $$('.dots i', side).forEach((d) => d.addEventListener('click', () => show(+d.dataset.d)));
      if (st.after) st.after(side);
      if (cfg.onStep) cfg.onStep(k, st);
    }

    plates.forEach((pl) => pl.addEventListener('click', (ev) => {
      if (ev.target.closest('.grid72')) return;   // cell clicks are handled by the step itself
      const k = steps.findIndex((s2) => s2.layer === pl.dataset.layer);
      if (k >= 0) show(k);
    }));
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      const box = host.getBoundingClientRect();
      if (box.bottom < 0 || box.top > window.innerHeight) return;
      if (e.key === 'ArrowRight') { show(cur + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { show(cur - 1); e.preventDefault(); }
    });
    show(cfg.start || 0);
  }

  /* hot/cold colour for a 1-D series (used by the 72-cell strips) */
  function heat(v, lo, hi) {
    const t = Math.max(0, Math.min(1, (v - lo) / Math.max(hi - lo, 1e-9)));
    const c1 = [26, 43, 68], c2 = [90, 150, 230], c3 = [150, 139, 255], c4 = [255, 190, 61];
    const seg = t < 0.5 ? [c1, c2, t / 0.5] : (t < 0.8 ? [c2, c3, (t - 0.5) / 0.3] : [c3, c4, (t - 0.8) / 0.2]);
    const a = seg[0], b = seg[1], u = seg[2];
    return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(',')})`;
  }

  /* ---------------------------------------------------------------------------
     Raw-data frames: sources → alignment/quality → features → one worked mapping.
     Shared by both walkthroughs; the page picks its example by key.
  --------------------------------------------------------------------------- */
  function renderRaw(key) {
    const R = DATA.raw, X = (R.examples || {})[key];
    if (!R) return;
    const k = (n) => nf(n, 0);

    /* ---------------- 1) sources ---------------- */
    const st = $('#src-table');
    if (st) {
      st.innerHTML = '<table class="doc"><thead><tr><th>원자료</th><th>주기</th><th>행 수</th>' +
        '<th style="text-align:left">기간</th><th style="text-align:left">만들어지는 특징</th></tr></thead><tbody>' +
        R.sources.map((s2) => `<tr><td style="text-align:left"><b>${s2.title}</b><br>
          <span class="tiny mono" style="color:var(--ink-3)">${s2.path}</span></td>
          <td style="white-space:normal">${s2.cadence}</td><td class="mono-cell">${k(s2.rows)}</td>
          <td style="text-align:left" class="mono-cell">${ts(s2.first)}<br><span style="color:var(--ink-3)">${ts(s2.last)}</span></td>
          <td style="text-align:left" class="mono-cell txt">${s2.makes}</td></tr>`).join('') +
        '</tbody></table>';
    }
    const kpi = $('#src-kpi');
    if (kpi) {
      kpi.innerHTML = [
        ['원자료 시간 격자', k(R.grid.hourly_rows) + ' h', `${ts(R.grid.first)} → ${ts(R.grid.last)}`],
        ['타깃 결측 채움', nf(R.grid.was_missing_pct, 1) + '%', `${k(R.grid.missing_hours)} h 를 ffill`],
        ['ACE 속도 관측 공백', nf(R.grid.ace_speed_null_pct, 1) + '%', '원자료 시간대 중 속도 없음'],
        ['SUVI 이미지 stale (F2)', nf(R.image_quality.stale_pct, 1) + '%',
          `age 중앙 ${nf(R.image_quality.age_p50_h, 2)} h`]
      ].map(([t, v, d]) => `<div class="kpi"><div class="k">${t}</div>
        <div class="v" style="font-size:19px">${v}</div><div class="d">${d}</div></div>`).join('');
    }

    /* ---------------- 2) strip geometry ---------------- */
    const lamHost = $('#lam-chart');
    if (lamHost) {
      const S = R.strip, hs = Array.from({ length: 72 }, (_, i) => i + 1);
      const colors = ['#a5a5b8', '#0d99ff', '#7b61ff', '#14ae5c', '#f24822'];
      const lams = S.speeds.flatMap((v) => hs.map((hh) => S.omega_deg_h * (S.tau_h[String(v)] - hh)));
      const lo = Math.min(0, ...lams), hi = Math.max(0, ...lams);
      chart(lamHost, {
        h: 300, pad: { l: 58, r: 96, t: 18, b: 42 },
        x: { min: 1, max: 72, ticks: [1, 12, 24, 36, 48, 60, 72], fmt: (v) => 'h' + v,
          label: '도달 시각 h' },
        y: { min: lo - 4, max: hi + 4, ticks: UI.niceTicks(lo - 2, hi + 2, 4), fmt: (v) => nf(v, 0),
          label: '유효 경도 λ (도, 서쪽 +)' },
        series: S.speeds.map((v, i) => ({
          x: hs, y: hs.map((hh) => S.omega_deg_h * (S.tau_h[String(v)] - hh)),
          color: colors[i], width: 2.2,
          tip: (j) => `${v} km/s · h${hs[j]} · λ=${nf(S.omega_deg_h * (S.tau_h[String(v)] - hs[j]), 1)}°`
        })),
        hlines: [{ y: 0, color: '#1e1e28', dash: '4 4', label: '지구 정면 (λ = 0)' }],
        vlines: [{ x: 24, color: '#c9c9d6', dash: '3 3' }]
      });
      legend($('#lam-legend'), S.speeds.map((v, i) =>
        ({ color: colors[i], label: `v = ${v} km/s (τ ${nf(S.tau_h[String(v)], 1)} h)` })));
    }

    /* ---------------- 3) missing by month ---------------- */
    const mm = $('#miss-chart');
    if (mm && R.missing_by_month) {
      const M = R.missing_by_month, xs = M.map((_, i) => i);
      const f1a = M.findIndex((r) => r.month === '2026-02');
      const f2a = M.findIndex((r) => r.month === '2026-05');
      const ticks = xs.filter((i) => M[i].month.endsWith('-01') || i === 0 || i === M.length - 1);
      chart(mm, {
        h: 230, pad: { l: 54, r: 18, t: 16, b: 40 },
        x: { min: 0, max: M.length - 1, ticks, fmt: (v) => M[v] ? M[v].month.slice(2) : '',
          label: '월' },
        y: { min: 0, max: Math.max(...M.map((r) => r.pct)) * 1.12,
          ticks: UI.niceTicks(0, Math.max(...M.map((r) => r.pct)), 4), fmt: (v) => nf(v, 0) + '%',
          label: '결측(채운) 비율' },
        bands: f1a >= 0 ? [{ x0: f1a, x1: f2a, fill: '#f8f7ff', label: 'F1' },
          { x0: f2a, x1: M.length - 1, fill: '#fbfbfd', label: 'F2' }] : [],
        series: [{ x: xs, y: M.map((r) => r.pct), color: '#0d99ff', width: 1.6, step: true,
          area: { at: 0, opacity: 0.1 },
          tip: (j) => `${M[j].month} · ${nf(M[j].pct, 1)}% (${M[j].hours} h)` }]
      });
    }

    const qt = $('#quality-table');
    if (qt) {
      const B = R.bundle, iq = R.image_quality;
      const rows = [
        ['ACE 원자료 속도 공백', `${nf(R.grid.ace_speed_null_pct, 2)}%`, '시간대 기준'],
        ['타깃 시계열 채움', `${nf(R.grid.was_missing_pct, 2)}% (${k(R.grid.missing_hours)} h)`,
          'ffill + was_missing 플래그'],
        ['번들 C (상수 90열)', `${k(B.c_nan)} / ${k(B.c_cells)} 셀 (${nf(B.c_nan / B.c_cells * 100, 2)}%)`,
          'origin 방향 ffill'],
        ['번들 E (가변 72×20)', `${k(B.e_nan)} / ${k(B.e_cells)} 셀 (${nf(B.e_nan / B.e_cells * 100, 2)}%)`,
          'origin 방향 ffill'],
        ['번들 past (24 h)', '0 셀', '속도 이력은 항상 존재'],
        ['이미지 stale (F2)', `${nf(iq.stale_pct, 2)}% · age 최대 ${nf(iq.age_max_h, 1)} h`,
          'status=ok 슬롯만 후보']
      ];
      qt.innerHTML = '<table class="doc"><thead><tr><th style="text-align:left">항목</th>' +
        '<th>값</th><th style="text-align:left">처리</th></tr></thead><tbody>' +
        rows.map(([a, b, c]) => `<tr><td style="text-align:left">${a}</td>
          <td class="mono-cell">${b}</td><td style="text-align:left" class="tiny">${c}</td></tr>`).join('') +
        '</tbody></table>';
    }

    /* ---------------- 3b) bundle box ---------------- */
    const bb = $('#bundle-box');
    if (bb) {
      const B = R.bundle;
      bb.innerHTML =
        `<span class="cm">// analysis/A0014/outputs/cache/*.npz</span><br>` +
        `C     ${k(B.C_shape[0])} × ${B.C_shape[1]}    <span class="cm">// origin 시점 값</span><br>` +
        `E     ${k(B.E_shape[0])} × ${B.E_shape[1]} × ${B.E_shape[2]} <span class="cm">// 72h 궤적</span><br>` +
        `past  ${k(B.past_shape[0])} × ${B.past_shape[1]}   <span class="cm">// 최근 24h 속도</span><br>` +
        `<span class="cm">// origin 격자</span> train ${k(B.train_origins)} · F1 ${k(B.f1_origins)} · F2 ${k(B.official_origins)}<br>` +
        `<span class="cm">// prefill</span> ${B.prefill}`;
    }

    /* ---------------- 4) feature families ---------------- */
    const ft = $('#fam-table');
    if (ft) {
      ft.innerHTML = '<table class="doc"><thead><tr><th style="text-align:left">특징 패밀리</th>' +
        '<th style="text-align:left">원천</th><th>개수</th></tr></thead><tbody>' +
        R.families.map((f) => `<tr><td style="text-align:left" class="mono-cell">${f.prefix}</td>
          <td style="text-align:left" class="tiny">${f.meaning}</td>
          <td class="mono-cell">${f.n}</td></tr>`).join('') +
        `</tbody></table>`;
      const tot = R.families.reduce((a, f) => a + f.n, 0);
      $('#fam-total').textContent = '합계 ' + tot + '개';
    }

    if (!X) return;

    /* ---------------- 5) worked raw → feature ---------------- */
    const wf = $('#map-table');
    if (wf) {
      const ra = X.raw_ace, F2 = X.features, C = X.map_checks || {};
      const pick = (k2) => (C[k2] ? C[k2] : [null, null]);
      const dec = (v) => (Math.abs(v) >= 100 ? 0 : (Math.abs(v) >= 10 ? 1 : (Math.abs(v) >= 1 ? 2 : 3)));
      const ok = (a, b) => (a == null || b == null) ? false
        : Math.abs(a - b) <= Math.max(1e-3, Math.abs(b) * 1e-4);
      const rows = [
        ['ACE 속도 @ origin', pick('speed_last'), '그대로', 'ace_speed_kms_last'],
        [`ACE 속도 6행 평균 (${ra.m6.rows}행 중 ${ra.m6.obs}개 관측)`, pick('speed_m6'), '평균', 'ace_speed_kms_m6'],
        [`ACE 속도 24행 평균 (${ra.m24.rows}행 중 ${ra.m24.obs}개 관측)`, pick('speed_m24'), '평균', 'ace_speed_kms_m24'],
        ['ACE 밀도 @ origin', pick('density_last'), '그대로', 'ace_density_cm3_last'],
        ['ACE 온도 @ origin', pick('temp_last'), '그대로', 'ace_temperature_k_last'],
        ['ACE Bt @ origin', pick('bt_last'), '그대로', 'ace_bt_nt_last'],
        ['ACE By(GSM) 24행 평균', pick('by_m24'), '평균', 'ace_by_gsm_nt_m24'],
        ['ACE EPAM p ch3 @ origin', pick('epam_log'), 'log₁₀', 'ace_epam_p_channel3_last'],
        ['ACE SIS &gt;10 MeV @ origin', pick('sis_log'), 'log₁₀', 'ace_sis_gt10_flux_last'],
        ['ACE 속도 @ 목표 h1 − 27일', pick('rec27_h1'), '27일 회귀', 'rec27 (h=1)'],
        [`SUVI 이미지 슬롯 ${ts(X.img.slot)}`, [null, F2.img.img_age_h], `age ${nf(X.img.age_h, 1)} h`, 'img_age_h'],
        ['코로나홀 지도 @ origin', [null, F2.ch.ch_ch_t055_area], '면적 합', 'ch_ch_t055_area'],
        ['WSA-Enlil 실행 @ origin', [null, F2.enlil.enlil_n_pending], '실행 상태', 'enlil_n_pending']
      ];
      wf.innerHTML = '<table class="doc"><thead><tr><th style="text-align:left">원자료</th><th>원자료 값</th>' +
        '<th style="text-align:left">변환</th><th style="text-align:left">특징</th><th>모델 입력</th>' +
        '<th>검산</th></tr></thead><tbody>' + rows.map(([a, kv, c, d]) => {
          const raw = kv[0], feats = kv[1];
          const mark = raw == null ? '<span style="color:var(--ink-4)">—</span>'
            : (ok(raw, feats) ? '<span style="color:#0d8446">✓</span>'
                              : '<span style="color:#c9320f">✗</span>');
          return `<tr><td style="text-align:left">${a}</td>
            <td class="mono-cell">${raw == null ? '—' : nf(raw, dec(raw))}</td>
            <td style="text-align:left" class="tiny">${c}</td>
            <td style="text-align:left" class="mono-cell txt">${d}</td>
            <td class="mono-cell">${feats == null ? '—' : nf(feats, dec(feats))}</td>
            <td class="mono-cell">${mark}</td></tr>`;
        }).join('') + '</tbody></table>';
    }

    /* ---------------- 6) SUVI strip hand calculation ---------------- */
    const pt = $('#patch-table');
    if (pt && X.suvi_patches.log_median) {
      const S = R.strip, c1 = X.strip_calc[0].v550, c2 = X.strip_calc[1].v550;
      const centers = S.cell_centers;
      const P = (w, lm) => nf(w * lm, 4);
      pt.innerHTML = '<table class="doc"><thead><tr><th>경도 셀</th><th>log_median</th><th>dark</th>' +
        '<th>w (h1)</th><th>w·log</th><th>w (h24)</th><th>w·log</th></tr></thead><tbody>' +
        centers.map((c, i) => {
          const lm = X.suvi_patches.log_median[i], dk = X.suvi_patches.dark_fraction[i];
          return `<tr><td class="mono-cell">${nf(c, 1)}°</td><td class="mono-cell">${nf(lm, 3)}</td>
            <td class="mono-cell">${nf(dk, 3)}</td>
            <td class="mono-cell">${nf(c1.w[i], 3)}</td><td class="mono-cell">${P(c1.w[i], lm)}</td>
            <td class="mono-cell">${nf(c2.w[i], 3)}</td><td class="mono-cell">${P(c2.w[i], lm)}</td></tr>`;
        }).join('') +
        `<tr class="pick"><td>가중 평균</td><td colspan="2" class="tiny">Σw = ${nf(c1.w.reduce((a, b) => a + b, 0), 3)} / ${nf(c2.w.reduce((a, b) => a + b, 0), 3)}</td>
          <td class="mono-cell">→</td><td class="mono-cell">${nf(c1.logmed, 4)}</td>
          <td class="mono-cell">→</td><td class="mono-cell">${nf(c2.logmed, 4)}</td></tr></tbody></table>`;
    }
    const sn = $('#strip-note');
    if (sn) {
      const d = Math.max(...R.strip.speeds.map((v) => X.strip_max_diff['v' + v] || 0));
      sn.innerHTML = '<b>재계산 = 번들.</b> 이미지 패치 12칸에서 위 식으로 만든 값이 번들의 ' +
        '<code>strip_v*_logmed</code>와 <b>다섯 속도 × 네 지점(h 1·24·48·72)에서 모두 일치</b>합니다' +
        ` (최대 차이 ${d.toExponential(1)}). 즉 이 특징은 손으로 따라간 그 식이 그대로 들어간 값입니다.`;
    }
    const so = $('#strip-out');
    if (so) {
      const S = R.strip;
      so.innerHTML = '<table class="doc"><thead><tr><th>h</th>' + S.speeds.map((v) => `<th>λ (v=${v})</th>`).join('') +
        '<th>strip_v550_logmed</th><th>번들 값</th><th>일치</th></tr></thead><tbody>' +
        X.strip_calc.map((c, j) => {
          const hh = c.h, idx = [0, 23, 47, 71][j];
          const calc = c.v550.logmed, bun = X.features.varying.strip_v550_logmed[j];
          const ok = X.strip_match.v550[j];
          return `<tr><td>${hh}</td>` +
            S.speeds.map((v) => `<td class="mono-cell">${nf(c['v' + v].lam, 1)}°</td>`).join('') +
            `<td class="mono-cell">${calc == null ? 'NaN' : nf(calc, 4)}</td>
             <td class="mono-cell">${bun == null ? 'NaN' : nf(bun, 4)}</td>
             <td class="mono-cell" style="color:${ok ? '#0d8446' : '#c9320f'}">${ok ? '✓' : '✗'}</td></tr>`;
        }).join('') + '</tbody></table>';
    }
  }

  /* ---------- flow arrows: turn the chevron down at a row break ---------- */
  function flowArrows(host) {
    const nodes = [...host.children].filter((e) => e.classList.contains('node'));
    nodes.forEach((n, i) => {
      n.classList.remove('rowend');
      if (i === nodes.length - 1) return;
      const a = n.getBoundingClientRect(), b = nodes[i + 1].getBoundingClientRect();
      if (Math.abs(a.top - b.top) > 2) n.classList.add('rowend');
    });
  }
  function allFlows() { document.querySelectorAll('.flow').forEach(flowArrows); }

  /* ---------- zoom ---------- */
  function zoom() {
    const stage = $('.stage');
    const label = $('#zoomLabel');
    if (!stage || !label) return;
    let z = 1;
    const apply = () => {
      stage.style.transform = `scale(${z})`;
      stage.style.width = `${100 / z}%`;
      label.textContent = Math.round(z * 100) + '%';
    };
    const set = (v) => { z = Math.min(1.6, Math.max(0.6, v)); apply(); };
    $('#zoomIn') && $('#zoomIn').addEventListener('click', () => set(z + 0.1));
    $('#zoomOut') && $('#zoomOut').addEventListener('click', () => set(z - 0.1));
    label.addEventListener('click', () => { z = 1; apply(); });
    apply();
  }

  /* ---------- layer rail scroll-spy ---------- */
  function spy() {
    const layers = $$('.layer');
    const secs = layers.map((l) => document.getElementById(l.dataset.target)).filter(Boolean);
    if (!secs.length || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (!e.isIntersecting) return;
        layers.forEach((l) => l.classList.toggle('on', l.dataset.target === e.target.id));
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });
    secs.forEach((s) => io.observe(s));
    layers.forEach((l) => l.addEventListener('click', () => {
      const el = document.getElementById(l.dataset.target);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
  }

  function boot() {
    bind();
    zoom();
    spy();
    if (typeof render === 'function') render();
    allFlows();
    let t = 0;
    window.addEventListener('resize', () => {
      clearTimeout(t);
      t = setTimeout(allFlows, 120);
    });
  }
  document.addEventListener('DOMContentLoaded', boot);

  /* round tick positions: 1/2/2.5/5 x 10^k */
  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / Math.max(1, n || 4);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s2) => s2 >= raw) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
    }
    return out;
  }
  function yFmt(lo, hi) {
    const span = Math.abs(hi - lo);
    if (span <= 1.5) return (v) => nf(v, 2);
    if (span <= 12) return (v) => nf(v, 1);
    return (v) => nf(v, 0);
  }

  window.UI = { $, $$, get, fmt, ts, tsShort, chart, legend, bind, showTip, hideTip, nf,
                niceTicks, yFmt, flowArrows, allFlows, renderRaw, scene, heat };
})();
