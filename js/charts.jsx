/* global React */

// ============================================================
// Score distribution curve (Monte Carlo first innings)
// p10 / p50 / p90 are the only data points; we render a smooth
// asymmetric distribution and shade between the percentiles.
// ============================================================
const ScoreCurve = ({ scenarios, width=520, height=180, lines=[] }) => {
  // scenarios: { teamName: { p10, p50, p90 } }
  const teams = Object.keys(scenarios);
  if (!teams.length) return null;
  const allVals = teams.flatMap(t => [scenarios[t].p10, scenarios[t].p50, scenarios[t].p90]);
  const lo = Math.floor((Math.min(...allVals) - 25) / 10) * 10;
  const hi = Math.ceil((Math.max(...allVals) + 25) / 10) * 10;
  const W = width, H = height, P = 32;
  const x = v => P + (v - lo) / (hi - lo) * (W - P*2);
  const y = v => H - 28 - v * (H - 60);

  // build curve as gaussian-ish around p50, tail-shaped via p10/p90
  const curve = (s) => {
    const pts = [];
    const N = 80;
    for (let i = 0; i <= N; i++) {
      const v = lo + (hi - lo) * (i / N);
      const sigmaL = (s.p50 - s.p10) / 1.28;
      const sigmaR = (s.p90 - s.p50) / 1.28;
      const sigma = v < s.p50 ? sigmaL : sigmaR;
      const z = (v - s.p50) / sigma;
      const pd = Math.exp(-0.5 * z * z);
      pts.push([x(v), y(pd)]);
    }
    return pts;
  };

  const pathOf = (pts) => 'M ' + pts.map(p => p.join(',')).join(' L ');
  const fillOf = (pts) => pathOf(pts) + ` L ${x(hi)},${y(0)} L ${x(lo)},${y(0)} Z`;

  const colors = teams.map(t => teamHue(t));
  const ticks = [];
  for (let v = Math.ceil(lo/20)*20; v <= hi; v += 20) ticks.push(v);

  // label band per team: team 0 labels go above (top of plot), team 1 labels go below (bottom)
  const labelY = (i, k) => {
    if (i === 0) {
      // above
      return k === 'p50' ? 18 : 30;
    }
    // below — just above x-axis ticks
    return H - 36;
  };
  return React.createElement('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: 'auto' },
    // grid
    ticks.map(t => React.createElement('line', {
      key: 't'+t, x1: x(t), x2: x(t), y1: 14, y2: H-28,
      stroke: 'var(--grid)', strokeDasharray: '2 4',
    })),
    // dist curves
    teams.map((tm, i) => {
      const pts = curve(scenarios[tm]);
      const col = colors[i];
      return React.createElement('g', { key: tm },
        React.createElement('path', { d: fillOf(pts), fill: col, opacity: 0.10 }),
        React.createElement('path', { d: pathOf(pts), fill: 'none', stroke: col, strokeWidth: 1.5 }),
        // p10/p50/p90 markers
        [['p10', scenarios[tm].p10],['p50', scenarios[tm].p50],['p90', scenarios[tm].p90]].map(([k,v]) =>
          React.createElement('g', { key: tm+k },
            React.createElement('line', {
              x1: x(v), x2: x(v),
              y1: y(0), y2: y(k==='p50' ? 1.05 : 0.55),
              stroke: col, strokeWidth: k==='p50' ? 1.5 : 1,
              strokeDasharray: k==='p50' ? '0' : '3 3',
            }),
            React.createElement('text', {
              x: x(v), y: labelY(i, k), fill: col,
              fontFamily: 'var(--mono)', fontSize: 9, textAnchor: 'middle',
              letterSpacing: '0.08em', fontWeight: k==='p50' ? 600 : 400,
            }, k.toUpperCase() + ' ' + Math.round(v)),
          )
        ),
      );
    }),
    // O/U lines
    lines.map((ln, i) => React.createElement('g', { key: 'l'+i },
      React.createElement('line', {
        x1: x(ln.line), x2: x(ln.line),
        y1: 14, y2: H-28,
        stroke: 'var(--accent)', strokeWidth: 1, strokeDasharray: '1 3', opacity: 0.6,
      }),
      React.createElement('text', {
        x: x(ln.line), y: 12, fill: 'var(--accent)',
        fontFamily: 'var(--mono)', fontSize: 9, textAnchor: 'middle',
      }, ln.line),
    )),
    // x-axis ticks
    ticks.map(t => React.createElement('text', {
      key: 'x'+t, x: x(t), y: H-12,
      fill: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 9, textAnchor: 'middle',
    }, t)),
    React.createElement('line', { x1: P, x2: W-P, y1: H-28, y2: H-28, stroke: 'var(--line)' }),
  );
};

// ============================================================
// Diverging horizontal bar — Model vs Bookmaker probability
// for each base learner & ensemble. Centre = 50%.
// ============================================================
const DivergingBar = ({ rows, width=420, rowH=22 }) => {
  // rows: [{label, value, accent?}]
  const W = width, H = rows.length * rowH + 20;
  const cx = W * 0.55;
  const max = 0.55; // half-width = 100% prob band
  const bw = W * 0.4;
  return React.createElement('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: 'auto' },
    // centre line
    React.createElement('line', { x1: cx, x2: cx, y1: 0, y2: H-10, stroke: 'var(--ink-4)' }),
    React.createElement('text', { x: cx, y: H-2, fill: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 9, textAnchor: 'middle' }, '50%'),
    rows.map((r, i) => {
      const y = i * rowH + 10;
      const v = r.value - 0.5;
      const w = (Math.abs(v) / max) * bw;
      const x = v >= 0 ? cx : cx - w;
      const col = r.accent || (v >= 0 ? 'var(--accent-2)' : 'var(--book)');
      return React.createElement('g', { key: r.label },
        React.createElement('text', {
          x: 4, y: y + rowH/2 + 3,
          fill: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 10,
          letterSpacing: '0.06em',
        }, r.label),
        React.createElement('rect', {
          x, y: y+4, width: w, height: rowH-10, fill: col, opacity: r.dim ? 0.4 : 0.85,
        }),
        React.createElement('text', {
          x: v >= 0 ? cx + w + 4 : cx - w - 4,
          y: y + rowH/2 + 3,
          fill: 'var(--ink-2)', fontFamily: 'var(--mono)', fontSize: 10,
          textAnchor: v >= 0 ? 'start' : 'end',
        }, (r.value*100).toFixed(1) + '%'),
      );
    }),
  );
};

// ============================================================
// Over/Under ladder
// ============================================================
const OULadder = ({ lines }) => {
  return React.createElement('div', { className: 'stack', style: { gap: 0 } },
    lines.map((ln, i) =>
      React.createElement('div', {
        key: ln.line,
        style: {
          display: 'grid',
          gridTemplateColumns: '54px 1fr 1fr 54px',
          alignItems: 'center',
          padding: '6px 0',
          borderBottom: i < lines.length - 1 ? '1px dashed var(--line)' : '0',
          fontFamily: 'var(--mono)', fontSize: 11.5,
        }
      },
        React.createElement('div', { style: { color: 'var(--ink-3)', letterSpacing: '0.05em' } },
          React.createElement('span', null, ln.line)),
        // OVER bar
        React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 6 } },
          React.createElement('span', { style: { color: 'var(--green)', width: 38 } }, (ln.p_over*100).toFixed(0)+'%'),
          React.createElement('div', { style: { flex: 1, height: 6, background: 'var(--bg-3)', border: '1px solid var(--line)', position:'relative' } },
            React.createElement('span', { style: { position: 'absolute', inset: 0, width: (ln.p_over*100)+'%', background: 'var(--green)', opacity: 0.65 } }),
          ),
        ),
        // UNDER bar
        React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 6 } },
          React.createElement('div', { style: { flex: 1, height: 6, background: 'var(--bg-3)', border: '1px solid var(--line)', position:'relative' } },
            React.createElement('span', { style: { position: 'absolute', right: 0, top: 0, bottom: 0, width: (ln.p_under*100)+'%', background: 'var(--red)', opacity: 0.55 } }),
          ),
          React.createElement('span', { style: { color: 'var(--red)', width: 38, textAlign: 'right' } }, (ln.p_under*100).toFixed(0)+'%'),
        ),
        React.createElement('div', { style: { textAlign: 'right', color: 'var(--ink-3)' } }, ln.line),
      )
    )
  );
};

// ============================================================
// Top-scorer ladder with horizontal probability bars
// ============================================================
const ScorerLadder = ({ players, accent }) => {
  const max = Math.max(...players.map(p => p.prob));
  return React.createElement('div', { className: 'stack', style: { gap: 4 } },
    players.map((p, i) => React.createElement('div', {
      key: p.player,
      style: { display: 'grid', gridTemplateColumns: '20px 1fr 56px 50px', alignItems: 'center', gap: 8, padding: '4px 0' }
    },
      React.createElement('span', { className: 'mono', style: { color: 'var(--ink-4)', fontSize: 10 } }, String(i+1).padStart(2,'0')),
      React.createElement('span', { style: { fontSize: 12 } }, p.player),
      React.createElement('div', { style: { height: 6, background: 'var(--bg-3)', border: '1px solid var(--line)', position: 'relative' } },
        React.createElement('span', { style: { position:'absolute', inset:0, width: (p.prob/max*100)+'%', background: accent, opacity: 0.85 } }),
      ),
      React.createElement('span', { className: 'mono', style: { textAlign: 'right', fontSize: 11, color: 'var(--ink-2)' } }, (p.prob*100).toFixed(1)+'%'),
    ))
  );
};

// ============================================================
// Bookmaker dot plot — every book's implied home prob, plus consensus
// ============================================================
const BookmakerDots = ({ byBook, consensus, modelP, homeName, awayName, width=520, height=180 }) => {
  const books = Object.entries(byBook);
  if (!books.length) return null;
  const W = width, H = height, P = 28;
  const all = books.map(([_, b]) => b.p_home);
  const lo = Math.max(0, Math.min(modelP, ...all) - 0.04);
  const hi = Math.min(1, Math.max(modelP, ...all) + 0.04);
  const x = v => P + (v - lo) / (hi - lo) * (W - P*2);

  // group books by p_home for stacking
  const bins = {};
  books.forEach(([k, b]) => {
    const key = (b.p_home*1000|0)/1000;
    bins[key] = bins[key] || [];
    bins[key].push(k);
  });

  return React.createElement('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: 'auto' },
    // axis
    React.createElement('line', { x1: P, x2: W-P, y1: H/2, y2: H/2, stroke: 'var(--line)' }),
    // ticks
    [0.3, 0.4, 0.5, 0.6, 0.7].filter(t => t >= lo && t <= hi).map(t =>
      React.createElement('g', { key: t },
        React.createElement('line', { x1: x(t), x2: x(t), y1: H/2-4, y2: H/2+4, stroke: 'var(--ink-4)' }),
        React.createElement('text', { x: x(t), y: H/2+18, fill: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 9, textAnchor: 'middle' },
          (t*100).toFixed(0)+'%'),
      )
    ),
    // dots
    Object.entries(bins).flatMap(([k, list]) =>
      list.map((bookName, idx) => React.createElement('circle', {
        key: bookName,
        cx: x(parseFloat(k)),
        cy: H/2 - 12 - idx*7,
        r: 3, fill: 'var(--book)', opacity: 0.9,
      }))
    ),
    // consensus marker
    consensus && React.createElement('g', null,
      React.createElement('line', {
        x1: x(consensus.p_home), x2: x(consensus.p_home),
        y1: H/2-60, y2: H/2+30,
        stroke: 'var(--book)', strokeWidth: 1, strokeDasharray: '3 3',
      }),
      React.createElement('text', {
        x: x(consensus.p_home), y: 14,
        fill: 'var(--book)', fontFamily: 'var(--mono)', fontSize: 10,
        textAnchor: 'middle', letterSpacing: '0.1em',
      }, 'BOOK CONSENSUS ' + (consensus.p_home*100).toFixed(1)+'%'),
    ),
    // model marker
    React.createElement('g', null,
      React.createElement('line', {
        x1: x(modelP), x2: x(modelP),
        y1: H/2-60, y2: H/2+50,
        stroke: 'var(--accent-2)', strokeWidth: 1.5,
      }),
      React.createElement('text', {
        x: x(modelP), y: H-6,
        fill: 'var(--accent-2)', fontFamily: 'var(--mono)', fontSize: 10,
        textAnchor: 'middle', letterSpacing: '0.1em',
      }, 'MODEL ' + (modelP*100).toFixed(1)+'%'),
    ),
    // labels at extremes
    React.createElement('text', {
      x: P, y: H-6, fill: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '0.1em',
    }, teamCode(awayName) + ' ←'),
    React.createElement('text', {
      x: W-P, y: H-6, fill: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '0.1em', textAnchor: 'end',
    }, '→ ' + teamCode(homeName)),
  );
};

// ============================================================
// Live win-prob arc — based on live target & sim distribution
// ============================================================
const LiveProb = ({ live }) => {
  if (!live || !live.live_prediction) return null;
  const lp = live.live_prediction;
  const target = live.target;
  // If sim's p50 < target, batting team likely loses; chance ≈ P(final>=target)
  // Approx via gaussian using p10/p90 → sigma, p50 → mean.
  const mean = lp.p50;
  const sigma = (lp.p90 - lp.p10) / 2.56;
  // P(final >= target-1)  (need target runs)
  const z = (target - 0.5 - mean) / sigma;
  const pWin = 1 - 0.5 * (1 + erf(z / Math.sqrt(2)));

  function erf(x) {
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y = 1 - ((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592*t;
    const r = 1 - y * Math.exp(-x*x);
    return x >= 0 ? r : -r;
  }

  const W = 420, H = 100;
  const cx = W/2, cy = H + 10, R = 110;
  const startA = Math.PI, endA = 0;
  const arc = (p) => {
    const a = startA - (startA - endA) * p;
    const x1 = cx + R * Math.cos(startA);
    const y1 = cy + R * Math.sin(startA);
    const x2 = cx + R * Math.cos(a);
    const y2 = cy + R * Math.sin(a);
    return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}`;
  };

  return React.createElement('svg', { viewBox: `0 0 ${W} ${H+10}`, width: '100%', height: 'auto' },
    React.createElement('path', { d: arc(1), fill: 'none', stroke: 'var(--line)', strokeWidth: 14 }),
    React.createElement('path', { d: arc(pWin), fill: 'none', stroke: 'var(--accent)', strokeWidth: 14, strokeLinecap: 'butt' }),
    React.createElement('text', {
      x: cx, y: H-2, fill: 'var(--ink)',
      fontFamily: 'var(--mono)', fontSize: 30, textAnchor: 'middle', fontWeight: 600,
      letterSpacing: '-0.02em',
    }, (pWin*100).toFixed(0)+'%'),
    React.createElement('text', {
      x: cx, y: H-32, fill: 'var(--ink-3)',
      fontFamily: 'var(--mono)', fontSize: 9, textAnchor: 'middle', letterSpacing: '0.16em',
    }, teamCode(live.batting_team) + ' WIN PROB'),
  );
};

// ============================================================
// Sparkline (used for line-movement)
// ============================================================
const Spark = ({ values, width=120, height=28, accent='var(--accent-2)' }) => {
  if (!values || !values.length) {
    return React.createElement('div', { className: 'small', style: { color: 'var(--ink-4)' } }, 'flat — single snap');
  }
  const lo = Math.min(...values), hi = Math.max(...values);
  const r = hi - lo || 1;
  const W = width, H = height;
  const pts = values.map((v, i) => [i / (values.length - 1) * W, H - ((v - lo) / r) * (H-4) - 2]);
  const path = 'M ' + pts.map(p => p.join(',')).join(' L ');
  return React.createElement('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H },
    React.createElement('path', { d: path, fill: 'none', stroke: accent, strokeWidth: 1.5 }),
  );
};

window.ScoreCurve = ScoreCurve;
window.DivergingBar = DivergingBar;
window.OULadder = OULadder;
window.ScorerLadder = ScorerLadder;
window.BookmakerDots = BookmakerDots;
window.LiveProb = LiveProb;
window.Spark = Spark;
