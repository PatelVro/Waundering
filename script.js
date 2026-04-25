/* Cirelay Cricket dashboard — load data.json and populate the page. */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const fmtPct = (x, d = 1) =>
  (x == null || Number.isNaN(x)) ? "—" : `${(x * 100).toFixed(d)}%`;
const fmtNum = (x, d = 0) =>
  (x == null || Number.isNaN(x)) ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d });
const fmt2 = x => (x == null || Number.isNaN(x)) ? "—" : Number(x).toFixed(2);

let _liveRefreshTimer = null;

async function loadData() {
  const r = await fetch("data.json", { cache: "no-cache" });
  return r.json();
}

(async function main() {
  let data;
  try {
    data = await loadData();
  } catch (e) {
    console.error("Failed to load data.json", e);
    return;
  }

  hydrateHero(data);
  hydratePrediction(data.latest_prediction);
  hydrateMetrics(data.model_metrics);
  hydrateTeams("elo-t20", data.top_teams_t20);
  hydrateTeams("elo-odi", data.top_teams_odi);
  hydrateRecent(data.recent_matches);
  hydrateDataStats(data.data_stats);
  hydrateFooter(data.generated_at);
  hydrateLive(data.live_match);
  initFilters();
  initReveal();
  startLiveRefreshIfNeeded(data.live_match);
})();

function startLiveRefreshIfNeeded(live) {
  const isLive = live && !live.is_complete;
  if (!isLive || _liveRefreshTimer) return;

  _liveRefreshTimer = setInterval(async () => {
    try {
      const data = await loadData();
      hydrateLive(data.live_match);
      if (!data.live_match || data.live_match.is_complete) {
        clearInterval(_liveRefreshTimer);
        _liveRefreshTimer = null;
      }
    } catch (_) {}
  }, 30_000);
}

function hydrateHero(d) {
  const m   = d.model_metrics?.t20 || {};
  const all = d.model_metrics?.all || {};
  $("#hs-t20").textContent = fmtPct(m.acc, 1);
  $("#hs-ece").textContent = fmtPct(all.ece, 2);
  // hi-conf accuracy headline (from cycle 4 final eval all-formats hi-conf)
  $("#hs-hi").textContent = "84%";

  $("#stat-matches").textContent = fmtNum(d.data_stats?.matches);
  $("#stat-balls").textContent   = fmtNum(d.data_stats?.balls);
}

function hydratePrediction(p) {
  if (!p) {
    $("#pred-vs").textContent = "No prediction available — run predict_match.py first.";
    return;
  }
  const home = p.match.home;
  const away = p.match.away;
  const ph   = p.prediction.p_home_wins;
  const pa   = p.prediction.p_away_wins;

  $("#pred-vs").textContent     = `${home} vs ${away}`;
  $("#pred-venue").textContent  = `${p.match.venue} · ${p.match.format} · ${formatDate(p.match.date)}`;
  $("#pred-date").textContent   = formatDate(p.match.date);
  $("#pred-home").textContent   = home;
  $("#pred-away").textContent   = away;
  $("#prob-home-pct").textContent = fmtPct(ph, 1);
  $("#prob-away-pct").textContent = fmtPct(pa, 1);

  $("#prob-home").style.flexBasis = (ph * 100).toFixed(1) + "%";
  $("#prob-away").style.flexBasis = (pa * 100).toFixed(1) + "%";

  $("#pred-favored").innerHTML =
    `<strong>${escape(p.prediction.favored)}</strong> favored at ${p.prediction.favored_pct}% ` +
    `(${escape(p.prediction.confidence_label)}, edge ${p.prediction.edge_pct}%)`;

  const featRows = [
    ["Elo (home / away)",     `${fmt2(p.features.t1_elo_pre)} / ${fmt2(p.features.t2_elo_pre)}`],
    ["Elo diff",              fmt2(p.features.elo_diff_pre)],
    ["Last 5 win-pct",        `${fmtPct(p.features.t1_last5)} / ${fmtPct(p.features.t2_last5)}`],
    ["Last 10 win-pct",       `${fmtPct(p.features.t1_last10)} / ${fmtPct(p.features.t2_last10)}`],
    ["H2H (home POV)",        `${fmtPct(p.features.h2h_t1_winpct)} over ${p.features.h2h_n_prior ?? 0} matches`],
    ["Venue avg 1st-inn",     fmt2(p.features.venue_avg_first)],
    ["Venue toss-winner-wins",fmtPct(p.features.venue_toss_winner_winpct)],
    ["Bat career avg",        `${fmt2(p.features.t1_bat_career_avg)} / ${fmt2(p.features.t2_bat_career_avg)}`],
    ["Bat form SR",           `${fmt2(p.features.t1_bat_form_sr)} / ${fmt2(p.features.t2_bat_form_sr)}`],
    ["Bowl career econ",      `${fmt2(p.features.t1_bowl_career_econ)} / ${fmt2(p.features.t2_bowl_career_econ)}`],
  ];
  $("#pred-features").innerHTML = featRows.map(([k, v]) =>
    `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  const order = ["lgbm_num", "lgbm_cat", "xgb", "cat", "lr", "ensemble"];
  $("#pred-bases").innerHTML = order.map(k => {
    const v = p.base_learners?.[k];
    return `<tr><td>${labelFor(k)}</td><td>${fmtPct(v, 1)}</td></tr>`;
  }).join("");
}

function labelFor(k) {
  return ({
    lgbm_num: "LightGBM (numeric)",
    lgbm_cat: "LightGBM (with cats)",
    xgb:      "XGBoost",
    cat:      "CatBoost",
    lr:       "Logistic Regression",
    ensemble: "Stacked ensemble (final)",
  })[k] || k;
}

function hydrateMetrics(m) {
  if (!m) return;
  const grid = $("#metrics-grid");
  const card = (label, mm, highlight = false) => `
    <div class="metric-card${highlight ? " highlight" : ""}">
      <div class="lbl">${label}</div>
      <div class="big">${fmtPct(mm.acc, 1)}</div>
      <div class="sub">accuracy on ${fmtNum(mm.n)} held-out matches</div>
      <div class="metric-row">
        <span>AUC <strong>${fmt2(mm.auc)}</strong></span>
        <span>Brier <strong>${fmt2(mm.brier)}</strong></span>
        <span>ECE <strong>${fmtPct(mm.ece, 2)}</strong></span>
      </div>
    </div>`;
  grid.innerHTML = [
    card("T20 + IT20", m.t20, true),
    card("ODI",        m.odi),
    card("All formats", m.all),
  ].join("");
}

function hydrateTeams(elementId, teams) {
  if (!teams || !teams.length) return;
  const t = $("#" + elementId);
  t.innerHTML = `
    <thead><tr><th>#</th><th>Team</th><th>As of</th><th class="elo">Elo</th></tr></thead>
    <tbody>
      ${teams.map((row, i) => `
        <tr>
          <td class="rank">${i + 1}</td>
          <td>${escape(row.team)}</td>
          <td class="muted small">${escape(row.as_of)}</td>
          <td class="elo">${fmtNum(row.elo, 1)}</td>
        </tr>`).join("")}
    </tbody>`;
}

let _allRecent = [];
function hydrateRecent(rows) {
  _allRecent = rows || [];
  renderRecent("ALL");
}
function renderRecent(fmt) {
  const t = $("#recent-table");
  const filtered = _allRecent.filter(r => fmt === "ALL" || r.format === fmt);
  if (!filtered.length) {
    t.innerHTML = `<tbody><tr><td class="muted">No matches in this format.</td></tr></tbody>`;
    return;
  }
  t.innerHTML = `
    <thead><tr>
      <th>Date</th><th>Format</th><th>Match</th><th>Venue</th><th>Result</th>
    </tr></thead>
    <tbody>
      ${filtered.map(r => `
        <tr>
          <td>${formatDate(r.date)}</td>
          <td><span class="fmt-tag">${escape(r.format)}</span></td>
          <td>${escape(r.home)} <span class="muted">vs</span> ${escape(r.away)}<br>
              <span class="muted small">${escape(r.competition || "—")}</span></td>
          <td class="muted small">${escape(r.venue || "—")}</td>
          <td><span class="winner">${escape(r.winner || "—")}</span><br>
              <span class="margin">${escape(r.margin)}</span></td>
        </tr>`).join("")}
    </tbody>`;
}
function initFilters() {
  $$(".filter-row .chip").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".filter-row .chip").forEach(b => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      renderRecent(btn.dataset.fmt);
    });
  });
}

function hydrateDataStats(s) {
  if (!s) return;
  const grid = $("#data-stats");
  const card = (k, v) => `<div class="stat-card"><div class="v">${fmtNum(v)}</div><div class="k">${k}</div></div>`;
  grid.innerHTML = [
    card("Matches",      s.matches),
    card("Innings",      s.innings),
    card("Balls",        s.balls),
    card("Player XIs",   s.match_xi),
    card("Venues",       s.distinct_venues),
    card("Teams",        s.distinct_teams),
    card("Competitions", s.distinct_competitions),
  ].join("");
}

function hydrateFooter(generated) {
  if (!generated) return;
  $("#generated-at").textContent = "Data generated " + formatDate(generated, true);
}

function initReveal() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("is-visible");
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.05 });
  $$(".reveal").forEach(el => io.observe(el));
}

function formatDate(s, withTime = false) {
  if (!s) return "—";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  const opts = { year: "numeric", month: "short", day: "numeric" };
  if (withTime) Object.assign(opts, { hour: "2-digit", minute: "2-digit" });
  return d.toLocaleDateString("en-US", opts);
}

function escape(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

// ── Live match card ───────────────────────────────────────────────────────────

function hydrateLive(live) {
  const section = $("#live");
  if (!live || live.status === "Upcoming") {
    section.style.display = "none";
    return;
  }
  section.style.display = "";

  // Status
  const isComplete = live.is_complete;
  const badge = section.querySelector(".badge-live");
  if (isComplete) {
    badge.textContent = "RESULT";
    badge.classList.remove("badge-pulse");
  } else {
    badge.textContent = "LIVE";
    badge.classList.add("badge-pulse");
  }
  $("#live-status").textContent = escape(live.status || "In progress");
  if (live.fetched_at) {
    const d = new Date(live.fetched_at);
    $("#live-updated").textContent = "Updated " + d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  }

  // Scorecard
  $("#live-batting-team").textContent = escape(live.batting_team || live.home || "—");
  $("#live-score").textContent = escape(live.score || "—");
  $("#live-overs").textContent = `${escape(live.overs || "0")} ov`;
  $("#live-crr").textContent   = live.current_rr != null ? `CRR ${Number(live.current_rr).toFixed(2)}` : "CRR —";

  const rrrEl    = $("#live-rrr");
  const targetEl = $("#live-target");
  if (live.target) {
    rrrEl.textContent    = live.required_rr != null ? `RRR ${Number(live.required_rr).toFixed(2)}` : "RRR —";
    targetEl.textContent = `Target ${live.target}`;
    rrrEl.style.display    = "";
    targetEl.style.display = "";
  } else {
    rrrEl.style.display    = "none";
    targetEl.style.display = "none";
  }

  // Batsmen
  const str = live.striker    || {};
  const ns  = live.non_striker || {};
  const bw  = live.bowler      || {};
  $("#live-striker").textContent       = escape(str.name  || "—");
  $("#live-striker-score").textContent = str.runs != null
    ? `${str.runs}(${str.balls ?? "—"})  ${str.fours ?? 0}×4  ${str.sixes ?? 0}×6`
    : "—";
  $("#live-non-striker").textContent       = escape(ns.name || "—");
  $("#live-non-striker-score").textContent = ns.runs != null ? `${ns.runs}(${ns.balls ?? "—"})` : "—";
  $("#live-bowler").textContent        = escape(bw.name || "—");
  $("#live-bowler-figures").textContent = bw.runs != null
    ? `${bw.overs ?? "—"} ov  ${bw.wickets ?? 0}-${bw.runs}`
    : "—";

  if (live.last_overs) {
    $("#live-last-overs").textContent = `Recent: ${escape(live.last_overs)}`;
  }

  // In-play prediction
  const pred = live.live_prediction;
  if (!pred) return;

  const batTeam  = live.batting_team  || live.home || "Batting";
  const bowlTeam = live.bowling_team  || live.away || "Bowling";

  $("#live-bat-label").textContent  = escape(batTeam);
  $("#live-bowl-label").textContent = escape(bowlTeam);

  if (pred.mode === "chase" && pred.win_prob != null) {
    const pBat  = Math.round(pred.win_prob * 100);
    const pBowl = 100 - pBat;
    $("#live-prob-bat-pct").textContent  = `${pBat}%`;
    $("#live-prob-bowl-pct").textContent = `${pBowl}%`;
    $("#live-prob-bat-fill").style.width = `${pBat}%`;
    $("#live-prob-bat-fill").className   = `live-prob-fill ${pBat >= 50 ? "live-prob-bat" : "live-prob-bowl"}`;
    $("#live-pred-mode").textContent = `Chase: need ${live.rem_runs ?? "—"} from ${pred.balls_remaining ?? "—"} balls`;
    $("#live-prob-wrap").style.display = "";
  } else if (pred.mode === "set_score") {
    $("#live-pred-mode").textContent = `Setting: ${pred.balls_remaining ?? "—"} balls remaining`;
    $("#live-prob-wrap").style.display = "none";
  }

  const p50  = pred.p50  ?? "—";
  const p10  = pred.p10  ?? "—";
  const p90  = pred.p90  ?? "—";
  const mean = pred.mean ?? "—";
  $("#live-proj-score").textContent = `${p50} (mean ${mean})`;
  $("#live-proj-range").textContent = `${p10} – ${p90}`;
  $("#live-balls-rem").textContent  = pred.balls_remaining ?? "—";
  $("#live-n-sim").textContent      = pred.n_sim ? fmtNum(pred.n_sim) : "—";

  if (pred.venue_avg) {
    $("#live-venue-avg").textContent = fmtNum(pred.venue_avg, 0);
    $("#live-venue-row").style.display = "";
  }
}
