/* global React, Card, Crosshairs, fmt, teamCode, teamHue,
   ScoreCurve, DivergingBar, OULadder, ScorerLadder, BookmakerDots, LiveProb, Spark,
   useTweaks, TweaksPanel, TweakSection, TweakRadio, TweakToggle, TweakSelect */

const { useState, useEffect, useMemo } = React;

// Static fallback list — slugs use canonical (alphabetical) ordering to
// match what the backend's _write_design_aliases writes. Most fixtures
// are auto-discovered from data.all_predictions on each refresh, so this
// list only matters before the first data.json fetch.
const PRED_FILES = [
  { id: 'kkr_vs_lsg', label: 'KKR · LSG', file: 'data/preds/kkr_vs_lsg.json' },
  { id: 'dc_vs_rcb',  label: 'DC · RCB',  file: 'data/preds/dc_vs_rcb.json' },
  { id: 'csk_vs_gt',  label: 'CSK · GT',  file: 'data/preds/csk_vs_gt.json' },
  { id: 'rr_vs_srh',  label: 'RR · SRH',  file: 'data/preds/rr_vs_srh.json' },
  { id: 'ban_vs_nz',  label: 'BAN · NZ',  file: 'data/preds/ban_vs_nz.json' },
];

// =====================================================================
// MAIN APP
// =====================================================================
// Auto-refresh cadence (ms). Browser pulls fresh data every 30s so the
// dashboard tracks the orchestrator's 30s live_loop without F5. Cheap —
// data.json is a static file served from the orchestrator's HTTP loop.
const REFRESH_INTERVAL_MS = 30_000;

function App() {
  const [data, setData] = useState(null);
  const [preds, setPreds] = useState({});
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
    "selected_match": "kkr_vs_lsg",
    "theme": "dark",
    "format_filter": "T20"
  }/*EDITMODE-END*/;

  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);

  useEffect(() => {
    document.documentElement.dataset.theme = tweaks.theme;
  }, [tweaks.theme]);

  // Single fetch routine — discovers any prediction files dynamically from
  // data.json's all_predictions index, so newly-saved fixtures appear without
  // editing the static PRED_FILES list.
  async function refresh(silent = false) {
    if (!silent) setRefreshing(true);
    try {
      const ts = Date.now();
      const d = await fetch(`data/data.json?_=${ts}`, { cache: "no-cache" }).then(r => r.json());

      // Build prediction-file map from data.json (preferred) + the static fallback list.
      // Use the canonical alphabetical-sort slug — same as the backend's
      // _write_design_aliases — so a Cricbuzz home/away flip on the same
      // match doesn't make the frontend ask for a slug the backend never wrote.
      const fileById = {};
      for (const p of PRED_FILES) fileById[p.id] = p.file;
      for (const p of (d.all_predictions || [])) {
        const id = p?._file?.replace(/\.json$/i, "")?.toLowerCase();
        if (!id) continue;
        const m = p.match || {};
        const home = (window.teamCode ? window.teamCode(m.home) : m.home || "").toLowerCase();
        const away = (window.teamCode ? window.teamCode(m.away) : m.away || "").toLowerCase();
        if (!home || !away) continue;
        const [lo, hi] = [home, away].sort();
        const shortId = `${lo}_vs_${hi}`;
        if (!fileById[shortId]) {
          fileById[shortId] = `data/preds/${shortId}.json`;
        }
      }

      const fetched = await Promise.all(
        Object.entries(fileById).map(async ([id, file]) => {
          try {
            const j = await fetch(`${file}?_=${ts}`, { cache: "no-cache" }).then(r => r.ok ? r.json() : null);
            return j ? [id, j] : null;
          } catch { return null; }
        })
      );
      const psObj = Object.fromEntries(fetched.filter(Boolean));

      setData(d);
      setPreds(psObj);
      setLastUpdate(new Date());
    } catch (e) {
      console.error("refresh failed", e);
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }

  // Initial load
  useEffect(() => { refresh(); }, []);

  // Auto-refresh every REFRESH_INTERVAL_MS — silent (no spinner flash). Pause
  // when the page is hidden so background tabs don't burn cycles.
  useEffect(() => {
    const tick = () => { if (!document.hidden) refresh(true); };
    const id = setInterval(tick, REFRESH_INTERVAL_MS);
    const onVis = () => { if (!document.hidden) refresh(true); };
    document.addEventListener("visibilitychange", onVis);
    return () => { clearInterval(id); document.removeEventListener("visibilitychange", onVis); };
  }, []);

  if (loading) return <div className="frame"><div className="mono small">Loading prediction terminal…</div></div>;
  if (!data) return <div className="frame"><div className="mono small" style={{ color: 'var(--red)' }}>Failed to load data.</div></div>;

  // Fallback to the first available prediction if the selected one is
  // missing — used to hardcode `preds.lsg_vs_kkr` but that breaks when
  // canonical-sorted aliases reorder it to `kkr_vs_lsg`. Picking the
  // first available is robust to slug renames + missing fixtures.
  const pred = preds[tweaks.selected_match] || preds[Object.keys(preds)[0]];
  const live = data.live_match;
  const metrics = data.model_metrics;
  const topT20 = (data.top_teams_t20 || []).slice(0, 10);
  const topODI = (data.top_teams_odi || []).slice(0, 10);
  const recent = (data.recent_matches || []).slice(0, 12);
  const bets = data.bets;

  return (
    <div className="frame">
      <TopBar data={data} preds={preds} lastUpdate={lastUpdate} refreshing={refreshing}
              onRefresh={() => refresh(false)} />
      <Masthead data={data} />

      {/* Live banner — only when there's an actually-playing match.
         Cricbuzz reports `is_complete=false` for ABANDONED and SCHEDULED
         matches too, so plain !is_complete isn't enough; filter those out
         explicitly to avoid a stale banner with last-match innings data. */}
      {isActuallyLive(live) && (
        <LiveStrip live={live} />
      )}

      {/* match selector */}
      <MatchSelector preds={preds} selected={tweaks.selected_match} setTweak={setTweak} />

      {/* TODAY'S CALLS — one row per fixture so you can scan all suggestions at once */}
      <TodaysCalls preds={preds} selected={tweaks.selected_match} setTweak={setTweak} />

      {/* HEADLINE PREDICTION */}
      <PredHeadline pred={pred} />

      {/* SUGGESTION FOR THIS FIXTURE — actionable, what-to-do block */}
      <Suggestion pred={pred} />

      {/* TOP ROW — model probs, market, value */}
      <div className="grid cols-12">
        <div className="col-7">
          <Card title="Win-probability ensemble" right={pred.model.type.replace(/_/g,' ').toUpperCase()}>
            <BaseLearners pred={pred} />
          </Card>
        </div>
        <div className="col-5">
          <Card title="Model-vs-market edge" right={pred.model_vs_book?.value_bet ? '⚑ value bet' : 'no edge'}>
            <ValueBox pred={pred} />
          </Card>
        </div>
      </div>

      {/* BOOKMAKER STRIP */}
      <div className="grid cols-12">
        <div className="col-8">
          <Card title="Bookmaker consensus" right={`${pred.odds?.h2h?.n_books ?? 0} books · snapshot ${pred.odds?.h2h?.snapshot_at?.slice(0,16) || '—'}`}>
            {pred.odds?.h2h?.n_books > 0 ? (
              <>
                <BookmakerDots
                  byBook={pred.odds.h2h.by_book}
                  consensus={pred.odds.h2h.consensus}
                  modelP={pred.prediction.p_home_wins}
                  homeName={pred.match.home}
                  awayName={pred.match.away}
                />
                <BookTable byBook={pred.odds.h2h.by_book} home={pred.match.home} away={pred.match.away} consensus={pred.odds.h2h.consensus} />
              </>
            ) : (
              <div className="small" style={{ color: 'var(--ink-3)' }}>No book lines available for this fixture.</div>
            )}
          </Card>
        </div>
        <div className="col-4">
          <Card title="Line movement · 24h">
            <LineMovement pred={pred} />
          </Card>
          <div style={{ height: 14 }} />
          <Card title="Match features" right="pre-match">
            <FeaturesList pred={pred} />
          </Card>
        </div>
      </div>

      {/* SCORE DISTRIBUTION + O/U */}
      <div className="grid cols-12">
        <div className="col-8">
          <Card title="First-innings score · Monte Carlo" right="5,000 sims · per scenario">
            <div className="row" style={{ marginBottom: 8, gap: 18 }}>
              {Object.keys(pred.totals.scenarios).map(k => {
                const t = k.replace(/_bat_first$/, '');
                return (
                  <span key={k} className="row gap-tight small" style={{ color: 'var(--ink-2)', gap: 6 }}>
                    <span className="legend-dot" style={{ background: teamHue(t) }} />
                    <span>{t} bat first</span>
                  </span>
                );
              })}
              <span className="row gap-tight small" style={{ color: 'var(--ink-2)', gap: 6 }}>
                <span className="legend-dot" style={{ background: 'var(--accent)', borderRadius: 0, width: 10, height: 1 }} />
                <span>Bookmaker totals lines</span>
              </span>
            </div>
            <ScoreCurve scenarios={pred.totals.scenarios} lines={pred.totals.over_under_lines} />
            <ScenarioGrid pred={pred} />
          </Card>
        </div>
        <div className="col-4">
          <Card title="Over / under · 1st innings" right="model-implied">
            <OULadder lines={pred.totals.over_under_lines} />
            <div className="small" style={{ marginTop: 10, color: 'var(--ink-3)' }}>
              <span className="mono">P50</span> = {Math.round(pred.totals.first_innings_p50)} runs ·
              90% interval [{Math.round(pred.totals.first_innings_p10)}, {Math.round(pred.totals.first_innings_p90)}]
            </div>
          </Card>
        </div>
      </div>

      {/* TOP SCORERS */}
      <div className="grid cols-12">
        {Object.entries(pred.top_scorer).map(([team, players], i) => (
          <div key={team} className="col-6">
            <Card title={`Top scorer · ${team}`} right="P(highest 1st-innings score)">
              <ScorerLadder players={players} accent={teamHue(team)} />
            </Card>
          </div>
        ))}
      </div>

      {/* MODEL METRICS + ELO */}
      <div className="grid cols-12">
        <div className="col-4">
          <Card title="Model calibration" right={`n=${metrics.all.n} hold-out matches`}>
            <CalibCard metrics={metrics} />
          </Card>
        </div>
        <div className="col-4">
          <Card title="T20 Elo league" right="top 10 · global">
            <EloTable rows={topT20} />
          </Card>
        </div>
        <div className="col-4">
          <Card title="ODI Elo league" right="top 10 · global">
            <EloTable rows={topODI} />
          </Card>
        </div>
      </div>

      {/* TRACK RECORD — past calls and how they actually went */}
      <div className="grid cols-12">
        <div className="col-12">
          <TrackRecord allPredictions={data.all_predictions || []} />
        </div>
      </div>

      {/* PREDICTION-VERSION HISTORY for the selected fixture (model's call
         evolution as new info — XI, toss — becomes available) */}
      <div className="grid cols-12">
        <div className="col-7">
          <VersionHistory pred={pred} />
        </div>
        <div className="col-5">
          <EdgeTrajectory pred={pred} />
        </div>
      </div>

      {/* PHASE TIMELINE + LEARNINGS — orchestrator activity feed +
         per-version error attribution from completed matches */}
      <div className="grid cols-12">
        <div className="col-7">
          <MatchTimeline events={data.match_timeline || []} />
        </div>
        <div className="col-5">
          <Learnings entries={data.learnings || []} />
        </div>
      </div>

      {/* RECENT + BET LEDGER */}
      <div className="grid cols-12">
        <div className="col-7">
          <Card title="Recent results · all formats" right={`${data.data_stats?.matches?.toLocaleString() || ''} matches in DB`}>
            <RecentTable rows={recent} />
          </Card>
        </div>
        <div className="col-5">
          <Card title="Bet ledger" right={`bankroll ₹${(bets?.bankroll || 0).toLocaleString()}`}>
            <BetLedger bets={bets} />
          </Card>
        </div>
      </div>

      {/* DATA SOURCES */}
      <div className="grid cols-12" style={{ marginTop: 14 }}>
        <div className="col-12">
          <DataSourcesStrip stats={data.data_stats} />
        </div>
      </div>

      <Footer data={data} />

      <TweaksPanel title="TWEAKS · WAUNDERING">
        <TweakSection title="Match in focus">
          <TweakSelect
            label="Fixture"
            value={tweaks.selected_match}
            options={PRED_FILES.map(p => ({ value: p.id, label: p.label }))}
            onChange={v => setTweak('selected_match', v)}
          />
        </TweakSection>
        <TweakSection title="Display">
          <TweakRadio
            label="Theme"
            value={tweaks.theme}
            options={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }]}
            onChange={v => setTweak('theme', v)}
          />
          <TweakRadio
            label="Format filter"
            value={tweaks.format_filter}
            options={[{ value: 'T20', label: 'T20' }, { value: 'ODI', label: 'ODI' }, { value: 'ALL', label: 'All' }]}
            onChange={v => setTweak('format_filter', v)}
          />
        </TweakSection>
      </TweaksPanel>
    </div>
  );
}

// =====================================================================
// SECTIONS
// =====================================================================
function TopBar({ data, preds, lastUpdate, refreshing, onRefresh }) {
  const tickerItems = useMemo(() => {
    const items = [];
    Object.values(preds).forEach(p => {
      if (!p) return;
      items.push({
        teams: `${teamCode(p.match.home)}–${teamCode(p.match.away)}`,
        winner: teamCode(p.prediction.favored),
        pct: p.prediction.favored_pct.toFixed(1),
      });
    });
    return items;
  }, [preds]);

  // "Updated 23s ago" — re-renders every 5s via a forceUpdate hook
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force(x => x + 1), 5000);
    return () => clearInterval(id);
  }, []);
  const ageSec = lastUpdate ? Math.max(0, Math.floor((Date.now() - lastUpdate.getTime()) / 1000)) : null;
  const ageLabel = ageSec == null ? "—"
                   : ageSec < 60 ? `${ageSec}s ago`
                   : ageSec < 3600 ? `${Math.floor(ageSec/60)}m ${ageSec%60}s ago`
                   : `${Math.floor(ageSec/3600)}h ${Math.floor((ageSec%3600)/60)}m ago`;

  const now = new Date(data.generated_at);
  return (
    <div className="topbar">
      <span className="brand"><span className="dot" />WAUNDERING</span>
      <span className="sep">/</span>
      <span>CRICKET PREDICTION TERMINAL · v0.4 · stack_lr_ensemble</span>
      <span className="ticker">
        <span className="ticker-track">
          {[...tickerItems, ...tickerItems].map((it, i) => (
            <span key={i} className="tk-pill">
              <span className="tk-w">{it.teams}</span>
              <span className="tk-m">{it.winner} {it.pct}%</span>
            </span>
          ))}
        </span>
      </span>
      <span className="sep">/</span>
      <span title="Auto-refreshes every 60s">
        UPDATED <strong style={{ color: refreshing ? 'var(--accent-2)' : (ageSec != null && ageSec > 180 ? 'var(--red)' : 'var(--ink)') }}>
          {refreshing ? "fetching…" : ageLabel}
        </strong>
        <button onClick={onRefresh}
                style={{
                  background: 'transparent', border: '1px solid var(--ink-4)',
                  color: 'var(--ink-3)', marginLeft: 8, padding: '2px 6px',
                  fontFamily: 'inherit', fontSize: '10px', cursor: 'pointer',
                  letterSpacing: '0.1em', textTransform: 'uppercase',
                }}
                title="Force refresh now">↻</button>
      </span>
      <span className="sep">/</span>
      <span>{now.toLocaleString('en-GB', { timeZone: 'UTC', hour12: false }).toUpperCase()} UTC</span>
    </div>
  );
}

function Masthead({ data }) {
  const stats = data.data_stats;
  return (
    <header className="masthead">
      <div>
        <div className="strap">VOL. 04 · ISSUE 26 · APR 2026 · DATA AS OF 16:30 UTC</div>
        <h1>The numbers <span className="em">behind</span> tomorrow's pitch.</h1>
        <div className="lede">
          A daily, model-driven readout of every fixture in the public T20 / ODI calendar — head-to-head probabilities from a stacked
          gradient-boosted ensemble, side-by-side with consensus market odds across {pred_n_books(data)}+ bookmakers, Monte-Carlo
          first-innings score distributions, top-scorer probability ladders, live in-match win-probability, and a settled-bet ledger.
        </div>
      </div>
      <div className="stats">
        <div className="cell">
          <div className="v">{(stats?.matches || 0).toLocaleString()}</div>
          <div className="k">Matches in corpus</div>
        </div>
        <div className="cell">
          <div className="v">{((stats?.balls || 0) / 1e6).toFixed(2)}M</div>
          <div className="k">Balls bowled (CricSheet)</div>
        </div>
        <div className="cell">
          <div className="v">{(stats?.distinct_venues || 0).toLocaleString()}</div>
          <div className="k">Venues modelled</div>
        </div>
        <div className="cell">
          <div className="v">{(stats?.distinct_teams || 0).toLocaleString()}</div>
          <div className="k">Teams in Elo system</div>
        </div>
      </div>
    </header>
  );
}
function pred_n_books(data) {
  // best-effort: peek a recent prediction
  return 28;
}

// True only when the match is actively playing (toss done, score updating).
// Excludes scheduled-but-not-started, abandoned/no-result, and complete.
function isActuallyLive(live) {
  if (!live || live.is_complete) return false;
  const s = (live.status || '').toLowerCase();
  if (!s) return false;
  if (s.includes('starts at') || s.includes('match starts')) return false;
  if (s.includes('abandon') || s.includes('no result')) return false;
  return true;
}

function LiveStrip({ live }) {
  return (
    <div className="live" style={{ marginTop: 16 }}>
      <Crosshairs />
      <div className="stack" style={{ gap: 4 }}>
        <span className="pill"><span className="led" />LIVE · IN PROGRESS</span>
        <span className="serif" style={{ fontSize: 22 }}>{teamCode(live.home)} vs {teamCode(live.away)}</span>
        <span className="small" style={{ color: 'var(--ink-2)' }}>{live.status}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, auto)', gap: 28, justifyContent: 'center' }}>
        <Stat label="Innings 1" value={live.innings?.[0]?.score || '—'} sub={live.innings?.[0]?.team} />
        <Stat label="Innings 2" value={`${live.score || '—'} (${live.overs?.toFixed(1) || '—'})`} sub={live.batting_team} />
        <Stat label="Target" value={live.target ?? '—'} sub={live.bowling_team + ' set'} />
        <Stat label="Req. RR" value={(live.required_rr ?? 0).toFixed(2)} sub={`vs current ${(live.current_rr ?? 0).toFixed(2)}`} />
      </div>
      <div style={{ width: 280 }}>
        <LiveProb live={live} />
      </div>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="stack" style={{ gap: 0 }}>
      <span className="label">{label}</span>
      <span className="mono" style={{ fontSize: 22, fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.02em' }}>{value}</span>
      {sub && <span className="small" style={{ color: 'var(--ink-3)' }}>{sub}</span>}
    </div>
  );
}

// Sort fixtures: live > upcoming-by-soonest > completed-by-recency
function _fixtureSortKey(p) {
  if (!p) return [9, "9999"];
  const status = p?.result?.status || "unknown";
  const date = p?.match?.date || "9999";
  if (status === "in_progress")     return [0, date];     // live now → top
  if (status === "awaiting_result") return [1, date];     // just-finished, parsing
  if (status !== "complete")        return [2, date];     // future / no-result-yet
  return [3, "Z" + date];                                 // completed (newest first via reverse-sort)
}

function sortedFixtures(preds) {
  const entries = Object.entries(preds).filter(([_, p]) => p);
  return entries.sort(([_a, a], [_b, b]) => {
    const ka = _fixtureSortKey(a);
    const kb = _fixtureSortKey(b);
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    // within bucket: upcoming = soonest first; completed = most recent first
    if (ka[0] === 3) return kb[1].localeCompare(ka[1]);
    return ka[1].localeCompare(kb[1]);
  });
}

function MatchSelector({ preds, selected, setTweak }) {
  const ordered = sortedFixtures(preds);
  // Auto-select sensible default if the current `selected` no longer maps to a fixture
  React.useEffect(() => {
    if (!ordered.length) return;
    const ids = ordered.map(([id]) => id);
    if (!ids.includes(selected)) setTweak('selected_match', ids[0]);
  }, [JSON.stringify(ordered.map(([id]) => id))]);

  const statusBadge = (p) => {
    const s = p?.result?.status;
    const ls = (p?.result?.live_status || "").toLowerCase();
    if (s === "in_progress") {
      // Cricbuzz reports `is_complete=false` for both scheduled and abandoned
      // matches, so the prediction's coarse "in_progress" needs disambiguation.
      if (ls.includes("starts at") || ls.includes("match starts"))
        return { label: "UPCOMING",  color: "var(--accent-2)", pulse: false };
      if (ls.includes("abandon") || ls.includes("no result"))
        return { label: "ABANDONED", color: "var(--ink-3)",    pulse: false };
      return { label: "LIVE", color: "var(--green)", pulse: true };
    }
    if (s === "awaiting_result") return { label: "SETTLING", color: "var(--accent)", pulse: false };
    if (s === "complete") {
      return { label: p.result.correct ? "WON" : "LOST", color: p.result.correct ? "var(--green)" : "var(--red)", pulse: false };
    }
    return { label: "UPCOMING", color: "var(--accent-2)", pulse: false };
  };

  return (
    <div className="row wrap" style={{ marginTop: 18, gap: 8 }}>
      <span className="label" style={{ marginRight: 4 }}>Fixtures →</span>
      {ordered.map(([id, pr]) => {
        const fav = pr.prediction.favored;
        const sb  = statusBadge(pr);
        return (
          <button
            key={id}
            className={`btn ${selected === id ? 'on' : ''}`}
            onClick={() => setTweak('selected_match', id)}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}
          >
            <span style={{
              fontSize: 9, padding: '1px 5px',
              border: `1px solid ${sb.color}`, color: sb.color,
              letterSpacing: '0.1em',
              animation: sb.pulse ? 'pulse 1.6s ease-in-out infinite' : undefined,
            }}>{sb.label}</span>
            <span>{teamCode(pr.match.home)} <span style={{ color: 'var(--ink-4)' }}>·</span> {teamCode(pr.match.away)}</span>
            <span style={{ color: selected === id ? 'var(--bg)' : 'var(--accent)' }}>
              {teamCode(fav)} {pr.prediction.favored_pct.toFixed(0)}%
            </span>
          </button>
        );
      })}
    </div>
  );
}

function PredHeadline({ pred }) {
  const m = pred.match;
  const p = pred.prediction;
  const homeFav = m.home === p.favored;
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <Crosshairs />
      <div className="hd">
        <div className="t">
          <span style={{ color: 'var(--ink)' }}>FIXTURE</span>
          <span className="sep" style={{ margin: '0 8px', color: 'var(--ink-4)' }}>·</span>
          <span>{m.format} · {fmt.date(m.date)} · {fmt.short(m.venue, 60)}</span>
        </div>
        <div className="r">{p.confidence_label.toUpperCase()}</div>
      </div>
      <div className="predstrip">
        <div className="side">
          <span className="label">Home · {teamCode(m.home)}</span>
          <span className="nm" style={{ color: teamHue(m.home) }}>{m.home}</span>
          <span className={`pct ${homeFav ? 'win' : 'dim'}`}>{(p.p_home_wins*100).toFixed(1)}%</span>
          <span className="small" style={{ color: 'var(--ink-3)' }}>
            implied odds {fmt.odds(1/p.p_home_wins)} · ELO {Math.round(pred.features.t1_elo_pre)}
          </span>
        </div>
        <div className="vs">
          <span className="lab">VS</span>
          <span style={{ color: 'var(--ink)', fontSize: 16 }}>EDGE</span>
          <span className="mono" style={{ fontSize: 22, color: 'var(--accent)' }}>{p.edge_pct.toFixed(1)}<span style={{ fontSize: 11 }}>pp</span></span>
          <span className="lab">{p.favored === m.home ? 'HOME LEANS' : 'AWAY LEANS'}</span>
        </div>
        <div className="side away">
          <span className="label">Away · {teamCode(m.away)}</span>
          <span className="nm" style={{ color: teamHue(m.away) }}>{m.away}</span>
          <span className={`pct ${!homeFav ? 'win' : 'dim'}`}>{(p.p_away_wins*100).toFixed(1)}%</span>
          <span className="small" style={{ color: 'var(--ink-3)' }}>
            ELO {Math.round(pred.features.t2_elo_pre)} · implied odds {fmt.odds(1/p.p_away_wins)}
          </span>
        </div>
      </div>
    </div>
  );
}

// =====================================================================
// SUGGESTION — actionable, per-fixture call
// =====================================================================
//
// Rules (mirror cricket_pipeline/work/bet_engine.py + HOW_TO_USE.md playbooks)
//   edge < 3pp     → PASS (no edge, vig eats it)
//   edge 3-5pp     → BET (full Kelly, sized at recommended stake)
//   edge 5-8pp     → BET (consider sizing down 30-50%)
//   edge 8-15pp    → INVESTIGATE (likely model blind spot — verify XI/news)
//   edge > 15pp    → SKIP (auto-blocked, almost always model error)
//   no odds        → WAITING_ON_MARKET
//
// Outputs a single "what to do" card + supporting reasons + wait-conditions.
function suggestionFor(pred) {
  const cons = pred?.odds?.h2h?.consensus;
  const mvb  = pred?.model_vs_book;
  const m    = pred.match;
  const venueHasDew = /Mumbai|Chennai|Hyderabad|Lucknow|Kolkata|Bengaluru|Delhi|Chandigarh|Ahmedabad|Jaipur|Pune|Mohali|Visakhapatnam|Cuttack|Nagpur|Indore|Ranchi|Guwahati|Dharamsala|Mirpur|Chattogram|Sylhet|Karachi|Lahore|Multan|Rawalpindi|Colombo|Pallekele|Galle/i.test(m.venue || "");
  const isNight = (m.format !== "Test"); // proxy: T20/ODI mostly night games
  const fmtTier = m.format === "Test" ? "test" : (m.format === "ODI" ? "odi" : "t20");
  const waits = [];

  if (!cons || !mvb) {
    return {
      verdict: "WAITING",
      tone:    "ghost",
      headline: "Waiting on market data",
      pick:    null, odds: null, stake: null, edge: null, kelly: null,
      reasons: ["No odds in the snapshot for this fixture yet — odds_loop polls on a quota-aware schedule."],
      waits:   ["Re-check after the next odds tick (≤30 min if match is within 6h of toss)."],
    };
  }

  const edge = mvb.best_side_edge_pp ?? Math.max(mvb.edge_home_pp, mvb.edge_away_pp);
  const sideTeam = mvb.best_side;
  const odds = mvb.best_odds;
  const kelly = mvb.kelly_fraction;
  const stake = (kelly && kelly > 0) ? Math.min(kelly * 1000, 50) : null; // approx (BANKROLL=1000 default)

  // Wait-conditions (informational, regardless of verdict)
  if (venueHasDew && isNight && fmtTier !== "test")
    waits.push("Night game in subcontinent — dew can swing the chase line ±5pp. Wait until the toss is confirmed.");
  if ((pred.features?.t1_career_n || 0) < 5 || (pred.features?.t2_career_n || 0) < 5)
    waits.push("One or both teams have <5 prior matches in the corpus — form features may be unreliable.");

  // Verdict tree
  if (edge < 3) {
    return {
      verdict: "PASS", tone: "ghost",
      headline: `Pass — edge ${edge.toFixed(1)}pp below 3pp threshold`,
      pick: sideTeam, odds, stake: null, edge, kelly: null,
      reasons: [
        `Model and market agree within ${Math.abs(edge).toFixed(1)}pp.`,
        `${cons.n_books} books · book vig ~${((cons.spread_pp || 0)).toFixed(1)}pp spread.`,
      ],
      waits,
    };
  }

  if (edge > 15) {
    return {
      verdict: "SKIP", tone: "red",
      headline: `Skip — edge ${edge.toFixed(1)}pp is suspicious, not opportunity`,
      pick: sideTeam, odds, stake: null, edge, kelly: null,
      reasons: [
        `Model says ${pred.match.home === sideTeam ? (pred.prediction.p_home_wins*100).toFixed(1) : (pred.prediction.p_away_wins*100).toFixed(1)}% on ${sideTeam}.`,
        `Market consensus across ${cons.n_books} books has it at ${pred.match.home === sideTeam ? (cons.p_home*100).toFixed(1) : (cons.p_away*100).toFixed(1)}%.`,
        "Disagreements this large almost always reflect model blind spots — stale lineup, missed news, wrong pitch read.",
      ],
      waits: [
        "Cross-check the announced XI vs the model's lineup proxy.",
        "Check Cricinfo / Cricbuzz for late injury news.",
        "If you can't find what the books know, trust the books — pass.",
      ],
    };
  }

  if (edge > 8) {
    return {
      verdict: "INVESTIGATE", tone: "amber",
      headline: `Investigate — edge ${edge.toFixed(1)}pp possible but suspicious`,
      pick: sideTeam, odds, stake: stake ? stake * 0.4 : null, edge, kelly,
      reasons: [
        `Model has ${sideTeam} at +${edge.toFixed(1)}pp vs market consensus.`,
        `${cons.n_books} books · best price ${odds.toFixed(2)}.`,
        "8-15pp gaps sometimes catch real injury/news lag — but more often, the books know something we don't.",
      ],
      waits: [
        "Verify the announced XI matches our lineup proxy (we use each side's most-recent prior XI).",
        "Check pitch/weather report.",
        "If everything checks out, size at ~40% of recommended stake; otherwise pass.",
      ],
    };
  }

  // 3-8pp: act
  const downsize = edge > 5;
  return {
    verdict: "BET", tone: "green",
    headline: `Bet ${sideTeam} @ ${odds.toFixed(2)}` + (downsize ? " (consider sizing down)" : ""),
    pick: sideTeam, odds,
    stake: stake ? (downsize ? stake * 0.7 : stake) : null,
    edge, kelly,
    reasons: [
      `Model edge +${edge.toFixed(1)}pp on ${sideTeam} vs the consensus of ${cons.n_books} bookmakers.`,
      `Best price ${odds.toFixed(2)} (book implied ${(100 / odds).toFixed(1)}%, model implied ${pred.match.home === sideTeam ? (pred.prediction.p_home_wins*100).toFixed(1) : (pred.prediction.p_away_wins*100).toFixed(1)}%).`,
      `Half-Kelly stake = ${(kelly*100).toFixed(2)}% of bankroll, capped at 5%.`,
    ],
    waits,
  };
}

function Suggestion({ pred }) {
  const sg = suggestionFor(pred);
  const verdictColor = {
    BET: 'var(--green)', PASS: 'var(--ink-3)', INVESTIGATE: 'var(--accent)',
    SKIP: 'var(--red)', WAITING: 'var(--book)',
  }[sg.verdict] || 'var(--ink)';

  return (
    <div className="card" style={{ marginTop: 14, borderColor: verdictColor }}>
      <Crosshairs />
      <div className="hd" style={{ borderColor: verdictColor }}>
        <div className="t" style={{ color: verdictColor, letterSpacing: '0.18em' }}>
          ⚑ Suggestion · {pred.match.format} · {teamCode(pred.match.home)} vs {teamCode(pred.match.away)}
        </div>
        <div className="r">{pred.prediction.confidence_label?.toUpperCase()}</div>
      </div>
      <div className="bd" style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 24 }}>
        {/* LEFT: action block */}
        <div className="stack" style={{ gap: 14 }}>
          <div>
            <div className="label">Verdict</div>
            <div className="serif" style={{ fontSize: 30, color: verdictColor, lineHeight: 1.1 }}>
              {sg.verdict}{sg.headline ? ' — ' : ''}<span style={{ fontStyle: 'normal', fontFamily: 'var(--sans)', fontSize: 20, color: 'var(--ink)' }}>{sg.headline}</span>
            </div>
          </div>

          {sg.pick && (
            <div className="stack" style={{ gap: 6 }}>
              <div className="label">Action</div>
              <div className="row wrap" style={{ gap: 18, alignItems: 'baseline' }}>
                <div>
                  <div className="tiny">Pick</div>
                  <div className="num" style={{ fontSize: 18, color: 'var(--ink)' }}>{sg.pick}</div>
                </div>
                {sg.odds != null && (
                  <div>
                    <div className="tiny">Min odds</div>
                    <div className="num" style={{ fontSize: 18, color: 'var(--accent-2)' }}>{sg.odds.toFixed(2)}</div>
                  </div>
                )}
                {sg.stake != null && (
                  <div>
                    <div className="tiny">Stake</div>
                    <div className="num" style={{ fontSize: 18, color: 'var(--ink)' }}>${sg.stake.toFixed(2)}</div>
                  </div>
                )}
                {sg.edge != null && (
                  <div>
                    <div className="tiny">Edge</div>
                    <div className="num" style={{ fontSize: 18, color: verdictColor }}>{sg.edge >= 0 ? '+' : ''}{sg.edge.toFixed(1)}pp</div>
                  </div>
                )}
                {sg.kelly != null && (
                  <div>
                    <div className="tiny">Kelly</div>
                    <div className="num" style={{ fontSize: 18, color: 'var(--ink-2)' }}>{(sg.kelly*100).toFixed(2)}%</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: reasons + waits */}
        <div className="stack" style={{ gap: 12 }}>
          <div>
            <div className="label">Why</div>
            <ul style={{ margin: '6px 0 0', paddingLeft: 18, lineHeight: 1.5 }}>
              {sg.reasons.map((r, i) => (
                <li key={i} className="small" style={{ color: 'var(--ink-2)' }}>{r}</li>
              ))}
            </ul>
          </div>
          {sg.waits.length > 0 && (
            <div>
              <div className="label">Wait conditions</div>
              <ul style={{ margin: '6px 0 0', paddingLeft: 18, lineHeight: 1.5 }}>
                {sg.waits.map((w, i) => (
                  <li key={i} className="small" style={{ color: 'var(--accent)' }}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =====================================================================
// TODAY'S CALLS — one row per fixture, click to focus
// =====================================================================
function TodaysCalls({ preds, selected, setTweak }) {
  // Live > upcoming > completed (matches MatchSelector ordering)
  const rows = sortedFixtures(preds).map(([id, p]) => ({ id, p, sg: suggestionFor(p) }));
  if (!rows.length) return null;
  const verdictColor = (v) => ({
    BET: 'var(--green)', PASS: 'var(--ink-3)', INVESTIGATE: 'var(--accent)',
    SKIP: 'var(--red)', WAITING: 'var(--book)',
    WON: 'var(--green)', LOST: 'var(--red)',
    LIVE: 'var(--green)', SETTLING: 'var(--accent)',
    ABANDONED: 'var(--ink-3)', UPCOMING: 'var(--accent-2)',
  }[v] || 'var(--ink)');

  // For settled fixtures the pre-match advisory (BET/SKIP/etc.) is irrelevant;
  // overlay the actual outcome so the table reflects what happened. Judge
  // WON/LOST against the row's displayed Pick (the betting recommendation
  // from model_vs_book), falling back to the model's overall favored when
  // there was no recommendation (PASS/WAITING). This keeps Pick and Verdict
  // self-consistent — e.g. LSG vs KKR with Pick=KKR and KKR-won-the-super-over
  // shows WON, not LOST (which would be the case if we judged against the
  // model's overall favored team LSG).
  const overrideForSettled = (p, sg) => {
    const r = p?.result || {};
    const status = r.status;
    if (status === "complete") {
      const winner = r.winner;
      const judged = sg.pick || p.prediction?.favored;
      if (!winner || !judged) return { ...sg, verdict: "RESULT", headline: r.live_status || "" };
      const won = winner === judged;
      return { ...sg, verdict: won ? "WON" : "LOST",
               headline: `${won ? "Won" : "Lost"} — ${r.live_status || ''}`.trim() };
    }
    if (status === "awaiting_result")
      return { ...sg, verdict: "SETTLING", headline: r.live_status || "Awaiting final scoreline" };
    // in_progress: only label as LIVE if there's actual scoring action,
    // else treat as scheduled/abandoned
    if (status === "in_progress") {
      const ls = (r.live_status || "").toLowerCase();
      if (ls.includes("starts at") || ls.includes("match starts"))
        return { ...sg, verdict: "UPCOMING", headline: r.live_status };
      if (ls.includes("abandon") || ls.includes("no result"))
        return { ...sg, verdict: "ABANDONED", headline: r.live_status };
      return { ...sg, verdict: "LIVE", headline: r.live_status || sg.headline };
    }
    return sg;
  };

  return (
    <Card title="Today's calls" right={`${rows.length} fixtures · click to focus`} className="ghost" pad={false}>
      <table className="t">
        <thead>
          <tr>
            <th>Fixture</th>
            <th>Format</th>
            <th>Verdict</th>
            <th>Pick</th>
            <th className="num">Odds</th>
            <th className="num">Stake</th>
            <th className="num">Edge</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ id, p, sg: rawSg }) => {
            const isSel = id === selected;
            const sg = overrideForSettled(p, rawSg);
            const settled = p?.result?.status === "complete";
            // For settled rows the model's "pick" is the favored team; show
            // it even when we wouldn't otherwise have shown a Pick column.
            const displayPick = settled
              ? (p.prediction?.favored || sg.pick || '—')
              : (sg.pick || '—');
            return (
              <tr key={id}
                  onClick={() => setTweak('selected_match', id)}
                  style={{
                    cursor: 'pointer',
                    background: isSel ? 'var(--bg-3)' : 'transparent',
                  }}>
                <td>
                  <strong style={{ color: 'var(--ink)' }}>{teamCode(p.match.home)}</strong>
                  <span className="dim"> vs </span>
                  <strong style={{ color: 'var(--ink)' }}>{teamCode(p.match.away)}</strong>
                  <div className="dim" style={{ fontSize: 10 }}>{fmt.date(p.match.date)} · {fmt.short(p.match.venue, 30)}</div>
                </td>
                <td className="dim">{p.match.format}</td>
                <td>
                  <span className="chip" style={{
                    color: verdictColor(sg.verdict),
                    borderColor: `color-mix(in oklch, ${verdictColor(sg.verdict)} 50%, var(--line))`,
                  }}>{sg.verdict}</span>
                </td>
                <td>{displayPick !== '—' ? teamCode(displayPick) : '—'}</td>
                <td className="num">{sg.odds ? sg.odds.toFixed(2) : '—'}</td>
                <td className="num">{!settled && sg.stake ? '$' + sg.stake.toFixed(0) : '—'}</td>
                <td className="num" style={{ color: sg.edge != null ? verdictColor(sg.verdict) : 'var(--ink-3)' }}>
                  {sg.edge != null ? (sg.edge >= 0 ? '+' : '') + sg.edge.toFixed(1) + 'pp' : '—'}
                </td>
                <td className="dim" style={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {sg.headline}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

function BaseLearners({ pred }) {
  const bl = pred.base_learners;
  const rows = [
    { label: 'LGBM·NUM', value: bl.lgbm_num },
    { label: 'LGBM·CAT', value: bl.lgbm_cat },
    { label: 'XGBOOST',  value: bl.xgb },
    { label: 'CATBOOST', value: bl.cat },
    { label: 'LOG·REG',  value: bl.lr },
    { label: 'STACK·ENS', value: bl.ensemble, accent: 'var(--accent)' },
  ].filter(r => r.value != null);
  return (
    <div>
      <div className="small" style={{ color: 'var(--ink-3)', marginBottom: 6 }}>
        Each base learner outputs P(home wins). The stacked logistic regression blends them.
      </div>
      <DivergingBar rows={rows} width={520} />
      <div className="grid cols-3" style={{ marginTop: 10, gap: 6 }}>
        <Kv k="ELO·DIFF" v={(pred.features.elo_diff_pre).toFixed(1)} />
        <Kv k="H2H WIN%" v={(pred.features.h2h_t1_winpct*100).toFixed(0)+'%'} />
        <Kv k="H2H N" v={pred.features.h2h_n_prior} />
        <Kv k="HOME L5" v={(pred.features.t1_last5*100).toFixed(0)+'%'} />
        <Kv k="AWAY L5" v={(pred.features.t2_last5*100).toFixed(0)+'%'} />
        <Kv k="VENUE 1ST WIN%" v={(pred.features.venue_bat1_winrate*100).toFixed(0)+'%'} />
      </div>
    </div>
  );
}

function Kv({ k, v }) {
  return (
    <div className="kv" style={{ borderBottom: '1px dashed var(--line)' }}>
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}

function ValueBox({ pred }) {
  const mvb = pred.model_vs_book;
  if (!mvb) return <div className="small">No book data.</div>;
  const isValue = mvb.value_bet;
  const cons = pred.odds?.h2h?.consensus;
  return (
    <div>
      <div className="row spread">
        <div>
          <div className="label">Best side · model edge</div>
          <div className="serif" style={{ fontSize: 28, color: isValue ? 'var(--green)' : 'var(--ink-2)' }}>
            {mvb.best_side}
          </div>
        </div>
        <div className={`sticker ${isValue ? 'value' : 'fade'}`}>
          {isValue ? '⚑ VALUE' : 'NO EDGE'}
        </div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 14, gap: 1, background: 'var(--line)', border: '1px solid var(--line)' }}>
        <BigCell k="EDGE" v={fmt.pp(mvb.best_side_edge_pp)} accent={isValue ? 'var(--green)' : 'var(--ink-2)'} />
        <BigCell k="BEST ODDS" v={fmt.odds(mvb.best_odds)} />
        <BigCell k="KELLY" v={(mvb.kelly_fraction*100).toFixed(2)+'%'} accent={isValue ? 'var(--accent)' : 'var(--ink-2)'} />
      </div>
      <hr className="dash" />
      <div className="stack" style={{ gap: 6 }}>
        <div className="kv"><span className="k">Model · home</span>
          <span className="v"><span className="legend-dot dot-model" /> {fmt.pct(pred.prediction.p_home_wins,1)}</span></div>
        <div className="kv"><span className="k">Book consensus · home</span>
          <span className="v"><span className="legend-dot dot-book" /> {cons ? fmt.pct(cons.p_home,1) : '—'}</span></div>
        <div className="kv"><span className="k">Edge home</span>
          <span className="v" style={{ color: mvb.edge_home_pp >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmt.pp(mvb.edge_home_pp)}</span></div>
        <div className="kv"><span className="k">Edge away</span>
          <span className="v" style={{ color: mvb.edge_away_pp >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmt.pp(mvb.edge_away_pp)}</span></div>
      </div>
    </div>
  );
}

function BigCell({ k, v, accent }) {
  return (
    <div style={{ background: 'var(--bg-2)', padding: '10px 12px' }}>
      <div className="label">{k}</div>
      <div className="mono" style={{ fontSize: 24, color: accent || 'var(--ink)', fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.02em' }}>{v}</div>
    </div>
  );
}

function BookTable({ byBook, home, away, consensus }) {
  const rows = Object.entries(byBook)
    .map(([name, b]) => ({ name, ...b }))
    .sort((a, b) => a.odds_home - b.odds_home);
  const top = rows.slice(0, 8);
  return (
    <div style={{ marginTop: 8 }}>
      <table className="t">
        <thead>
          <tr>
            <th>BOOK</th>
            <th className="num">{teamCode(home)} ODDS</th>
            <th className="num">{teamCode(away)} ODDS</th>
            <th className="num">{teamCode(home)} P</th>
            <th className="num">VIG</th>
          </tr>
        </thead>
        <tbody>
          {top.map(r => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td className="num">{fmt.odds(r.odds_home)}</td>
              <td className="num">{fmt.odds(r.odds_away)}</td>
              <td className="num">{(r.p_home*100).toFixed(1)}%</td>
              <td className="num dim">{r.vig_pct.toFixed(2)}%</td>
            </tr>
          ))}
          {consensus && (
            <tr style={{ borderTop: '1px solid var(--line)' }}>
              <td style={{ color: 'var(--book)', fontWeight: 600 }}>CONSENSUS · {consensus.n_books}</td>
              <td className="num" style={{ color: 'var(--book)' }}>{fmt.odds(1/consensus.p_home)}</td>
              <td className="num" style={{ color: 'var(--book)' }}>{fmt.odds(1/consensus.p_away)}</td>
              <td className="num" style={{ color: 'var(--book)' }}>{(consensus.p_home*100).toFixed(1)}%</td>
              <td className="num dim">spread {consensus.spread_pp.toFixed(2)}pp</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function LineMovement({ pred }) {
  const lm = pred.odds?.line_movement_24h;
  if (!lm) return <div className="small" style={{ color: 'var(--ink-3)' }}>No book data.</div>;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row spread">
        <span className="label">Δ p(home)</span>
        <span className="mono" style={{ color: lm.delta_pp_home >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {fmt.pp(lm.delta_pp_home)}
        </span>
      </div>
      <Spark values={lm.n_snaps > 1 ? null : null} />
      <div className="kv"><span className="k">First snap</span><span className="v">{(lm.first_snap || '').slice(11,16) || '—'}</span></div>
      <div className="kv"><span className="k">Last snap</span><span className="v">{(lm.last_snap || '').slice(11,16) || '—'}</span></div>
      <div className="kv"><span className="k">N snaps</span><span className="v">{lm.n_snaps}</span></div>
      <div className="small" style={{ color: 'var(--ink-3)' }}>
        Movement is computed from polled book snapshots. With one snap the line is by definition flat — more snapshots stream in approaching toss.
      </div>
    </div>
  );
}

function FeaturesList({ pred }) {
  const f = pred.features;
  const m = pred.match;
  const rows = [
    ['Venue avg · 1st inn',     Math.round(f.venue_avg_first) + ' runs'],
    ['Bat-1st win rate',        (f.venue_bat1_winrate*100).toFixed(0) + '%'],
    ['Toss-winner win rate',    (f.venue_toss_winner_winpct*100).toFixed(0) + '%'],
    [`${teamCode(m.home)} bat avg`, f.t1_bat_career_avg.toFixed(1)],
    [`${teamCode(m.away)} bat avg`, f.t2_bat_career_avg.toFixed(1)],
    [`${teamCode(m.home)} form SR`, f.t1_bat_form_sr.toFixed(1)],
    [`${teamCode(m.away)} form SR`, f.t2_bat_form_sr.toFixed(1)],
    [`${teamCode(m.home)} bowl econ`, f.t1_bowl_career_econ.toFixed(2)],
    [`${teamCode(m.away)} bowl econ`, f.t2_bowl_career_econ.toFixed(2)],
  ];
  return (
    <div>
      {rows.map(([k, v]) => <Kv key={k} k={k} v={v} />)}
    </div>
  );
}

function ScenarioGrid({ pred }) {
  const sc = pred.totals.scenarios;
  return (
    <div className="grid cols-2" style={{ marginTop: 16, gap: 10 }}>
      {Object.entries(sc).map(([t, v]) => (
        <div key={t} className="card flat">
          <div style={{ padding: '10px 12px' }}>
            <div className="label">If {t} bat 1st</div>
            <div className="row" style={{ gap: 16, marginTop: 6 }}>
              <Stat label="P10" value={Math.round(v.p10)} sub="bottom 10%" />
              <Stat label="P50" value={Math.round(v.p50)} sub="median" />
              <Stat label="P90" value={Math.round(v.p90)} sub="top 10%" />
              <Stat label="Spread" value={v.spread.toFixed(1)} sub="P90–P10" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function CalibCard({ metrics }) {
  const sets = ['t20', 'odi', 'all'];
  return (
    <div>
      <table className="t" style={{ fontSize: 10.5 }}>
        <thead><tr>
          <th style={{ padding: '4px 4px' }}>SET</th>
          <th className="num" style={{ padding: '4px 4px' }}>N</th>
          <th className="num" style={{ padding: '4px 4px' }}>ACC</th>
          <th className="num" style={{ padding: '4px 4px' }}>AUC</th>
          <th className="num" style={{ padding: '4px 4px' }}>BRIER</th>
          <th className="num" style={{ padding: '4px 4px' }}>ECE</th>
        </tr></thead>
        <tbody>
          {sets.map(s => {
            const m = metrics[s];
            return (
              <tr key={s}>
                <td style={{ textTransform: 'uppercase', padding: '5px 4px' }}>{s}</td>
                <td className="num" style={{ padding: '5px 4px' }}>{m.n.toLocaleString()}</td>
                <td className="num" style={{ padding: '5px 4px' }}>{(m.acc*100).toFixed(1)}%</td>
                <td className="num" style={{ padding: '5px 4px' }}>{m.auc.toFixed(3)}</td>
                <td className="num" style={{ padding: '5px 4px' }}>{m.brier.toFixed(3)}</td>
                <td className="num" style={{ padding: '5px 4px' }}>{(m.ece*100).toFixed(2)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="small" style={{ marginTop: 10, color: 'var(--ink-3)' }}>
        ACC = win-pick accuracy · AUC = ranking · Brier = sq-error · ECE = expected calibration error.
        T20 ECE 3.16% means model probabilities are well-calibrated within ±3pp of empirical.
      </div>
    </div>
  );
}

function EloTable({ rows }) {
  const max = Math.max(...rows.map(r => r.elo));
  const min = Math.min(...rows.map(r => r.elo));
  return (
    <table className="t">
      <thead><tr><th>#</th><th>TEAM</th><th className="num">ELO</th><th className="num">AS-OF</th></tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={r.team}>
            <td className="dim">{String(i+1).padStart(2,'0')}</td>
            <td>{r.team}</td>
            <td className="num" style={{ position: 'relative', minWidth: 80 }}>
              <span style={{
                position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)',
                height: 4, width: ((r.elo - min) / (max - min) * 50) + 'px',
                background: 'var(--accent-2)', opacity: 0.4,
              }} />
              <span style={{ position: 'relative' }}>{r.elo.toFixed(1)}</span>
            </td>
            <td className="num dim">{r.as_of}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// =====================================================================
// TRACK RECORD — settled calls, ordered newest first
// =====================================================================
function TrackRecord({ allPredictions }) {
  // Settled = match ended AND we have a verdict (correct true/false)
  // Pending = match ended but the live source hasn't published a clean winner yet
  const settled = allPredictions
    .filter(p => p?.result?.status === "complete" && p?.result?.correct != null)
    .sort((a, b) => (b.match.date || "").localeCompare(a.match.date || ""));
  const awaiting = allPredictions
    .filter(p => p?.result?.status === "awaiting_result")
    .sort((a, b) => (b.match.date || "").localeCompare(a.match.date || ""));

  const n = settled.length;
  const correct = settled.filter(p => p.result.correct).length;
  const wrong   = n - correct;
  const hitRate = n > 0 ? correct / n : null;

  // Confidence bucket
  const hiConf = settled.filter(p => p.prediction.favored_pct >= 65);
  const hiConfHit = hiConf.filter(p => p.result.correct).length;
  const hiConfRate = hiConf.length > 0 ? hiConfHit / hiConf.length : null;

  // Bookmaker comparison: where we have book consensus AND were correct, did the book also pick this side?
  const withBook = settled.filter(p => p?.odds?.h2h?.consensus);
  const beatBook = withBook.filter(p => {
    const ph_book = p.odds.h2h.consensus.p_home;
    const ph_model = p.prediction.p_home_wins;
    const actual_home = p.result.winner === p.match.home ? 1 : 0;
    // model "correct" but book wrong = beat the book
    const model_pick_home = ph_model >= 0.5;
    const book_pick_home  = ph_book  >= 0.5;
    return p.result.correct && (model_pick_home !== book_pick_home);
  }).length;

  return (
    <Card title="Track record · settled calls" right={n === 0
      ? "no settled calls yet"
      : `${correct}-${wrong} record · ${(hitRate*100).toFixed(0)}% hit rate`}>
      {/* Awaiting-result strip: matches that have ENDED but live source is still parsing */}
      {awaiting.length > 0 && (
        <div className="small" style={{
          marginBottom: 12, padding: '8px 12px',
          border: '1px dashed var(--accent)', borderRadius: 4,
          color: 'var(--accent)', background: 'rgba(255,184,74,0.06)',
        }}>
          ⏳ <strong>Awaiting final scoreline</strong> ({awaiting.length}):
          {' '}
          {awaiting.map((p, i) => (
            <span key={i} style={{ marginRight: 14 }}>
              <strong>{teamCode(p.match.home)} vs {teamCode(p.match.away)}</strong>
              <span className="dim" style={{ marginLeft: 6 }}>
                — {p.result.live_status || 'completed'}
              </span>
            </span>
          ))}
          <div className="dim" style={{ marginTop: 4, fontSize: 10 }}>
            Match ended on Cricbuzz but the winner string hasn't published yet. Auto-grades on the next live tick (≤30s).
          </div>
        </div>
      )}

      {n === 0 ? (
        <div className="small" style={{ color: 'var(--ink-3)' }}>
          No fixtures have settled with a saved prediction yet.
          As today/tomorrow's tracked matches finish, this section will populate automatically.
        </div>
      ) : (
        <>
          {/* Summary strip */}
          <div className="grid cols-3" style={{ marginBottom: 12, gap: 1, background: 'var(--line)', border: '1px solid var(--line)' }}>
            <div style={{ background: 'var(--bg)', padding: '12px 14px' }}>
              <div className="tiny">Hit rate</div>
              <div className="num" style={{ fontSize: 22, color: hitRate >= 0.65 ? 'var(--green)' : hitRate >= 0.50 ? 'var(--ink)' : 'var(--red)' }}>
                {(hitRate*100).toFixed(1)}%
              </div>
              <div className="dim small">{correct} of {n}</div>
            </div>
            <div style={{ background: 'var(--bg)', padding: '12px 14px' }}>
              <div className="tiny">Hi-conf hit rate (≥65%)</div>
              <div className="num" style={{ fontSize: 22, color: hiConfRate == null ? 'var(--ink-3)' : hiConfRate >= 0.7 ? 'var(--green)' : 'var(--ink)' }}>
                {hiConfRate == null ? '—' : (hiConfRate*100).toFixed(1) + '%'}
              </div>
              <div className="dim small">{hiConfHit} of {hiConf.length}</div>
            </div>
            <div style={{ background: 'var(--bg)', padding: '12px 14px' }}>
              <div className="tiny">Beat the book</div>
              <div className="num" style={{ fontSize: 22, color: beatBook > 0 ? 'var(--green)' : 'var(--ink)' }}>
                {beatBook}
              </div>
              <div className="dim small">model right where book was wrong</div>
            </div>
          </div>

          {/* Table */}
          <table className="t">
            <thead>
              <tr>
                <th></th>
                <th>Date</th>
                <th>Fixture</th>
                <th>Model pick</th>
                <th className="num">Confidence</th>
                <th>Book pick</th>
                <th>Actual winner</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {settled.slice(0, 12).map((p, i) => {
                const cons = p?.odds?.h2h?.consensus;
                const bookPick = cons
                  ? (cons.p_home >= 0.5 ? p.match.home : p.match.away)
                  : null;
                const ok = p.result.correct;
                return (
                  <tr key={i}>
                    <td style={{ width: 24 }}>
                      <span style={{
                        display: 'inline-block', width: 16, height: 16, lineHeight: '15px',
                        textAlign: 'center', borderRadius: '50%',
                        color: ok ? 'var(--green)' : 'var(--red)',
                        border: `1px solid ${ok ? 'var(--green)' : 'var(--red)'}`,
                        fontWeight: 700, fontSize: 11,
                      }}>{ok ? '✓' : '✗'}</span>
                    </td>
                    <td className="dim">{fmt.date(p.match.date)}</td>
                    <td>
                      <strong>{teamCode(p.match.home)}</strong>
                      <span className="dim"> vs </span>
                      <strong>{teamCode(p.match.away)}</strong>
                      <span className="dim small" style={{ marginLeft: 8 }}>{p.match.format}</span>
                    </td>
                    <td>
                      <span style={{ color: ok ? 'var(--green)' : 'var(--red)' }}>
                        {teamCode(p.prediction.favored)}
                      </span>
                    </td>
                    <td className="num">{p.prediction.favored_pct.toFixed(1)}%</td>
                    <td>
                      {bookPick
                        ? <span style={{ color: bookPick === p.result.winner ? 'var(--ink-2)' : 'var(--ink-3)' }}>{teamCode(bookPick)}</span>
                        : <span className="dim">—</span>}
                    </td>
                    <td><strong>{teamCode(p.result.winner)}</strong></td>
                    <td className="dim small">{p.result.live_status || ''}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {settled.length > 12 && (
            <div className="dim small" style={{ marginTop: 8 }}>
              Showing 12 of {settled.length} settled calls.
            </div>
          )}
        </>
      )}
    </Card>
  );
}


// =====================================================================
// VERSION HISTORY — model's call evolution as new info lands
// =====================================================================
// The phase machine emits versioned predictions:
//   pre_match_v0 (T-30m, no XI) → pre_start_v1 (T-5m, announced XI) →
//   toss_aware_v2 (after toss). Each is saved as a snapshot in the
//   prediction file's `versions[]` array. This component renders the
//   trajectory so the user can see how each new information drop moved
//   the model's call.
const VERSION_LABELS = {
  pre_match_v0:  { name: "Pre-match",   sub: "T-30m · proxy XI",         color: "var(--ink-3)" },
  pre_start_v1:  { name: "Pre-start",   sub: "T-5m · announced XI",      color: "var(--accent-2)" },
  toss_aware_v2: { name: "Post-toss",   sub: "after toss · final pre-game", color: "var(--accent)" },
  legacy:        { name: "Pre-match",   sub: "single-version snapshot",  color: "var(--ink-3)" },
};

function VersionHistory({ pred }) {
  if (!pred?.versions?.length) return null;
  const versions = pred.versions;
  const m = pred.match || {};

  return (
    <Card title="Prediction trajectory"
           right={`${versions.length} version${versions.length === 1 ? '' : 's'} · ${pred.current || 'legacy'}`}>
      <table className="t" style={{ tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th style={{ width: 130 }}>VERSION</th>
            <th>WHEN</th>
            <th className="num">P({teamCode(m.home)})</th>
            <th className="num">P({teamCode(m.away)})</th>
            <th>FAVOURED</th>
            <th className="num">EDGE</th>
            <th>Δ vs prev</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((v, i) => {
            const meta = VERSION_LABELS[v.tag] || { name: v.tag, sub: '', color: 'var(--ink)' };
            const p = v.prediction || {};
            const ph = p.p_home_wins;
            const pa = p.p_away_wins;
            const fav = p.favored;
            const mvb = v.model_vs_book || {};
            const edge = mvb.best_side_edge_pp;
            const prev = i > 0 ? (versions[i - 1].prediction?.p_home_wins) : null;
            const delta = (ph != null && prev != null) ? (ph - prev) * 100 : null;
            const isCurrent = pred.current === v.tag;
            return (
              <tr key={v.tag + i} style={{ background: isCurrent ? 'var(--bg-3)' : 'transparent' }}>
                <td>
                  <span style={{ color: meta.color, fontWeight: 600 }}>{meta.name}</span>
                  <div className="dim small">{meta.sub}</div>
                </td>
                <td className="dim small">{v.at ? new Date(v.at).toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}</td>
                <td className="num">{ph != null ? (ph * 100).toFixed(1) + '%' : '—'}</td>
                <td className="num">{pa != null ? (pa * 100).toFixed(1) + '%' : '—'}</td>
                <td>{fav ? teamCode(fav) : '—'}</td>
                <td className="num" style={{ color: edge != null ? (edge >= 5 ? 'var(--green)' : edge < 0 ? 'var(--red)' : 'var(--ink)') : 'var(--ink-3)' }}>
                  {edge != null ? (edge >= 0 ? '+' : '') + edge.toFixed(1) + 'pp' : '—'}
                </td>
                <td className="num" style={{ color: delta == null ? 'var(--ink-3)' : Math.abs(delta) < 1 ? 'var(--ink-3)' : (delta > 0 ? 'var(--accent-2)' : 'var(--accent)') }}>
                  {delta == null ? '—' : (delta >= 0 ? '+' : '') + delta.toFixed(1) + 'pp'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {versions.length === 1 && versions[0].tag === 'legacy' && (
        <div className="small dim" style={{ marginTop: 10 }}>
          This prediction was saved before the phase-versioning rework.
          Future fixtures will show the full pre-match → pre-start → post-toss
          trajectory as the orchestrator fires each phase action.
        </div>
      )}
    </Card>
  );
}


// =====================================================================
// EDGE TRAJECTORY — sparkline of model-vs-book edge across versions
// =====================================================================
// Visual companion to the table above. Plots the favoured-side edge
// (in pp vs market consensus) at each prediction version, with the
// final dot highlighted. A flat line means no information moved the
// model's view; a steep climb means each new piece (XI, toss) helped
// the model find more value than the books had.
function EdgeTrajectory({ pred }) {
  if (!pred?.versions?.length) return null;
  const versions = pred.versions.filter(v => (v.model_vs_book?.best_side_edge_pp) != null);
  if (!versions.length) {
    return (
      <Card title="Edge trajectory" right="model vs market">
        <div className="small dim" style={{ padding: 8 }}>
          No bookmaker odds attached to any version yet. Edge appears
          once The Odds API has a quote for the fixture.
        </div>
      </Card>
    );
  }

  const points = versions.map((v, i) => ({
    tag:  v.tag,
    edge: v.model_vs_book.best_side_edge_pp,
    side: v.model_vs_book.best_side,
  }));
  const W = 360, H = 110, P = 18;
  const minE = Math.min(0, ...points.map(p => p.edge));
  const maxE = Math.max(0, ...points.map(p => p.edge), 5);
  const yScale = (e) => P + (H - 2 * P) * (1 - (e - minE) / (maxE - minE || 1));
  const xScale = (i) => points.length === 1 ? W / 2 : P + i * (W - 2 * P) / (points.length - 1);
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(p.edge).toFixed(1)}`).join(' ');
  const zeroY = yScale(0);
  const last = points[points.length - 1];

  return (
    <Card title="Edge trajectory" right={`${last.side ? teamCode(last.side) : ''} ${last.edge >= 0 ? '+' : ''}${last.edge.toFixed(1)}pp`}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {/* zero-edge baseline */}
        <line x1={P} x2={W - P} y1={zeroY} y2={zeroY}
               stroke="var(--ink-4)" strokeDasharray="3 3" strokeWidth="1" />
        <text x={P} y={zeroY - 4} className="mono" fontSize="9" fill="var(--ink-3)">edge = 0</text>
        {/* trajectory */}
        <path d={pathD} fill="none" stroke="var(--accent-2)" strokeWidth="2" />
        {/* points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={xScale(i)} cy={yScale(p.edge)} r={i === points.length - 1 ? 5 : 3.5}
                     fill={i === points.length - 1 ? 'var(--accent)' : 'var(--accent-2)'} />
            <text x={xScale(i)} y={yScale(p.edge) - 9}
                   textAnchor="middle" className="mono" fontSize="10"
                   fill={i === points.length - 1 ? 'var(--accent)' : 'var(--ink-2)'}>
              {p.edge >= 0 ? '+' : ''}{p.edge.toFixed(1)}
            </text>
            <text x={xScale(i)} y={H - 4}
                   textAnchor="middle" className="mono" fontSize="9" fill="var(--ink-3)">
              {(VERSION_LABELS[p.tag]?.name || p.tag).split(' ')[0].toLowerCase()}
            </text>
          </g>
        ))}
      </svg>
      <div className="small dim" style={{ marginTop: 4 }}>
        Each point = a phase prediction. Climb means new info (XI, toss)
        widened the gap vs market; descent means the books caught up.
      </div>
    </Card>
  );
}


// =====================================================================
// MATCH TIMELINE — orchestrator phase events feed
// =====================================================================
// Renders the last N phase transitions and timed-action firings across
// all tracked fixtures. Mirrors what's in match_timeline.jsonl, with
// home/away enriched server-side. Useful as a "what's the orchestrator
// doing right now" pulse, especially around toss / kickoff.
const PHASE_COLOR = {
  DISCOVERED: 'var(--ink-3)',
  SCHEDULED:  'var(--accent-2)',
  PRE_START:  'var(--accent)',
  LIVE:       'var(--green)',
  COMPLETE:   'var(--ink-2)',
  REVIEWED:   'var(--book)',
  ABANDONED:  'var(--red)',
};

function MatchTimeline({ events }) {
  if (!events?.length) {
    return (
      <Card title="Phase timeline" right="orchestrator activity">
        <div className="small dim">
          No events yet — the timeline populates once the phase machine fires
          its first transition.
        </div>
      </Card>
    );
  }

  // Newest first
  const ordered = [...events].reverse().slice(0, 40);

  return (
    <Card title="Phase timeline"
           right={`${events.length} events · newest first`}>
      <table className="t" style={{ tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th style={{ width: 90 }}>WHEN</th>
            <th>FIXTURE</th>
            <th>EVENT</th>
            <th>DETAIL</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((e, i) => {
            const when = e.at ? new Date(e.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
            const home = e.home ? teamCode(e.home) : '—';
            const away = e.away ? teamCode(e.away) : '—';
            let label = '', detail = '', color = 'var(--ink)';
            if (e.event === 'transition') {
              label = `${e.from || 'NEW'} → ${e.to}`;
              color = PHASE_COLOR[e.to] || 'var(--ink)';
              detail = '';
            } else if (e.event === 'action') {
              label = e.action;
              color = e.ok ? 'var(--green)' : 'var(--red)';
              detail = e.ok ? 'fired' : 'failed';
            } else if (e.event === 'toss') {
              label = 'TOSS';
              color = 'var(--accent)';
              detail = `${teamCode(e.toss_winner)} chose to ${e.toss_decision}`;
            } else if (e.event === 'rescheduled') {
              label = 'RESCHEDULED';
              color = 'var(--accent)';
              const oldS = e.old_ts ? new Date(e.old_ts * 1000).toLocaleString() : '—';
              const newS = e.new_ts ? new Date(e.new_ts * 1000).toLocaleString() : '—';
              detail = `${oldS} → ${newS}`;
            }
            return (
              <tr key={i}>
                <td className="dim small mono">{when}</td>
                <td><strong>{home}</strong><span className="dim"> vs </span><strong>{away}</strong></td>
                <td><span style={{ color, fontWeight: 600 }}>{label}</span></td>
                <td className="dim small">{detail}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}


// =====================================================================
// LEARNINGS — per-version error attribution from completed matches
// =====================================================================
// Sources `learnings/post_match_log.jsonl` (written by
// post_match_review.review_one() when phase_loop fires COMPLETE.review).
// Each entry shows what each phase prediction added (or didn't) and the
// primary error factor when the final call missed.
function Learnings({ entries }) {
  if (!entries?.length) {
    return (
      <Card title="Recent learnings" right="post-match attribution">
        <div className="small dim">
          The learning ledger populates as the phase machine fires
          <code style={{ padding: '0 4px' }}>COMPLETE.review</code> actions
          on settled matches. Each entry is a per-version error breakdown.
        </div>
      </Card>
    );
  }
  return (
    <Card title="Recent learnings" right={`${entries.length} recent · newest first`}>
      <div className="stack" style={{ gap: 10 }}>
        {entries.slice(0, 6).map((e, i) => {
          const m = e.match || {};
          const a = e.attribution || {};
          const final_correct = a.final_correct;
          const verdict = final_correct ? 'WON' : 'LOST';
          const verdictColor = final_correct ? 'var(--green)' : 'var(--red)';
          const winner = (e.actual || {}).winner || '—';
          return (
            <div key={i} style={{
              borderTop: i === 0 ? 'none' : '1px dashed var(--line)',
              paddingTop: i === 0 ? 0 : 8,
            }}>
              <div className="spread" style={{ alignItems: 'baseline' }}>
                <div>
                  <strong>{teamCode(m.home)}</strong>
                  <span className="dim"> vs </span>
                  <strong>{teamCode(m.away)}</strong>
                  <span className="dim small" style={{ marginLeft: 8 }}>
                    {fmt.date(m.date)} · {m.format}
                  </span>
                </div>
                <span className="chip" style={{ color: verdictColor, borderColor: `color-mix(in oklch, ${verdictColor} 50%, var(--line))` }}>
                  {verdict}
                </span>
              </div>
              <div className="small dim" style={{ marginTop: 2 }}>
                Actual: <strong style={{ color: 'var(--ink-2)' }}>{teamCode(winner)}</strong>
                {(e.actual || {}).live_status ? ` · ${(e.actual || {}).live_status}` : ''}
              </div>
              {/* Per-version line: which phase predictions got it right? */}
              {(e.versions || []).length > 1 && (
                <div className="small" style={{ marginTop: 6, color: 'var(--ink-2)' }}>
                  {(e.versions || []).map((v, j) => (
                    <span key={j} style={{ marginRight: 12 }}>
                      <span className="dim">{(VERSION_LABELS[v.tag]?.name || v.tag).split(' ')[0]}:</span>{' '}
                      <span style={{ color: v.correct ? 'var(--green)' : 'var(--red)' }}>
                        {teamCode(v.predicted_winner)}
                      </span>
                    </span>
                  ))}
                </div>
              )}
              {/* Attribution headline */}
              {a.primary_error_factor && !final_correct && (
                <div className="small" style={{ marginTop: 4, color: 'var(--accent)' }}>
                  Primary error: <strong>{a.primary_error_factor}</strong>
                  {a.luck_signal && ' · narrow margin (some luck)'}
                </div>
              )}
              {a.correct_versions?.length > 0 && !final_correct && (
                <div className="small" style={{ marginTop: 2, color: 'var(--ink-3)' }}>
                  Earlier versions got it right: {a.correct_versions.join(', ')} — info added later moved the call wrong.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}


function RecentTable({ rows }) {
  return (
    <table className="t">
      <thead><tr>
        <th>DATE</th><th>FMT</th><th>FIXTURE</th><th>WINNER</th><th>MARGIN</th>
      </tr></thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.match_id}>
            <td className="num dim">{fmt.date(r.date)}</td>
            <td><span className="chip">{r.format}</span></td>
            <td>{teamCode(r.home)} <span style={{ color: 'var(--ink-4)' }}>vs</span> {teamCode(r.away)}
              <div className="small" style={{ color: 'var(--ink-3)' }}>{fmt.short(r.competition, 36)}</div>
            </td>
            <td style={{ color: 'var(--accent)' }}>{teamCode(r.winner)}</td>
            <td className="dim">{r.margin}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BetLedger({ bets }) {
  if (!bets) return <div className="small">No bet data.</div>;
  const rows = bets.recent_bets || [];
  return (
    <div>
      <div className="grid cols-3" style={{ gap: 1, background: 'var(--line)', border: '1px solid var(--line)' }}>
        <BigCell k="STAKED" v={`₹${(bets.staked || 0).toFixed(0)}`} />
        <BigCell k="PNL" v={`${bets.pnl >= 0 ? '+' : ''}₹${(bets.pnl || 0).toFixed(0)}`} accent={bets.pnl >= 0 ? 'var(--green)' : 'var(--red)'} />
        <BigCell k="OPEN" v={(bets.pending || 0) + (bets.open_tickets?.length || 0)} accent="var(--accent)" />
      </div>
      <hr className="dash" />
      <table className="t">
        <thead><tr><th>SEL</th><th className="num">ODDS</th><th className="num">STAKE</th><th className="num">EDGE</th><th>STATUS</th></tr></thead>
        <tbody>
          {rows.map(b => (
            <tr key={b.bet_id}>
              <td>{teamCode(b.selection)}<div className="small" style={{ color: 'var(--ink-3)' }}>{b.market} · {(b.placed_at || '').slice(11,16)}</div></td>
              <td className="num">{fmt.odds(b.decimal_odds)}</td>
              <td className="num">₹{b.stake.toFixed(0)}</td>
              <td className="num" style={{ color: b.edge_pct >= 5 ? 'var(--green)' : 'var(--ink-2)' }}>+{b.edge_pct.toFixed(1)}pp</td>
              <td>
                <span className={`chip ${b.status.includes('void') ? 'red' : b.status.includes('pending') ? 'amber' : 'green'}`}>
                  {b.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="small" style={{ marginTop: 10, color: 'var(--ink-3)' }}>
        Mode: {bets.mode} · stake-cap voids any bet exceeding the half-Kelly fraction.
      </div>
    </div>
  );
}

function DataSourcesStrip({ stats }) {
  const sources = [
    'CricSheet',
    'Cricinfo Statsguru',
    'ICC Rankings',
    'Cricbuzz Live',
    'Wikidata',
    'Nominatim',
    'Visual Crossing',
    'OpenWeatherMap',
    'GDELT',
    'NewsAPI',
    'CricAPI',
    'Reddit RSS',
  ];
  return (
    <div className="card">
      <Crosshairs />
      <div className="hd">
        <div className="t">17 INGESTERS · 14 TABLES · 15 DERIVED VIEWS · 3 MODELS</div>
        <div className="r">DUCKDB · {(stats?.balls / 1e6).toFixed(2)}M BALLS</div>
      </div>
      <div className="bd">
        <div className="row wrap" style={{ gap: 6 }}>
          {sources.map(s => <span key={s} className="chip ghost">{s}</span>)}
        </div>
        <div className="small" style={{ marginTop: 12, color: 'var(--ink-3)', maxWidth: 86+'ch' }}>
          Pipeline: ingest → DuckDB → time-aware feature views (no temporal leakage) → LightGBM(num+cat) ⊕ XGBoost ⊕ CatBoost ⊕ LR → stacked logistic
          regression → isotonic calibration → 5,000-sim Monte Carlo innings rollout → ensemble blender (60% match-model · 25% form prior · 15% H2H prior).
          Sequence Transformer (last 12 balls) trains on CUDA — ~30× faster than CPU.
        </div>
      </div>
    </div>
  );
}

function Footer({ data }) {
  return (
    <div className="footnote">
      <span>WAUNDERING · CRICKET PREDICTION TERMINAL · v0.4</span>
      <span>GENERATED {new Date(data.generated_at).toISOString().slice(0, 16).replace('T', ' ')} UTC · NEXT REFRESH +60M</span>
      <span>WITHOUT BALL-TRACKING · WICKET AUC ≈ 0.57 · CEILING ACKNOWLEDGED</span>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
