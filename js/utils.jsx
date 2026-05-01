/* global React */

// ============================================================
// helpers
// ============================================================
const fmt = {
  pct: (v, d=1) => (v == null ? '—' : (v * 100).toFixed(d) + '%'),
  pp:  (v, d=1) => (v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(d) + 'pp'),
  num: (v, d=0) => (v == null || isNaN(v) ? '—' : Number(v).toFixed(d)),
  odds: (v) => v == null ? '—' : v.toFixed(2),
  date: (s) => {
    if (!s) return '—';
    const d = new Date(s);
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }).toUpperCase();
  },
  time: (s) => {
    if (!s) return '—';
    const d = new Date(s);
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  },
  team: (s) => s ? s.toUpperCase() : '—',
  short: (s, n=18) => !s ? '' : (s.length > n ? s.slice(0, n-1) + '…' : s),
};

const teamCode = (name) => {
  const m = {
    'Lucknow Super Giants': 'LSG',
    'Kolkata Knight Riders': 'KKR',
    'Delhi Capitals': 'DC',
    'Royal Challengers Bengaluru': 'RCB',
    'Chennai Super Kings': 'CSK',
    'Gujarat Titans': 'GT',
    'Rajasthan Royals': 'RR',
    'Sunrisers Hyderabad': 'SRH',
    'Mumbai Indians': 'MI',
    'Punjab Kings': 'PBKS',
    'New Zealand': 'NZ',
    'Bangladesh': 'BAN',
    'India': 'IND',
    'Australia': 'AUS',
    'England': 'ENG',
    'South Africa': 'SA',
    'Pakistan': 'PAK',
    'Sri Lanka': 'SL',
    'West Indies': 'WI',
    'Afghanistan': 'AFG',
    'Ireland': 'IRE',
    'Netherlands': 'NED',
    'Scotland': 'SCO',
    'Nepal': 'NEP',
    'Uganda': 'UGA',
    'United Arab Emirates': 'UAE',
    'United States of America': 'USA',
  };
  return m[name] || (name ? name.split(' ').map(w => w[0]).slice(0,4).join('').toUpperCase() : '—');
};

const teamHue = (name) => {
  // colours kept restrained — just used as accents on the team-row strips
  const m = {
    'Lucknow Super Giants': '#a3c8ff',
    'Kolkata Knight Riders': '#b39bff',
    'Delhi Capitals': '#7ec8ff',
    'Royal Challengers Bengaluru': '#ff8a8a',
    'Chennai Super Kings': '#ffd166',
    'Gujarat Titans': '#7eb8d6',
    'Rajasthan Royals': '#ff9bd0',
    'Sunrisers Hyderabad': '#ff9a4a',
    'Mumbai Indians': '#7eaaff',
    'Punjab Kings': '#ff7a7a',
    'New Zealand': '#3a3a3a',
    'Bangladesh': '#4ea567',
    'India': '#4ad6ff',
  };
  return m[name] || '#9aa3b2';
};

// ============================================================
// SVG primitives
// ============================================================
const Crosshairs = () => (
  React.createElement(React.Fragment, null,
    React.createElement('span', { className: 'crossh tl' }),
    React.createElement('span', { className: 'crossh tr' }),
    React.createElement('span', { className: 'crossh bl' }),
    React.createElement('span', { className: 'crossh br' }),
  )
);

const Card = ({ title, right, children, className='', pad=true }) =>
  React.createElement('div', { className: `card ${className}` },
    React.createElement(Crosshairs),
    title && React.createElement('div', { className: 'hd' },
      React.createElement('div', { className: 't' }, title),
      right && React.createElement('div', { className: 'r' }, right),
    ),
    React.createElement('div', { className: pad ? 'bd' : '' }, children),
  );

window.fmt = fmt;
window.teamCode = teamCode;
window.teamHue = teamHue;
window.Crosshairs = Crosshairs;
window.Card = Card;
