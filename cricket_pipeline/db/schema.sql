-- Cricket prediction pipeline — DuckDB schema
-- One row per ball is the source of truth. Everything else is derived.

CREATE TABLE IF NOT EXISTS matches (
    match_id           VARCHAR PRIMARY KEY,
    format             VARCHAR,                -- Test, ODI, T20, IT20, etc.
    competition        VARCHAR,                -- IPL, BBL, World Cup, bilateral
    season             VARCHAR,
    start_date         DATE,
    end_date           DATE,
    venue              VARCHAR,
    city               VARCHAR,
    country            VARCHAR,
    team_home          VARCHAR,
    team_away          VARCHAR,
    toss_winner        VARCHAR,
    toss_decision      VARCHAR,                -- bat / field
    winner             VARCHAR,
    win_margin_runs    INTEGER,
    win_margin_wickets INTEGER,
    player_of_match    VARCHAR,
    umpires            VARCHAR,
    source             VARCHAR DEFAULT 'cricsheet'
);

CREATE TABLE IF NOT EXISTS innings (
    match_id     VARCHAR,
    innings_no   INTEGER,
    batting_team VARCHAR,
    bowling_team VARCHAR,
    total_runs   INTEGER,
    total_wkts   INTEGER,
    total_overs  DOUBLE,
    target       INTEGER,
    PRIMARY KEY (match_id, innings_no)
);

CREATE TABLE IF NOT EXISTS balls (
    match_id         VARCHAR,
    innings_no       INTEGER,
    over_no          INTEGER,                  -- 0-indexed
    ball_in_over     INTEGER,                  -- 1..6 (extras may push higher)
    legal_ball_no    INTEGER,                  -- cumulative legal balls in innings
    batting_team     VARCHAR,
    bowling_team     VARCHAR,
    batter           VARCHAR,
    non_striker      VARCHAR,
    bowler           VARCHAR,
    runs_batter      INTEGER,
    runs_extras      INTEGER,
    runs_total       INTEGER,
    extras_type      VARCHAR,                  -- wides, noballs, byes, legbyes, penalty
    is_wicket        BOOLEAN,
    wicket_kind      VARCHAR,                  -- bowled, caught, lbw, runout, stumped...
    player_out       VARCHAR,
    fielders         VARCHAR,
    PRIMARY KEY (match_id, innings_no, over_no, ball_in_over)
);

CREATE TABLE IF NOT EXISTS players (
    player_id     VARCHAR PRIMARY KEY,         -- CricSheet people.csv registry id
    name          VARCHAR,
    unique_name   VARCHAR,
    country       VARCHAR,
    dob           DATE,
    role          VARCHAR,
    batting_hand  VARCHAR,
    bowling_type  VARCHAR,
    key_cricinfo  VARCHAR,
    key_cricbuzz  VARCHAR,
    key_bcci      VARCHAR,
    key_opta      VARCHAR,
    key_nvplay    VARCHAR,
    key_pulse     VARCHAR,
    profile_url   VARCHAR,
    enriched_at   TIMESTAMP
);

-- Aggregated player career splits from Statsguru (optional enrichment)
CREATE TABLE IF NOT EXISTS player_splits (
    player_name  VARCHAR,
    format       VARCHAR,
    split_type   VARCHAR,                      -- overall / vs_opposition / at_venue / by_year
    split_key    VARCHAR,                      -- e.g. "India", "MCG", "2024"
    matches      INTEGER,
    innings      INTEGER,
    runs         INTEGER,
    balls        INTEGER,
    avg          DOUBLE,
    sr           DOUBLE,
    hs           INTEGER,
    hundreds     INTEGER,
    fifties      INTEGER,
    wickets      INTEGER,
    bbi          VARCHAR,
    econ         DOUBLE,
    bowl_avg     DOUBLE,
    bowl_sr      DOUBLE,
    fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_name, format, split_type, split_key)
);

CREATE TABLE IF NOT EXISTS venues (
    venue       VARCHAR PRIMARY KEY,
    city        VARCHAR,
    country     VARCHAR,
    lat         DOUBLE,
    lon         DOUBLE,
    boundary_m  DOUBLE,                        -- nominal straight boundary
    capacity    INTEGER,
    ends        VARCHAR,                       -- end names ("Pavilion End / Members End")
    established INTEGER,
    notes       VARCHAR
);

CREATE TABLE IF NOT EXISTS weather_daily (
    venue      VARCHAR,
    date       DATE,
    temp_c     DOUBLE,
    humidity   DOUBLE,
    dew_point  DOUBLE,
    wind_kmh   DOUBLE,
    cloud_pct  DOUBLE,
    precip_mm  DOUBLE,
    source     VARCHAR,
    PRIMARY KEY (venue, date)
);

CREATE TABLE IF NOT EXISTS rankings (
    snapshot_date DATE,
    format        VARCHAR,              -- test / odi / t20i
    category      VARCHAR,              -- batting / bowling / allrounder / team
    rank          INTEGER,
    name          VARCHAR,
    country       VARCHAR,
    rating        INTEGER,
    source        VARCHAR DEFAULT 'icc',
    PRIMARY KEY (snapshot_date, format, category, rank)
);

CREATE TABLE IF NOT EXISTS news (
    url            VARCHAR PRIMARY KEY,
    published_at   TIMESTAMP,
    source         VARCHAR,
    title          VARCHAR,
    summary        VARCHAR,
    entities       VARCHAR,             -- JSON array of detected teams/players
    sentiment      DOUBLE,              -- VADER compound score [-1, 1]
    fetched_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id    VARCHAR PRIMARY KEY,
    format        VARCHAR,
    competition   VARCHAR,
    start_date    DATE,
    venue         VARCHAR,
    city          VARCHAR,
    country       VARCHAR,
    team_home     VARCHAR,
    team_away     VARCHAR,
    status        VARCHAR,              -- upcoming, live, completed
    source        VARCHAR,
    fetched_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS umpires (
    name        VARCHAR PRIMARY KEY,
    matches     INTEGER,                -- counted from matches table
    formats     VARCHAR                  -- comma-joined formats officiated
);

CREATE TABLE IF NOT EXISTS match_officials (
    match_id    VARCHAR,
    role        VARCHAR,                 -- umpire / tv_umpire / reserve_umpire / match_referee
    name        VARCHAR,
    PRIMARY KEY (match_id, role, name)
);

-- Indexes to make the common filters cheap
CREATE INDEX IF NOT EXISTS idx_balls_match     ON balls(match_id);
CREATE INDEX IF NOT EXISTS idx_balls_batter    ON balls(batter);
CREATE INDEX IF NOT EXISTS idx_balls_bowler    ON balls(bowler);
CREATE INDEX IF NOT EXISTS idx_matches_format  ON matches(format);
CREATE INDEX IF NOT EXISTS idx_matches_venue   ON matches(venue);
CREATE INDEX IF NOT EXISTS idx_matches_date    ON matches(start_date);
