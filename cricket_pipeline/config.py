import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

CRICSHEET_ZIPS = {
    "all_json":       "https://cricsheet.org/downloads/all_json.zip",
    "tests_json":     "https://cricsheet.org/downloads/tests_json.zip",
    "odis_json":      "https://cricsheet.org/downloads/odis_json.zip",
    "t20s_json":      "https://cricsheet.org/downloads/t20s_json.zip",
    "ipl_json":       "https://cricsheet.org/downloads/ipl_json.zip",
    "bbl_json":       "https://cricsheet.org/downloads/bbl_json.zip",
    "psl_json":       "https://cricsheet.org/downloads/psl_json.zip",
    "the_hundred":    "https://cricsheet.org/downloads/the_hundred_json.zip",
}

STATSGURU_BASE = "https://stats.espncricinfo.com/ci/engine/stats/index.html"
STATSGURU_USER_AGENT = (
    "Mozilla/5.0 (CricketPipeline research bot; "
    "contact: set via STATSGURU_CONTACT env var)"
)
STATSGURU_SLEEP_SECONDS = float(os.environ.get("STATSGURU_SLEEP", "2.5"))

VISUAL_CROSSING_KEY = os.environ.get("VISUAL_CROSSING_KEY", "")
VISUAL_CROSSING_BASE = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/"
    "rest/services/timeline"
)

for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
