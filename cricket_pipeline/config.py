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

CRICSHEET_PEOPLE_CSV = "https://cricsheet.org/register/people.csv"

NOMINATIM_BASE = "https://nominatim.openstreetmap.org/search"
NOMINATIM_SLEEP_SECONDS = float(os.environ.get("NOMINATIM_SLEEP", "1.1"))

ICC_RANKINGS_BASE = "https://www.icc-cricket.com/rankings/mens/player-rankings"
ICC_TEAM_RANKINGS_BASE = "https://www.icc-cricket.com/rankings/mens/team-rankings"
ICC_SLEEP_SECONDS = float(os.environ.get("ICC_SLEEP", "2.0"))

OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY", "")
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"

NEWS_FEEDS = {
    "espncricinfo":   "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
    "cricbuzz":       "https://www.cricbuzz.com/rss/cricket-news.rss",
    "icc":            "https://www.icc-cricket.com/rss/news",
    "wisden":         "https://wisden.com/feed",
}

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1"

CRICAPI_KEY = os.environ.get("CRICAPI_KEY", "")
CRICAPI_BASE = "https://api.cricapi.com/v1"

for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
