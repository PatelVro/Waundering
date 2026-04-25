import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

CRICSHEET_ZIPS = {
    # international
    "all_json":                       "https://cricsheet.org/downloads/all_json.zip",
    "tests_json":                     "https://cricsheet.org/downloads/tests_json.zip",
    "odis_json":                      "https://cricsheet.org/downloads/odis_json.zip",
    "t20s_json":                      "https://cricsheet.org/downloads/t20s_json.zip",

    # men's franchise / domestic T20
    "ipl_json":                       "https://cricsheet.org/downloads/ipl_json.zip",
    "bbl_json":                       "https://cricsheet.org/downloads/bbl_json.zip",
    "psl_json":                       "https://cricsheet.org/downloads/psl_json.zip",
    "cpl_json":                       "https://cricsheet.org/downloads/cpl_json.zip",
    "lpl_json":                       "https://cricsheet.org/downloads/lpl_json.zip",
    "bpl_json":                       "https://cricsheet.org/downloads/bpl_json.zip",
    "sa20_json":                      "https://cricsheet.org/downloads/sa20_json.zip",
    "ilt20_json":                     "https://cricsheet.org/downloads/ilt20_json.zip",
    "mlc_json":                       "https://cricsheet.org/downloads/mlc_json.zip",
    "the_hundred":                    "https://cricsheet.org/downloads/the_hundred_json.zip",
    "the_hundred_men":                "https://cricsheet.org/downloads/the_hundred_men_json.zip",
    "vitality_blast":                 "https://cricsheet.org/downloads/t20_blast_json.zip",
    "super_smash":                    "https://cricsheet.org/downloads/ssm_json.zip",
    "syed_mushtaq_ali":               "https://cricsheet.org/downloads/smat_json.zip",

    # men's domestic first-class / list A
    "county_championship":            "https://cricsheet.org/downloads/county_json.zip",
    "ranji_trophy":                   "https://cricsheet.org/downloads/ranji_json.zip",
    "sheffield_shield":               "https://cricsheet.org/downloads/sheffield_json.zip",
    "royal_one_day_cup":              "https://cricsheet.org/downloads/rlodc_json.zip",

    # women's
    "women_t20s":                     "https://cricsheet.org/downloads/wt20s_json.zip",
    "women_odis":                     "https://cricsheet.org/downloads/wodis_json.zip",
    "women_tests":                    "https://cricsheet.org/downloads/wtests_json.zip",
    "wpl_json":                       "https://cricsheet.org/downloads/wpl_json.zip",
    "wbbl_json":                      "https://cricsheet.org/downloads/wbb_json.zip",
    "the_hundred_women":              "https://cricsheet.org/downloads/the_hundred_women_json.zip",
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

CRICINFO_PROFILE_BASE = "https://www.espncricinfo.com/cricketers"
CRICINFO_SLEEP_SECONDS = float(os.environ.get("CRICINFO_SLEEP", "2.5"))

for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
