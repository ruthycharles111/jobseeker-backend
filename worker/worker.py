"""Resilient PocketBase worker for JobSeeker AI backend.

Responsibilities:
  * Legacy chat replies (chat_messages collection)
  * Coach replies fallback (coach_messages, only if frontend did not respond)
  * Job aggregation from 25 sources with per-source health tracking
  * Tier limits: free = 20 jobs / 4 searches per day; contributor = 78 / unlimited
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import quote, urlencode

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
from openai import OpenAI
from serpapi import GoogleSearch


# ─── Config ──────────────────────────────────────────────────────────────────
POCKETBASE_URL = os.getenv("POCKETBASE_URL", "http://localhost:8090").rstrip("/")
POCKETBASE_ADMIN_TOKEN = os.getenv("PB_ADMIN_TOKEN") or os.getenv("POCKETBASE_ADMIN_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
FINDWORK_KEY = os.getenv("FINDWORK_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
JSEARCH_HOST = "jsearch.p.rapidapi.com"

MISTRAL_MODELS = [
    m.strip()
    for m in os.getenv(
        "MISTRAL_MODELS",
        "mistral-tiny,open-mistral-nemo,ministral-8b-latest,ministral-3b-latest",
    ).split(",")
    if m.strip()
]

FREE_TIER_JOBS_PER_DAY = int(os.getenv("FREE_TIER_JOBS_PER_DAY", "20"))
CONTRIBUTOR_TIER_JOBS_PER_DAY = int(os.getenv("CONTRIBUTOR_TIER_JOBS_PER_DAY", "78"))
FREE_TIER_SEARCHES_PER_DAY = int(os.getenv("FREE_TIER_SEARCHES_PER_DAY", "4"))

HTTP_TIMEOUT = (5, 20)
DEFAULT_HEADERS = {"User-Agent": "JobSeekerAI/1.0 (+https://jobseekerai.name.ng)"}

CHAT_SYSTEM_PROMPT = (
    "You are JobSeeker AI Coach, a friendly and knowledgeable career advisor. "
    "Always respond directly to the user's latest message. "
    "Be natural, warm, and ask follow-up questions when appropriate. "
    "Remember previous conversation history. "
    "Never output generic templates, welcome messages, structured plans, or numbered lists "
    "unless the user explicitly asks for a plan."
)

# Fallback prompts if PB fetch of coach_prompts fails
COACH_SYSTEM_PROMPTS = {
    "interview": (
        "You are JobSeeker AI Interview Coach. You act as both the interviewer and the "
        "coach. Ask role-specific questions one at a time; after the mock interview ends, "
        "give detailed feedback on each answer, show what a winning answer looks like, and "
        "give a 24-48h practice plan. Never output generic templates."
    ),
    "business": (
        "You are JobSeeker AI Business Coach for solopreneurs. Help the user brainstorm "
        "business ideas, land clients, write contracts, price services, and scale. Explain "
        "everything in plain language. Always end with one concrete next step."
    ),
    "social_media": (
        "You are JobSeeker AI Social Media Growth Coach, outperforming vidIQ and generic "
        "growth tools. You master YouTube SEO, TikTok algorithm, Instagram reach, LinkedIn "
        "recruiter signals, viral hooks, and monetization. Every answer is tactical, "
        "specific to the user's niche, and includes at least one hook or format."
    ),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
)
log = logging.getLogger("jobseeker-worker")
ai = OpenAI(api_key=MISTRAL_API_KEY, base_url="https://api.mistral.ai/v1") if MISTRAL_API_KEY else None
claim_lock = threading.Lock()
claimed_messages: dict[str, float] = {}
claimed_coach_messages: dict[str, float] = {}
source_health: dict[str, dict[str, Any]] = {}


# ─── HTTP / PB helpers ───────────────────────────────────────────────────────
def _request(method: str, url: str, **kwargs: Any):
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.setdefault("User-Agent", DEFAULT_HEADERS["User-Agent"])
    kwargs["headers"] = headers
    for attempt in range(3):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
            return response
        except requests.RequestException as exc:
            if attempt == 2:
                log.warning("HTTP %s %s failed: %s", method, url, exc)
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def pb(method: str, path: str, json_data=None):
    if not POCKETBASE_ADMIN_TOKEN:
        log.error("PB_ADMIN_TOKEN is not configured")
        return None
    url = f"{POCKETBASE_URL}/api/{path.lstrip('/')}"
    return _request(
        method, url,
        headers={"Authorization": f"Bearer {POCKETBASE_ADMIN_TOKEN}"},
        json=json_data,
    )


def _json(response) -> dict:
    if response is None or response.status_code >= 400:
        return {}
    try:
        value = response.json()
        return value if isinstance(value, dict) else {}
    except (ValueError, requests.RequestException):
        return {}


# ─── AI with model fallback chain ────────────────────────────────────────────
def _ai_chat(messages: list, temperature: float, max_tokens: int) -> str:
    if ai is None:
        raise RuntimeError("MISTRAL_API_KEY is not configured")
    last_error = None
    for model in MISTRAL_MODELS:
        for attempt in range(2):
            try:
                response = ai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = response.choices[0].message.content if response.choices else ""
                if not answer:
                    raise RuntimeError("Mistral returned an empty response")
                return answer.strip()
            except Exception as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                log.warning("model=%s attempt=%d status=%s err=%s", model, attempt, status, exc)
                if status in (401, 402):
                    raise RuntimeError(f"Mistral auth/quota error on key: {exc}")
                time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Mistral failed across all models: {last_error}")


# ─── Text / query helpers ────────────────────────────────────────────────────
def strip_html(text) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(text, "html.parser").get_text(" ")).strip()


def parse_natural_query(raw_query: str) -> dict:
    prompt = f"""Extract job search parameters from this query. Return only valid JSON with:
title, location, remote, company, additional_filters. Use null when absent.
Query: {raw_query}"""
    try:
        content = _ai_chat([{"role": "user", "content": prompt}], 0.1, 200)
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        value = json.loads(content)
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        log.warning("Natural query parsing failed; using raw query: %s", exc)
        return {"title": raw_query, "location": None, "remote": False, "company": None}


def generate_job_titles(raw_query: str, limit: int = 5) -> list:
    prompt = (
        f"Return only a JSON array of up to {limit} concise job-title searches "
        f"for this request: {raw_query}"
    )
    try:
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "",
                         _ai_chat([{"role": "user", "content": prompt}], 0.2, 160))
        titles = json.loads(content)
        if isinstance(titles, list):
            cleaned = [str(t).strip() for t in titles if str(t).strip()]
            if cleaned:
                return cleaned[:limit]
    except Exception as exc:
        log.warning("Job-title generation failed; using raw query: %s", exc)
    return [raw_query.strip()]


def _job(title, company="", description="", location="", remote=False,
         link="", posted="", source_url="") -> dict:
    url = str(link or source_url or "")
    return {
        "title": str(title or "").strip(),
        "company": str(company or "").strip(),
        "description": strip_html(str(description or "")),
        "location": str(location or "").strip(),
        "remote": bool(remote),
        "application_link": url,
        "source_url": str(source_url or url),
        "posted_date": str(posted or ""),
        "match_score": 0,
    }


# ─── Tier / usage helpers ────────────────────────────────────────────────────
def _today_utc() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _parse_pb_date(s: str) -> float:
    if not s:
        return 0.0
    try:
        return datetime.strptime(s.replace("Z", ""), "%Y-%m-%d %H:%M:%S.%f").replace(
            tzinfo=timezone.utc).timestamp()
    except Exception:
        try:
            return datetime.strptime(s.replace("Z", ""), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp()
        except Exception:
            return 0.0


def get_user_tier(user_id: str) -> str:
    try:
        user = _json(pb("GET", f"/collections/users/records/{quote(user_id)}"))
        tier = user.get("tier") or "free"
        expires = user.get("tier_expires_at")
        if tier == "contributor" and expires:
            exp_ts = _parse_pb_date(expires)
            if exp_ts and exp_ts < time.time():
                return "free"
        return tier
    except Exception:
        return "free"


def get_or_create_daily_usage(user_id: str) -> dict:
    today = _today_utc()
    try:
        data = _json(pb(
            "GET",
            f"/collections/daily_usage/records?filter=(user='{quote(user_id)}' && date='{today}')&perPage=1",
        ))
        if data.get("items"):
            return data["items"][0]
        created = _json(pb("POST", "/collections/daily_usage/records", {
            "user": user_id, "date": today, "jobs_shown": 0, "job_searches": 0,
        }))
        return created
    except Exception as exc:
        log.warning("get_or_create_daily_usage failed user=%s: %s", user_id, exc)
        return {}


def increment_usage(user_id: str, field: str, amount: int = 1) -> None:
    usage = get_or_create_daily_usage(user_id)
    if not usage.get("id"):
        return
    current = usage.get(field, 0) or 0
    pb("PATCH", f"/collections/daily_usage/records/{usage['id']}", {field: current + amount})


def get_active_coach_prompt(mode: str) -> str:
    try:
        data = _json(pb(
            "GET",
            f"/collections/coach_prompts/records?filter=(mode='{mode}' && active=true)&sort=-version&perPage=1",
        ))
        if data.get("items"):
            txt = data["items"][0].get("prompt_text") or ""
            if txt.strip():
                return txt
    except Exception:
        pass
    return COACH_SYSTEM_PROMPTS.get(mode, "")


# ─── Source functions (ALL 25 — unchanged) ──────────────────────────────────
def search_serpapi(query: str, location: str = "United States", num: int = 10) -> list:
    if not SERPAPI_KEY:
        return []
    try:
        results = GoogleSearch({
            "engine": "google_jobs", "q": query, "location": location,
            "hl": "en", "api_key": SERPAPI_KEY, "num": num,
        }).get_dict()
        jobs = [
            _job(
                item.get("title"), item.get("company_name"), item.get("description"),
                item.get("location"),
                "remote" in str(item.get("description", "")).lower(),
                item.get("apply_link") or item.get("share_link"),
                item.get("detected_extensions", {}).get("posted_at"),
                item.get("share_link"),
            )
            for item in results.get("jobs_results", [])
        ]
        return jobs[:num]
    except Exception as exc:
        log.warning("SerpAPI failed: %s", exc)
        return []


def search_adzuna_country(query, country="us", location="", num=10) -> list:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []
    try:
        params = {
            "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY,
            "what": query, "where": location, "max_days_old": 30,
            "results_per_page": min(num, 50),
        }
        response = _request("GET", f"https://api.adzuna.com/v1/api/jobs/{country}/search/1", params=params)
        data = _json(response)
        return [
            _job(
                item.get("title"), item.get("company", {}).get("display_name"),
                item.get("description"), item.get("location", {}).get("display_name"),
                False, item.get("redirect_url"), item.get("created"),
                item.get("redirect_url"),
            )
            for item in data.get("results", [])[:num]
        ]
    except Exception as exc:
        log.warning("Adzuna %s failed: %s", country, exc)
        return []


def search_adzuna(query, location="United States", num=4) -> list:
    return search_adzuna_country(query, "us", location, num)


def search_remotive(query: str, num: int = 4) -> list:
    try:
        data = _json(_request("GET", "https://remotive.com/api/remote-jobs", params={"search": query}))
        return [
            _job(j.get("title"), j.get("company_name"), j.get("description"),
                 j.get("candidate_required_location"), True, j.get("url"),
                 j.get("publication_date"), j.get("url"))
            for j in data.get("jobs", [])[:num]
        ]
    except Exception as exc:
        log.warning("Remotive failed: %s", exc)
        return []


def search_remoteok(query: str, num: int = 10) -> list:
    try:
        response = _request("GET", "https://remoteok.com/api", params={"search": query},
                            headers={"User-Agent": "JobSeekerAI/1.0"})
        data = response.json() if response is not None and response.status_code < 400 else []
        return [
            _job(j.get("position"), j.get("company"), j.get("description"),
                 j.get("location"), True, j.get("url"), j.get("epoch"), j.get("url"))
            for j in data[1:] if isinstance(j, dict)
        ][:num]
    except Exception as exc:
        log.warning("RemoteOK failed: %s", exc)
        return []


def search_findwork(query: str, num: int = 4) -> list:
    if not FINDWORK_KEY:
        return []
    try:
        data = _json(_request("GET", "https://findwork.dev/api/jobs/",
                              params={"search": query},
                              headers={"Authorization": f"Token {FINDWORK_KEY}"}))
        return [
            _job(j.get("role"), j.get("company_name"), j.get("text"),
                 j.get("location"), j.get("remote"), j.get("url"),
                 j.get("date_posted"), j.get("url"))
            for j in data.get("results", [])[:num]
        ]
    except Exception as exc:
        log.warning("FindWork failed: %s", exc)
        return []


def search_jsearch(query: str, location=None, num: int = 4) -> list:
    if not RAPIDAPI_KEY:
        return []
    try:
        params = {"query": query, "page": "1", "num_pages": "1", "date_posted": "all"}
        if location:
            params["location"] = location
        data = _json(_request("GET", "https://jsearch.p.rapidapi.com/search", params=params,
                              headers={"X-RapidAPI-Key": RAPIDAPI_KEY,
                                       "X-RapidAPI-Host": JSEARCH_HOST}))
        return [
            _job(r.get("job_title"), r.get("employer_name"), r.get("job_description"),
                 f"{r.get('job_city', '')}, {r.get('job_country', '')}",
                 r.get("job_is_remote"), r.get("job_apply_link"),
                 r.get("job_posted_at_datetime_utc"), r.get("job_google_link"))
            for r in data.get("data", [])[:num]
        ]
    except Exception as exc:
        log.warning("JSearch failed: %s", exc)
        return []


def _rss_jobs(url: str, query: str, num: int = 4) -> list:
    response = _request("GET", url, headers={"User-Agent": "JobSeekerAI/1.0"})
    if response is None or response.status_code >= 400:
        return []
    soup = BeautifulSoup(response.text, "xml")
    jobs = []
    for item in soup.find_all(["item", "entry"]):
        text = " ".join(item.stripped_strings)
        if query and query.lower() not in text.lower():
            continue
        link_node = item.find("link")
        link = (link_node.get("href", "") if link_node and link_node.has_attr("href")
                else (link_node.get_text(strip=True) if link_node else ""))
        jobs.append(_job(item.find_text("title"), "",
                         item.find_text("description") or item.find_text("summary"),
                         "", True, link,
                         item.find_text("pubDate") or item.find_text("published"), link))
    return jobs[:num]


def search_upwork_rss(query: str, num: int = 4) -> list:
    return _rss_jobs(f"https://www.upwork.com/ab/feed/jobs/rss?{urlencode({'q': query})}", query, num)


def search_reddit_forhire(query: str, num: int = 4) -> list:
    try:
        response = _request("GET", "https://www.reddit.com/r/forhire/new.json",
                            params={"limit": 50}, headers={"User-Agent": "JobSeekerAI/1.0"})
        data = response.json() if response is not None and response.status_code < 400 else {}
        jobs = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title = post.get("title", "")
            if query.lower() not in title.lower() and query.lower() not in post.get("selftext", "").lower():
                continue
            jobs.append(_job(title, "Reddit /r/forhire", post.get("selftext"),
                             "Remote / Reddit", True,
                             f"https://www.reddit.com{post.get('permalink', '')}",
                             post.get("created_utc"),
                             f"https://www.reddit.com{post.get('permalink', '')}"))
        return jobs[:num]
    except Exception as exc:
        log.warning("Reddit forhire failed: %s", exc)
        return []


def search_hackernews(query: str, num: int = 4) -> list:
    try:
        data = _json(_request("GET", "https://hn.algolia.com/api/v1/search_by_date",
                              params={"query": f"who is hiring {query}", "tags": "story",
                                      "hitsPerPage": num}))
        return [
            _job(h.get("title"), "Hacker News", h.get("story_text"),
                 "Remote / Hacker News", True,
                 h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                 h.get("created_at"), h.get("url", ""))
            for h in data.get("hits", [])
        ]
    except Exception as exc:
        log.warning("Hacker News failed: %s", exc)
        return []


def search_careerjet(query: str, country: str = "us", num: int = 4) -> list:
    return _rss_jobs(f"https://www.careerjet.com/search/rss?s={quote(query)}&l=&c={country}", query, num)


# Public SearXNG instances — first one that responds wins.
# SSL errors on public instances are tolerated (they are third-party and often misconfigured).
SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://search.inetol.net",
    "https://searx.tiekoetter.com",
    "https://baresearch.org",
    "https://search.rhscz.eu",
    "https://priv.au",
    "https://search.bus-hit.me",
    "https://search.sapti.me",   # last resort, was failing previously
]


def search_searxng(query: str, instance: str = None, num: int = 4) -> list:
    """Try every known SearXNG instance until one works.
    Tolerates SSL errors on individual instances (public third-party service)."""
    instances = [instance] if instance else SEARXNG_INSTANCES
    for inst in instances:
        try:
            response = requests.get(
                f"{inst.rstrip('/')}/search",
                params={"q": f"{query} jobs", "format": "json", "categories": "it"},
                headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]},
                timeout=HTTP_TIMEOUT,
                verify=False,
            )
            if response is None or response.status_code >= 400:
                log.warning("SearXNG %s status=%s", inst,
                            response.status_code if response is not None else "no-response")
                continue
            try:
                data = response.json() if response.text else {}
            except ValueError:
                log.warning("SearXNG %s returned non-JSON", inst)
                continue
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                continue
            log.info("SearXNG %s returned %d results", inst, len(results))
            return [
                _job(r.get("title"), "SearXNG", r.get("content"), "",
                     True, r.get("url"), "", r.get("url"))
                for r in results[:num]
            ]
        except Exception as exc:
            log.warning("SearXNG %s failed: %s", inst, exc)
            continue
    log.warning("SearXNG all instances failed for query=%s", query)
    return []


def search_metager(query: str, num: int = 10) -> list:
    return search_searxng(query, "https://metager.org", num)


def search_mojeek(query: str, num: int = 10) -> list:
    try:
        response = _request("GET", "https://www.mojeek.com/search", params={"q": f"{query} jobs"})
        soup = BeautifulSoup(response.text if response is not None else "", "html.parser")
        return [_job(node.get_text(" ", strip=True), "Mojeek", "", "",
                     True, node.get("href"), "", node.get("href"))
                for node in soup.select("a.title")[:num]]
    except Exception as exc:
        log.warning("Mojeek failed: %s", exc)
        return []


def search_stract(query: str, num: int = 10) -> list:
    return search_searxng(query, "https://stract.com", num)


def search_arbeitnow(query: str, num: int = 7) -> list:
    try:
        data = _json(_request("GET", "https://www.arbeitnow.com/api/job-board-api"))
        return [_job(j.get("title"), j.get("company_name"), j.get("description"),
                     j.get("location"), j.get("remote"), j.get("url"),
                     j.get("created_at"), j.get("url"))
                for j in data.get("data", []) if query.lower() in json.dumps(j).lower()][:num]
    except Exception as exc:
        log.warning("Arbeitnow failed: %s", exc)
        return []


def search_jobicy(query: str, num: int = 10) -> list:
    try:
        data = _json(_request("GET", "https://jobicy.com/api/v2/remote-jobs", params={"count": 30}))
        return [_job(j.get("jobTitle"), j.get("companyName"), j.get("jobDescription"),
                     j.get("jobGeo"), True, j.get("url"), j.get("pubDate"), j.get("url"))
                for j in data.get("jobs", []) if query.lower() in json.dumps(j).lower()][:num]
    except Exception as exc:
        log.warning("Jobicy failed: %s", exc)
        return []


def search_himalayas(query: str, num: int = 5) -> list:
    try:
        data = _json(_request("GET", "https://himalayas.app/jobs/api"))
        return [_job(j.get("title"), j.get("companyName"), j.get("description"),
                     j.get("location"), True,
                     j.get("applicationLink") or j.get("url"),
                     j.get("pubDate"), j.get("url"))
                for j in data.get("jobs", []) if query.lower() in json.dumps(j).lower()][:num]
    except Exception as exc:
        log.warning("Himalayas failed: %s", exc)
        return []


def search_working_nomads(query: str, num: int = 5) -> list:
    try:
        response = _request("GET", "https://www.workingnomads.com/api/exposed_jobs/")
        data = response.json() if response is not None and response.status_code < 400 else []
        return [_job(j.get("title"), j.get("company"), j.get("description"),
                     j.get("location"), True, j.get("url"), j.get("date"), j.get("url"))
                for j in data if query.lower() in json.dumps(j).lower()][:num]
    except Exception as exc:
        log.warning("Working Nomads failed: %s", exc)
        return []


def search_themuse(query: str, num: int = 10) -> list:
    try:
        data = _json(_request("GET", "https://www.themuse.com/api/public/jobs",
                              params={"page": 0, "descending": "true"}))
        return [_job(j.get("name"), j.get("company", {}).get("name"), j.get("contents"),
                     j.get("locations", [{}])[0].get("name"), False,
                     j.get("refs", {}).get("landing_page"), "",
                     j.get("refs", {}).get("landing_page"))
                for j in data.get("results", []) if query.lower() in json.dumps(j).lower()][:num]
    except Exception as exc:
        log.warning("The Muse failed: %s", exc)
        return []


def search_jobspresso(query: str, num: int = 5) -> list:
    return _rss_jobs("https://jobspresso.co/remote-work/feed/", query, num)


def search_weworkremotely(query: str, num: int = 7) -> list:
    return _rss_jobs("https://weworkremotely.com/remote-jobs.rss", query, num)


def search_remoteco(query: str, num: int = 5) -> list:
    return _rss_jobs("https://remote.co/remote-jobs/feed/", query, num)


# ─── Normalization / matching ────────────────────────────────────────────────
def normalize_and_deduplicate(jobs: list) -> list:
    seen: set = set()
    unique = []
    for job in jobs:
        if not job.get("title"):
            continue
        key = "|".join(str(job.get(f, "")).lower().strip()
                       for f in ("title", "company", "source_url"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


def location_match(job: dict, desired_location) -> bool:
    if not desired_location:
        return bool(job.get("remote"))
    location = re.sub(r"\s+", " ", re.sub(r"[,\-]+", " ", str(job.get("location", "")).lower())).strip()
    desired = desired_location.lower().strip()
    return bool(job.get("remote")) or desired in location


def _source_functions(title: str, location: str) -> list:
    query = f"{title} {location}".strip()
    return [
        ("SerpAPI", lambda: search_serpapi(query, location)),
        ("Adzuna-us", lambda: search_adzuna_country(title, "us", location)),
        ("Adzuna-ca", lambda: search_adzuna_country(title, "ca", location)),
        ("Adzuna-gb", lambda: search_adzuna_country(title, "gb", location)),
        ("Adzuna-ng", lambda: search_adzuna_country(title, "ng", location)),
        ("Remotive", lambda: search_remotive(title)),
        ("RemoteOK", lambda: search_remoteok(title)),
        ("FindWork", lambda: search_findwork(query)),
        ("JSearch", lambda: search_jsearch(query, location)),
        ("UpworkRSS", lambda: search_upwork_rss(title)),
        ("RedditForHire", lambda: search_reddit_forhire(title)),
        ("HackerNews", lambda: search_hackernews(title)),
        ("CareerJet", lambda: search_careerjet(title)),
        ("SearXNG", lambda: search_searxng(title)),
        ("MetaGer", lambda: search_metager(title)),
        ("Mojeek", lambda: search_mojeek(title)),
        ("Stract", lambda: search_stract(title)),
        ("Arbeitnow", lambda: search_arbeitnow(title)),
        ("Jobicy", lambda: search_jobicy(title)),
        ("Himalayas", lambda: search_himalayas(title)),
        ("WorkingNomads", lambda: search_working_nomads(title)),
        ("TheMuse", lambda: search_themuse(title)),
        ("Jobspresso", lambda: search_jobspresso(title)),
        ("WeWorkRemotely", lambda: search_weworkremotely(title)),
        ("Remote.co", lambda: search_remoteco(title)),
    ]


def agentic_job_search(title: str, location, num_per_source: int = 3):
    location = location or "United States"
    all_jobs = []
    sources = _source_functions(title, location)
    with ThreadPoolExecutor(max_workers=min(12, len(sources))) as executor:
        active_sources = [(name, fn) for name, fn in sources
                          if not source_health.get(name, {}).get("disabled")]
        futures = {executor.submit(fn): name for name, fn in active_sources}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()[:num_per_source]
                all_jobs.extend(result)
                health = source_health.setdefault(name, {"empty_runs": 0, "disabled": False})
                if result:
                    health["empty_runs"] = 0
                else:
                    health["empty_runs"] += 1
                    if health["empty_runs"] >= 3:
                        health["disabled"] = True
                        log.warning("source=%s disabled_after_consecutive_empty_runs=3", name)
                log.info("source=%s jobs=%d query=%s", name, len(result), title)
            except Exception as exc:
                health = source_health.setdefault(name, {"empty_runs": 0, "disabled": False})
                health["empty_runs"] += 1
                if health["empty_runs"] >= 3:
                    health["disabled"] = True
                    log.warning("source=%s disabled_after_consecutive_errors=3", name)
                log.warning("source=%s error=%s", name, exc)
    unique = normalize_and_deduplicate(all_jobs)
    exact = [job for job in unique if location_match(job, location)]
    others = [job for job in unique if job not in exact]
    combined = (exact + others)[:50]
    log.info("agentic_search title=%s exact=%d total=%d sources=%d",
             title, len(exact), len(combined), len(sources))
    return combined, len(exact)


def insert_or_get_ids(jobs: list) -> list:
    ids = []
    for job in jobs:
        try:
            filter_value = (f"(title='{job.get('title', '')}'&&company='{job.get('company', '')}'"
                            f"&&source_url='{job.get('source_url', '')}')")
            path = f"/collections/job_listings/records?filter={quote(filter_value)}"
            response = pb("GET", path)
            data = _json(response)
            if data.get("totalItems", 0) > 0:
                ids.append(data["items"][0]["id"])
            else:
                created = pb("POST", "/collections/job_listings/records", json_data=job)
                created_data = _json(created)
                if created_data.get("id"):
                    ids.append(created_data["id"])
            time.sleep(0.05)
        except Exception as exc:
            log.warning("Job insert failed: %s", exc)
    return ids


# ─── Search request processor (with tier limit) ─────────────────────────────
def process_search_requests() -> None:
    log.info("Search request processor thread started (tier-aware)")
    while True:
        try:
            data = _json(pb("GET", "/collections/job_search_requests/records?filter=(status='pending')&sort=created&perPage=5"))
            for request in data.get("items", []):
                request_id = request.get("id")
                user_id = request.get("user")
                raw_query = request.get("query", "")
                if not request_id or not raw_query:
                    continue

                # Tier gate
                tier = get_user_tier(user_id) if user_id else "free"
                daily_cap = 999999 if tier == "contributor" else FREE_TIER_SEARCHES_PER_DAY
                usage = get_or_create_daily_usage(user_id) if user_id else {}
                current_searches = (usage.get("job_searches") or 0) if usage else 0
                if current_searches >= daily_cap:
                    pb("PATCH", f"/collections/job_search_requests/records/{request_id}",
                       {"status": "limit_reached"})
                    log.info("search_request=%s user=%s tier=%s daily_cap_reached=%d",
                             request_id, user_id, tier, current_searches)
                    continue

                pb("PATCH", f"/collections/job_search_requests/records/{request_id}",
                   {"status": "running"})
                if user_id:
                    increment_usage(user_id, "job_searches", 1)

                params = parse_natural_query(raw_query)
                title = params.get("title") or raw_query
                location = params.get("location") or "United States"
                jobs, exact_count = agentic_job_search(title, location)
                job_ids = insert_or_get_ids(jobs)
                pb("PATCH", f"/collections/job_search_requests/records/{request_id}",
                   {"status": "completed", "results": job_ids, "exact_match_count": exact_count})
                log.info("search_request=%s completed jobs=%d exact=%d",
                         request_id, len(job_ids), exact_count)
        except Exception as exc:
            log.exception("Search request loop recovered from error: %s", exc)
        time.sleep(10)


# ─── Legacy chat processing (chat_messages) ─────────────────────────────────
def _claim_message(message_id: str) -> bool:
    with claim_lock:
        now = time.time()
        for key, claimed_at in list(claimed_messages.items()):
            if now - claimed_at > 900:
                claimed_messages.pop(key, None)
        if message_id in claimed_messages:
            return False
        claimed_messages[message_id] = now
        return True


def _chat_once(message: dict) -> None:
    message_id = message.get("id")
    user_id = message.get("user")
    text = str(message.get("message", "")).strip()
    if not message_id or not user_id or not text or not _claim_message(message_id):
        return
    try:
        history_data = _json(pb("GET", f"/collections/chat_messages/records?filter=(user='{quote(str(user_id))}')&sort=created&perPage=10"))
        history = []
        for item in history_data.get("items", []):
            if item.get("id") == message_id:
                continue
            if item.get("message"):
                history.append({"role": "user", "content": str(item["message"])})
            if item.get("response"):
                history.append({"role": "assistant", "content": str(item["response"])})
        history.append({"role": "user", "content": text})

        user_data = _json(pb("GET", f"/collections/users/records/{quote(str(user_id))}"))
        profile = "\n".join(
            f"{label}: {user_data.get(field, '')}"
            for label, field in (("Name", "full_name"), ("Skills", "skills"),
                                 ("Desired Job", "desired_job_title"),
                                 ("Location", "location"))
            if user_data.get(field)
        )
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        if profile:
            messages.append({"role": "system", "content": f"Known user profile:\n{profile}"})
        messages.extend(history[-10:])
        answer = _ai_chat(messages, temperature=0.8, max_tokens=500)
        patched = pb("PATCH", f"/collections/chat_messages/records/{message_id}", {"response": answer})
        if patched is None or patched.status_code >= 400:
            raise RuntimeError(f"PB did not accept chat response: {patched.status_code if patched else 'no response'}")
        log.info("Replied to chat message=%s", message_id)
    except Exception as exc:
        log.exception("Chat message failed message=%s: %s", message_id, exc)
    finally:
        with claim_lock:
            claimed_messages.pop(str(message_id), None)


def fast_chat_loop(worker_name: str = "chat-1", startup_offset: int = 0) -> None:
    if startup_offset:
        time.sleep(startup_offset)
    log.info("Chat worker started name=%s", worker_name)
    while True:
        try:
            data = _json(pb("GET", "/collections/chat_messages/records?filter=(response='')&sort=created&perPage=10"))
            for message in data.get("items", []):
                _chat_once(message)
        except Exception as exc:
            log.exception("Chat loop recovered name=%s error=%s", worker_name, exc)
        time.sleep(3)


# ─── Coach fallback processor (NEW) ─────────────────────────────────────────
def _claim_coach(message_id: str) -> bool:
    with claim_lock:
        now = time.time()
        for key, claimed_at in list(claimed_coach_messages.items()):
            if now - claimed_at > 900:
                claimed_coach_messages.pop(key, None)
        if message_id in claimed_coach_messages:
            return False
        claimed_coach_messages[message_id] = now
        return True


def _process_coach_fallback(msg: dict) -> None:
    msg_id = msg.get("id")
    conv_id = msg.get("conversation")
    user_id = msg.get("user")
    text = str(msg.get("content", "")).strip()
    if not msg_id or not conv_id or not user_id or not text:
        return

    # If an assistant reply already exists in this conversation after this user msg, skip.
    check = _json(pb("GET", f"/collections/coach_messages/records?filter=(conversation='{quote(conv_id)}' && role='assistant')&sort=-created&perPage=1"))
    if check.get("items"):
        latest_assistant_created = check["items"][0].get("created", "")
        if latest_assistant_created > msg.get("created", ""):
            return

    if not _claim_coach(msg_id):
        return

    try:
        conv = _json(pb("GET", f"/collections/coach_conversations/records/{quote(conv_id)}"))
        mode = conv.get("mode", "interview")
        system_prompt = get_active_coach_prompt(mode)

        hist_data = _json(pb("GET", f"/collections/coach_messages/records?filter=(conversation='{quote(conv_id)}')&sort=created&perPage=20"))
        history = []
        for h in hist_data.get("items", []):
            if h.get("id") == msg_id:
                continue
            if h.get("role") in ("user", "assistant") and h.get("content"):
                history.append({"role": h["role"], "content": h["content"]})

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": text})

        answer = _ai_chat(messages, temperature=0.8, max_tokens=600)
        pb("POST", "/collections/coach_messages/records", {
            "conversation": conv_id, "user": user_id,
            "role": "assistant", "content": answer,
            "model_used": "worker-fallback",
        })
        pb("PATCH", f"/collections/coach_conversations/records/{quote(conv_id)}",
           {"last_message_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000Z")})
        log.info("coach_fallback replied msg=%s mode=%s", msg_id, mode)
    except Exception as exc:
        log.warning("coach_fallback failed msg=%s: %s", msg_id, exc)
    finally:
        with claim_lock:
            claimed_coach_messages.pop(str(msg_id), None)


def coach_chat_loop() -> None:
    """Server-side safety net: if the frontend cannot reach Mistral,
    this loop generates the assistant reply for unanswered user messages.
    Waits 30s before acting so we never race the frontend."""
    log.info("Coach fallback processor thread started")
    while True:
        try:
            data = _json(pb("GET", "/collections/coach_messages/records?filter=(role='user')&sort=-created&perPage=15"))
            now = time.time()
            for msg in data.get("items", []):
                created_ts = _parse_pb_date(msg.get("created", ""))
                if not created_ts or (now - created_ts) < 30:
                    continue
                _process_coach_fallback(msg)
        except Exception as exc:
            log.exception("coach_chat_loop recovered: %s", exc)
        time.sleep(15)


# ─── Health check ────────────────────────────────────────────────────────────
def health_check_loop() -> None:
    while True:
        try:
            data = _json(pb("GET", "/collections/chat_messages/records?filter=(response='')&perPage=1"))
            unanswered = data.get("totalItems", 0)
            coach_data = _json(pb("GET", "/collections/coach_messages/records?filter=(role='user')&perPage=1"))
            coach_total = coach_data.get("totalItems", 0)
            reachable = False
            if MISTRAL_API_KEY:
                response = _request("GET", "https://api.mistral.ai/v1/models",
                                    headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"})
                reachable = response is not None and response.status_code < 400
            log.info("health chat_unanswered=%s coach_total=%s mistral=%s",
                     unanswered, coach_total, reachable)
        except Exception as exc:
            log.exception("Health check recovered from error: %s", exc)
        time.sleep(60)


# ─── Scraping loop (tier-aware) ─────────────────────────────────────────────
def scraping_loop() -> None:
    log.info("Scraping worker started (tier-aware: free=%d / contributor=%d per day)",
             FREE_TIER_JOBS_PER_DAY, CONTRIBUTOR_TIER_JOBS_PER_DAY)
    while True:
        try:
            users = _json(pb("GET", "/collections/users/records?perPage=200")).get("items", [])
            for user in users:
                try:
                    user_id = user.get("id")
                    if not user_id:
                        continue
                    tier = user.get("tier") or "free"
                    daily_cap = CONTRIBUTOR_TIER_JOBS_PER_DAY if tier == "contributor" else FREE_TIER_JOBS_PER_DAY

                    usage = get_or_create_daily_usage(user_id)
                    jobs_shown = (usage.get("jobs_shown") or 0) if usage else 0
                    if jobs_shown >= daily_cap:
                        log.info("user=%s tier=%s at daily cap (%d/%d) - skipping",
                                 user_id, tier, jobs_shown, daily_cap)
                        continue

                    query = " ".join(filter(None, [
                        str(user.get("desired_job_title", "")).strip(),
                        str(user.get("skills", "")).strip(),
                    ]))
                    if not query:
                        continue
                    if user.get("remote_preference") == "remote":
                        query += " remote"

                    remaining = daily_cap - jobs_shown
                    jobs, _ = agentic_job_search(query, user.get("location") or "United States")
                    jobs = jobs[:remaining]
                    insert_or_get_ids(jobs)
                    increment_usage(user_id, "jobs_shown", len(jobs))
                    log.info("user=%s tier=%s scraped=%d remaining_cap=%d",
                             user_id, tier, len(jobs), remaining - len(jobs))
                    time.sleep(2)
                except Exception as exc:
                    log.warning("scraping for user %s failed: %s", user.get("id"), exc)
            log.info("Scraping cycle complete; sleeping 1 hour")
        except Exception as exc:
            log.exception("Scraping loop recovered from error: %s", exc)
        time.sleep(3600)


# ─── Bootstrap ──────────────────────────────────────────────────────────────
def _start_thread(target: Callable[..., None], name: str, *args: Any) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, name=name, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    if not POCKETBASE_ADMIN_TOKEN:
        raise RuntimeError("PB_ADMIN_TOKEN is required")
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is required")
    log.info("Starting resilient JobSeeker worker threads")
    log.info("Model chain: %s", " → ".join(MISTRAL_MODELS))
    _start_thread(fast_chat_loop, "chat-worker-1", "chat-1", 0)
    _start_thread(fast_chat_loop, "chat-worker-2", "chat-2", 5)
    _start_thread(coach_chat_loop, "coach-fallback-worker")
    _start_thread(process_search_requests, "search-worker")
    _start_thread(health_check_loop, "health-worker")
    scraping_loop()
