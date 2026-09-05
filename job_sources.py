"""
job_sources.py
--------------
One function per job platform. Every function has the same signature and
the same return shape, so app.py can call any of them interchangeably.

STRICT RULE: this file must NEVER import from app.py. It only depends on
the standard library and `requests`.

Adding a new source later only requires:
    1. Writing one new search_<source>(query, limit) function that
       returns a list of job dicts in the shared shape (see _make_job).
    2. Adding one line to the SOURCES list at the bottom of this file.
No other code, anywhere, needs to change.

Design notes:
- We use official/public JSON or RSS endpoints only. Sites that prohibit
  scraping (e.g. LinkedIn, Indeed's HTML pages) are implemented as stub
  functions that return an empty result with a clear "not available"
  status, instead of scraping their pages. If the user has a legitimate
  API key or a company-provided export for a platform, those stubs are
  the place to plug that in later - no other code needs to change.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

REQUEST_TIMEOUT = 10  # seconds - keep searches fast and predictable

# API keys for sources that require one. Read once at import time so
# every source function can just reference the constant.
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY", "")

# Adzuna is queried per-country. Since this app targets worldwide remote
# roles (not any single country), we sweep a handful of major English-
# language job markets per search rather than hardcoding one region.
# Add or remove country codes here to change coverage - no other code
# needs to change.
ADZUNA_COUNTRIES = ["us", "gb", "ca", "in", "sg", "ae"]


def _make_job(
    title: str = "",
    company: str = "",
    location: str = "",
    remote: bool = False,
    salary: str = "",
    employment_type: str = "",
    posting_date: str = "",
    source: str = "",
    apply_url: str = "",
    description: str = "",
) -> dict:
    """
    Build a job dict in the shared shape every source function returns.
    Keeping this in one place means every source produces identical keys,
    so app.py never needs source-specific handling.
    """
    return {
        "title": title,
        "company": company,
        "location": location,
        "remote": remote,
        "salary": salary,
        "employment_type": employment_type,
        "posting_date": posting_date,
        "source": source,
        "apply_url": apply_url,
        "description": description,
    }


def _matches_query(job_text: str, query: str) -> bool:
    """Simple case-insensitive keyword match used by sources with no search API."""
    if not query:
        return True
    return query.lower() in job_text.lower()


# --- Real, public-API sources -----------------------------------------------------


def search_remoteok(query: str, limit: int = 20) -> list[dict]:
    """
    Search RemoteOK's public JSON feed (https://remoteok.com/api).
    No API key required.
    """
    try:
        response = requests.get(
            "https://remoteok.com/api",
            headers={"User-Agent": "Mozilla/5.0 (AI Job Hunter)"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        # Let this bubble up to run_source() so the real reason (blocked,
        # timed out, bad response, etc) shows up in Search Transparency
        # instead of silently looking like "no results".
        raise RuntimeError(f"RemoteOK request failed: {exc}") from exc

    jobs = []
    # The first item is metadata, not a job - skip it.
    for item in data[1:]:
        title = item.get("position", "")
        description = item.get("description", "")
        if not _matches_query(f"{title} {description}", query):
            continue
        jobs.append(
            _make_job(
                title=title,
                company=item.get("company", ""),
                location=item.get("location", "") or "Remote",
                remote=True,
                salary=item.get("salary", "") or "",
                employment_type="",
                posting_date=item.get("date", ""),
                source="RemoteOK",
                apply_url=item.get("url", ""),
                description=description[:500],
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def search_workingnomads(query: str, limit: int = 20) -> list[dict]:
    """
    Search Working Nomads' public jobs API.
    No API key required.
    """
    try:
        response = requests.get(
            "https://www.workingnomads.com/api/exposed_jobs/",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Working Nomads request failed: {exc}") from exc

    jobs = []
    for item in data:
        title = item.get("title", "")
        description = item.get("description", "")
        if not _matches_query(f"{title} {description}", query):
            continue
        jobs.append(
            _make_job(
                title=title,
                company=item.get("company_name", ""),
                location=item.get("location", "") or "Remote",
                remote=True,
                salary="",
                employment_type=", ".join(item.get("category_name", "").split(",")) if item.get("category_name") else "",
                posting_date=item.get("pub_date", ""),
                source="Working Nomads",
                apply_url=item.get("url", ""),
                description=description[:500],
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def search_weworkremotely(query: str, limit: int = 20) -> list[dict]:
    """
    Search We Work Remotely's public RSS feed.
    No API key required.
    """
    try:
        response = requests.get(
            "https://weworkremotely.com/remote-jobs.rss",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        raise RuntimeError(f"We Work Remotely request failed: {exc}") from exc

    jobs = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        if not _matches_query(f"{title} {description}", query):
            continue

        # WWR titles are usually "Company: Job Title"
        company, _, role = title.partition(":")
        jobs.append(
            _make_job(
                title=role.strip() or title,
                company=company.strip() if role else "",
                location="Remote",
                remote=True,
                salary="",
                employment_type="",
                posting_date=pub_date,
                source="We Work Remotely",
                apply_url=link,
                description=description[:500],
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def search_greenhouse(query: str, limit: int = 20, company_slugs: list[str] | None = None) -> list[dict]:
    """
    Search one or more companies' public Greenhouse job boards.
    No API key required, but you must know each company's Greenhouse slug
    (the part of their board URL, e.g. "stripe" for boards.greenhouse.io/stripe).

    NOTE: this only searches whichever companies are named in company_slugs
    (or the small default list below), so for most searches it will find
    nothing - that's expected, not a bug. It's here as scaffolding for a
    future feature where a user supplies their own target-company list.
    Broad, unrestricted coverage comes from search_remotive, search_arbeitnow,
    search_adzuna, and search_jooble instead.
    """
    slugs = company_slugs or ["stripe", "airbnb", "notion", "figma"]
    jobs = []

    for slug in slugs:
        try:
            response = requests.get(
                f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue

        for item in data.get("jobs", []):
            title = item.get("title", "")
            if not _matches_query(title, query):
                continue
            jobs.append(
                _make_job(
                    title=title,
                    company=slug.title(),
                    location=(item.get("location") or {}).get("name", ""),
                    remote="remote" in title.lower(),
                    salary="",
                    employment_type="",
                    posting_date=item.get("updated_at", ""),
                    source="Greenhouse",
                    apply_url=item.get("absolute_url", ""),
                    description="",
                )
            )
            if len(jobs) >= limit:
                return jobs
    return jobs


def search_lever(query: str, limit: int = 20, company_slugs: list[str] | None = None) -> list[dict]:
    """
    Search one or more companies' public Lever job boards.
    No API key required, but you must know each company's Lever slug.
    Same narrow-by-design caveat as search_greenhouse above.
    """
    slugs = company_slugs or ["netflix", "shopify"]
    jobs = []

    for slug in slugs:
        try:
            response = requests.get(
                f"https://api.lever.co/v0/postings/{slug}?mode=json",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue

        for item in data:
            title = item.get("text", "")
            if not _matches_query(title, query):
                continue
            categories = item.get("categories", {})
            jobs.append(
                _make_job(
                    title=title,
                    company=slug.title(),
                    location=categories.get("location", ""),
                    remote="remote" in (categories.get("location", "") or "").lower(),
                    salary="",
                    employment_type=categories.get("commitment", ""),
                    posting_date="",
                    source="Lever",
                    apply_url=item.get("hostedUrl", ""),
                    description="",
                )
            )
            if len(jobs) >= limit:
                return jobs
    return jobs


def search_adzuna(query: str, limit: int = 20) -> list[dict]:
    """
    Search Adzuna's public API across several major job markets, keeping
    only remote-tagged results (or results whose title/description
    mentions remote work) since this app is for worldwide remote roles,
    not any single country. Requires ADZUNA_APP_ID and ADZUNA_APP_KEY.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []

    jobs = []
    per_country_limit = max(1, limit // len(ADZUNA_COUNTRIES))

    for country in ADZUNA_COUNTRIES:
        try:
            response = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
                params={
                    "app_id": ADZUNA_APP_ID,
                    "app_key": ADZUNA_APP_KEY,
                    "what": query,
                    "what_and": "remote",
                    "results_per_page": per_country_limit,
                    "content-type": "application/json",
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue

        for item in data.get("results", []):
            title = item.get("title", "")
            description = item.get("description", "")
            location = (item.get("location") or {}).get("display_name", "")
            is_remote = "remote" in f"{title} {description} {location}".lower()

            salary_min = item.get("salary_min")
            salary_max = item.get("salary_max")
            salary = f"{salary_min:.0f}-{salary_max:.0f}" if salary_min and salary_max else ""

            jobs.append(
                _make_job(
                    title=title,
                    company=(item.get("company") or {}).get("display_name", ""),
                    location=location,
                    remote=is_remote,
                    salary=salary,
                    employment_type=item.get("contract_time", "") or "",
                    posting_date=item.get("created", ""),
                    source="Adzuna",
                    apply_url=item.get("redirect_url", ""),
                    description=description[:500],
                )
            )
            if len(jobs) >= limit:
                return jobs
    return jobs


def search_jooble(query: str, limit: int = 20) -> list[dict]:
    """
    Search the Jooble API for remote roles. Jooble doesn't take a
    location filter that means "anywhere" universally, so we pass the
    word "remote" as the location to bias results toward remote-friendly
    postings worldwide. Requires JOOBLE_API_KEY.
    """
    if not JOOBLE_API_KEY:
        return []

    try:
        response = requests.post(
            f"https://jooble.org/api/{JOOBLE_API_KEY}",
            json={"keywords": query, "location": "remote"},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    jobs = []
    for item in data.get("jobs", []):
        title = item.get("title", "")
        location = item.get("location", "")
        snippet = item.get("snippet", "")
        jobs.append(
            _make_job(
                title=title,
                company=item.get("company", ""),
                location=location,
                remote="remote" in f"{title} {location} {snippet}".lower(),
                salary=item.get("salary", "") or "",
                employment_type=item.get("type", "") or "",
                posting_date=item.get("updated", ""),
                source="Jooble",
                apply_url=item.get("link", ""),
                description=snippet[:500],
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def search_remotive(query: str, limit: int = 20) -> list[dict]:
    """
    Search Remotive's public jobs API (https://remotive.com/api/remote-jobs).
    No API key required. Broad remote-job coverage across many companies,
    not limited to a hardcoded list like Greenhouse/Lever above.
    """
    try:
        response = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": query} if query else {},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Remotive request failed: {exc}") from exc

    jobs = []
    for item in data.get("jobs", []):
        jobs.append(
            _make_job(
                title=item.get("title", ""),
                company=item.get("company_name", ""),
                location=item.get("candidate_required_location", "") or "Remote",
                remote=True,
                salary=item.get("salary", "") or "",
                employment_type=item.get("job_type", ""),
                posting_date=item.get("publication_date", ""),
                source="Remotive",
                apply_url=item.get("url", ""),
                description=(item.get("description", "") or "")[:500],
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def search_arbeitnow(query: str, limit: int = 20) -> list[dict]:
    """
    Search Arbeitnow's public job board API
    (https://www.arbeitnow.com/api/job-board-api). No API key required.
    Covers both remote and on-site roles across many companies.
    """
    try:
        response = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Arbeitnow request failed: {exc}") from exc

    jobs = []
    for item in data.get("data", []):
        title = item.get("title", "")
        description = item.get("description", "")
        if not _matches_query(f"{title} {description}", query):
            continue
        jobs.append(
            _make_job(
                title=title,
                company=item.get("company_name", ""),
                location=item.get("location", "") or ("Remote" if item.get("remote") else ""),
                remote=bool(item.get("remote")),
                salary="",
                employment_type=", ".join(item.get("job_types", []) or []),
                posting_date=str(item.get("created_at", "")),
                source="Arbeitnow",
                apply_url=item.get("url", ""),
                description=description[:500],
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def search_reddit_jobs(query: str, limit: int = 20) -> list[dict]:
    """
    Search a handful of job-focused subreddits' public JSON feeds
    (no login/API key required for this kind of read-only access).
    Results are raw posts, not structured job listings, so match
    quality is rougher than the other sources - useful as a
    supplementary source, not a primary one.
    """
    subreddits = ["remotejs", "forhire", "remotejobs", "WorkOnline"]
    jobs = []

    for subreddit in subreddits:
        try:
            response = requests.get(
                f"https://www.reddit.com/r/{subreddit}/new.json",
                headers={"User-Agent": "AI Job Hunter/1.0"},
                params={"limit": 50},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            # A single blocked/rate-limited subreddit shouldn't stop the others.
            continue

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title = post.get("title", "")
            # Reddit's [FOR HIRE] tag is from the poster's own perspective,
            # not job listings - skip those to keep results relevant.
            if title.strip().upper().startswith("[FOR HIRE]"):
                continue
            if not _matches_query(title, query):
                continue
            jobs.append(
                _make_job(
                    title=title,
                    company="",
                    location="Remote",
                    remote=True,
                    salary="",
                    employment_type="",
                    posting_date="",
                    source=f"Reddit (r/{subreddit})",
                    apply_url=f"https://www.reddit.com{post.get('permalink', '')}",
                    description=(post.get("selftext", "") or "")[:500],
                )
            )
            if len(jobs) >= limit:
                return jobs
    return jobs


# --- Stub sources: no scraping, need official API access -----------------------------------------------------
#
# These platforms either prohibit scraping outright (LinkedIn, Indeed) or
# require a paid/partner API (Wellfound, Upwork, Freelancer, FlexJobs).
# Each stub returns an empty list so it fails safely inside SOURCES - the
# UI will show "0 jobs found - requires official API access" for it
# rather than the app breaking. Swap in a real implementation later
# (e.g. an official partner API key, or a user-provided data export)
# without touching any other file.


def search_indeed(query: str, limit: int = 20) -> list[dict]:
    """Stub: Indeed has no public search API and prohibits scraping its pages."""
    return []


def search_linkedin(query: str, limit: int = 20) -> list[dict]:
    """Stub: LinkedIn prohibits scraping. Requires an official API partnership
    or a user-provided export (e.g. saved-jobs CSV) to populate this."""
    return []


def search_wellfound(query: str, limit: int = 20) -> list[dict]:
    """Stub: Wellfound (AngelList Talent) has no public self-serve search API."""
    return []


def search_upwork(query: str, limit: int = 20) -> list[dict]:
    """Stub: Upwork's API requires an approved partner application."""
    return []


def search_freelancer(query: str, limit: int = 20) -> list[dict]:
    """Stub: Freelancer.com's API requires an approved developer application."""
    return []


# --- Source registry -----------------------------------------------------
#
# Every enabled source lives here as (display_name, function). app.py
# loops over this list so it never needs to know about a specific source
# by name. Add a new source by adding one line here.

SOURCES: list[tuple[str, Callable[..., list[dict]]]] = [
    ("RemoteOK", search_remoteok),
    ("Working Nomads", search_workingnomads),
    ("We Work Remotely", search_weworkremotely),
    ("Greenhouse", search_greenhouse),
    ("Lever", search_lever),
    ("Remotive", search_remotive),
    ("Arbeitnow", search_arbeitnow),
    ("Reddit", search_reddit_jobs),
    ("Adzuna", search_adzuna),
    ("Jooble", search_jooble),
    ("Indeed", search_indeed),
    ("LinkedIn", search_linkedin),
    ("Wellfound", search_wellfound),
    ("Upwork", search_upwork),
    ("Freelancer", search_freelancer),
]


def run_source(name: str, func: Callable[..., list[dict]], query: str, limit: int = 20) -> dict:
    """
    Run one source function and capture timing + status alongside its
    results, in the shape the "Search Transparency" section needs.
    """
    start = time.time()
    try:
        jobs = func(query, limit)
        status = "ok" if jobs else "no results"
    except Exception as exc:  # noqa: BLE001 - a single bad source shouldn't break the search
        jobs = []
        status = f"error: {exc}"
    elapsed = round(time.time() - start, 2)

    return {
        "name": name,
        "status": status,
        "jobs_found": len(jobs),
        "search_time": elapsed,
        "jobs": jobs,
    }


def search_all_sources(query: str, limit_per_source: int = 20) -> list[dict]:
    """
    Run every enabled source in SOURCES for a single query and return
    one result dict per source (see run_source). app.py calls this once
    per generated search query and merges the results.
    """
    return [run_source(name, func, query, limit_per_source) for name, func in SOURCES]
