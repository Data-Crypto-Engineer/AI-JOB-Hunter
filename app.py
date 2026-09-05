"""
app.py
------
AI Job Hunter - Streamlit UI and AI orchestration.

Flow:
1. User provides a Master Resume (paste or upload PDF/DOCX/TXT) and any
   optional preferences.
2. On "Find Jobs": Gemini reads the resume + preferences and builds a
   candidate profile (explicit skills, hidden/transferable skills,
   likely job titles, experience level, etc).
3. Gemini turns that profile into several diversified search queries.
4. Every enabled source in job_sources.py is searched for every query.
5. Gemini ranks the combined, de-duplicated results against the profile,
   explains each match, and flags missing skills.
6. Results are shown in a sortable/filterable table with export options,
   plus a "Search Transparency" panel and AI-curated recommendation lists.

STRICT RULE: this file may import job_sources.py, but job_sources.py
must never import this file.
"""

from __future__ import annotations

import io
import json
import re
import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os

from job_sources import SOURCES, search_all_sources

# --- Configuration -----------------------------------------------------
# Kept here (not a separate config.py) to respect the two-file structure
# this project uses: app.py and job_sources.py only.

load_dotenv()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Change this single value to switch to another Gemini model.
# gemini-3.5-flash-lite is used here instead of gemini-3.6-flash because
# this app's AI steps (profile extraction, query generation, job ranking)
# are classification/extraction tasks, not deep reasoning - Flash-Lite is
# built for exactly that: much faster and cheaper, at a small cost in
# reasoning depth we don't really need here. Switch back to
# "gemini-3.6-flash" if you want stronger reasoning and don't mind the
# extra latency/cost.
GEMINI_MODEL: str = "gemini-3.5-flash-lite"

MAX_QUERIES: int = 12         # how many diversified search queries the AI generates
MAX_JOBS_TO_RANK: int = 30    # jobs per AI ranking batch call (keeps each call's token usage bounded)
TOTAL_JOB_CAP: int = 150      # hard ceiling on jobs ranked per search, across all batches
RESULTS_PER_SOURCE: int = 15  # jobs pulled per source, per query


# --- Resume extraction -----------------------------------------------------
# Lives here (not job_sources.py) because it's about the candidate's
# profile, not a job source.


def extract_pdf(file) -> str:
    """Extract text from an uploaded PDF resume."""
    import pypdf

    reader = pypdf.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def extract_docx(file) -> str:
    """Extract text from an uploaded DOCX resume."""
    from docx import Document

    document = Document(io.BytesIO(file.read()))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()


def extract_txt(file) -> str:
    """Extract text from an uploaded TXT resume."""
    return file.read().decode("utf-8", errors="ignore").strip()


def extract_resume_file(file) -> str:
    """Dispatch to the right extractor based on file extension."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(file)
    if name.endswith(".docx"):
        return extract_docx(file)
    if name.endswith(".txt"):
        return extract_txt(file)
    raise ValueError(f"Unsupported file type: {name}")


# --- Gemini helper -----------------------------------------------------

GEMINI_MAX_RETRIES = 3       # how many attempts before giving up
GEMINI_RETRY_SECONDS = 5     # wait between retries - Gemini overload errors are usually brief


def _call_gemini_json(prompt: str, system_instruction: str) -> Any:
    """
    Call Gemini with JSON-mode generation and parse the result.
    Retries a few times on transient server-side overload (503) errors,
    since those are almost always brief. Returns None (rather than
    raising) if every attempt fails, so callers can fall back gracefully
    instead of crashing the whole search.
    """
    if not GEMINI_API_KEY:
        return None

    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                },
            )
            return json.loads(response.text)
        except Exception as exc:  # noqa: BLE001 - surface as a soft failure
            last_error = exc
            # Only worth retrying on server-side overload/unavailability,
            # not on things like a bad API key or a malformed request.
            is_retryable = "503" in str(exc) or "UNAVAILABLE" in str(exc)
            if is_retryable and attempt < GEMINI_MAX_RETRIES:
                time.sleep(GEMINI_RETRY_SECONDS)
                continue
            break

    st.session_state.setdefault("ai_warnings", []).append(str(last_error))
    return None


# --- AI: understand the candidate -----------------------------------------------------


def understand_profile(resume_text: str, preferences: dict) -> dict:
    """
    Ask Gemini to build a structured candidate profile from the resume:
    explicit skills, hidden/transferable skills, likely job titles,
    experience level, and skill categories (technical, soft, management,
    research, writing, teaching, creative).
    """
    system_instruction = (
        "You are an expert career analyst. Given a candidate's resume, "
        "extract a structured profile. Infer skills that are implied but "
        "not explicitly stated - for example, an engineer who mentions "
        "tender documents likely knows procurement and vendor management; "
        "someone who wrote technical reports has strong technical writing "
        "skills. Never invent employers, job titles, or dates that aren't "
        "supported by the resume - only infer skills and capabilities. "
        "Respond with JSON only, matching this exact shape: "
        '{"skills": [], "hidden_skills": [], "transferable_skills": [], '
        '"industries": [], "likely_job_titles": [], "experience_level": "", '
        '"technical_skills": [], "soft_skills": [], "management_skills": [], '
        '"research_skills": [], "writing_skills": [], "teaching_skills": [], '
        '"creative_skills": []}'
    )
    prompt = (
        f"RESUME:\n{resume_text}\n\n"
        f"OPTIONAL PREFERENCES:\n{json.dumps(preferences, default=str)}\n\n"
        "Build the candidate profile as instructed."
    )
    result = _call_gemini_json(prompt, system_instruction)
    return result or {}


def generate_search_queries(profile: dict, preferences: dict) -> list[str]:
    """
    Ask Gemini to turn the profile into several diversified job-title
    search queries (not just the candidate's current title), tailored to
    their inferred skills and stated preferences.
    """
    system_instruction = (
        "You generate job search queries for a job board search engine. "
        "Given a candidate profile, produce a diverse list of job titles "
        "or short search phrases the candidate is genuinely qualified for. "
        "The profile has several distinct skill clusters (e.g. technical/"
        "engineering, AI/ML, writing/documentation, education/curriculum, "
        "creative/design, research) - generate at least one or two queries "
        "PER distinct cluster you find, not just queries clustered around "
        "the candidate's most recent job title. Include adjacent and "
        "transferable roles a generalist recruiter might miss (e.g. "
        "'Technical Writer - Renewable Energy', 'Instructional Designer "
        "STEM', 'AI Prompt Engineer', 'Grid/Power Systems Analyst', "
        "'Educational Content Developer'), not just generic titles. "
        "Respond with JSON only: "
        '{"queries": ["...", "..."]}'
    )
    prompt = (
        f"CANDIDATE PROFILE:\n{json.dumps(profile, default=str)}\n\n"
        f"PREFERENCES:\n{json.dumps(preferences, default=str)}\n\n"
        f"Return up to {MAX_QUERIES} queries, covering every distinct skill "
        "cluster in the profile rather than concentrating on one."
    )
    result = _call_gemini_json(prompt, system_instruction)
    if result and isinstance(result.get("queries"), list):
        return result["queries"][:MAX_QUERIES]

    # Fallback: if AI query generation fails, at least search on the
    # candidate's own likely job titles so the app still works.
    fallback = profile.get("likely_job_titles") or []
    return fallback[:MAX_QUERIES] if fallback else [""]


def _rank_batch(jobs_batch: list[dict], profile: dict, preferences: dict) -> list[dict]:
    """Score, explain, and flag missing skills for one batch of jobs."""
    candidate_country = preferences.get("country") or "an unspecified country"
    system_instruction = (
        "You are an expert recruiter. Score how well each job matches the "
        "candidate profile. For every job, return a match_score (0-100), "
        "a short why_matches explanation (a few bullet-style phrases of "
        "what fits), and a missing_skills list (skills the job likely "
        f"requires that the candidate's profile doesn't show).\n\n"
        f"IMPORTANT - work-location eligibility: the candidate is based in "
        f"{candidate_country} and needs a role they can legally and "
        "practically do remotely from there. If a job's location/"
        "description clearly restricts eligibility to a different specific "
        "country or region (e.g. 'must reside in the US', 'UK-based only', "
        "'Canadian work authorization required') and does not mention "
        f"{candidate_country} or open worldwide eligibility, score it low "
        "(under 20) and add an item like 'Not eligible from "
        f"{candidate_country} - requires <that place>' to missing_skills. "
        "If eligibility is unclear or unrestricted, judge on skills/fit "
        "as normal and do not penalize for it.\n\n"
        "Respond with JSON only, matching this exact shape: "
        '{"results": [{"index": 0, "match_score": 0, "why_matches": "", '
        '"missing_skills": []}]}. '
        "The index field must match the 0-based position of the job in "
        "the JOBS list you were given."
    )
    compact_jobs = [
        {
            "index": i,
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "location": job.get("location", ""),
            "remote": job.get("remote", False),
            "description": (job.get("description") or "")[:300],
        }
        for i, job in enumerate(jobs_batch)
    ]
    prompt = (
        f"CANDIDATE PROFILE:\n{json.dumps(profile, default=str)}\n\n"
        f"PREFERENCES:\n{json.dumps(preferences, default=str)}\n\n"
        f"JOBS:\n{json.dumps(compact_jobs, default=str)}"
    )
    result = _call_gemini_json(prompt, system_instruction)

    scored_by_index = {}
    if result and isinstance(result.get("results"), list):
        for entry in result["results"]:
            idx = entry.get("index")
            if isinstance(idx, int) and 0 <= idx < len(jobs_batch):
                scored_by_index[idx] = entry

    ranked = []
    for i, job in enumerate(jobs_batch):
        entry = scored_by_index.get(i, {})
        job = {**job}
        job["match_score"] = entry.get("match_score", 0)
        job["why_matches"] = entry.get("why_matches") or (
            "AI scoring was unavailable for this batch - showing unscored." if not result else ""
        )
        job["missing_skills"] = entry.get("missing_skills", [])
        ranked.append(job)
    return ranked


def rank_and_explain_jobs(jobs: list[dict], profile: dict, preferences: dict) -> list[dict]:
    """
    Ask Gemini to score every collected job against the profile, in
    batches of MAX_JOBS_TO_RANK at a time, so a large result set still
    gets every job scored instead of leaving most of them unranked.
    Jobs beyond TOTAL_JOB_CAP are dropped from ranking entirely (with a
    UI warning) to keep the number of Gemini calls per search bounded.
    """
    if not jobs:
        return []

    jobs_to_rank = jobs[:TOTAL_JOB_CAP]
    if len(jobs) > TOTAL_JOB_CAP:
        st.session_state.setdefault("ai_warnings", []).append(
            f"{len(jobs) - TOTAL_JOB_CAP} extra jobs were found beyond the "
            f"{TOTAL_JOB_CAP}-job ranking cap and were left out entirely. "
            "Narrow your queries or preferences to see more of the long tail."
        )

    ranked: list[dict] = []
    for start in range(0, len(jobs_to_rank), MAX_JOBS_TO_RANK):
        batch = jobs_to_rank[start : start + MAX_JOBS_TO_RANK]
        ranked.extend(_rank_batch(batch, profile, preferences))
    return ranked


# --- Job collection -----------------------------------------------------


def collect_jobs(queries: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Run every source for every generated query, de-duplicate the combined
    results (by title + company), and return them alongside the
    per-source transparency stats.
    """
    all_jobs: list[dict] = []
    transparency: dict[str, dict] = {}

    for query in queries:
        source_results = search_all_sources(query, RESULTS_PER_SOURCE)
        for source_result in source_results:
            name = source_result["name"]
            # Merge stats across queries for the same source.
            stats = transparency.setdefault(
                name, {"name": name, "status": source_result["status"], "jobs_found": 0, "search_time": 0.0}
            )
            stats["jobs_found"] += source_result["jobs_found"]
            stats["search_time"] += source_result["search_time"]
            if source_result["status"] == "ok":
                stats["status"] = "ok"
            all_jobs.extend(source_result["jobs"])

    # De-duplicate by (title, company) - keeps the first occurrence.
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        key = (job.get("title", "").strip().lower(), job.get("company", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique_jobs.append(job)

    return unique_jobs, list(transparency.values())


# --- Country-eligibility heuristic -----------------------------------------------------

# Common phrases postings use to restrict who can be hired, paired with a
# capture of the country/region named. This is a plain keyword heuristic
# (not AI) used as a quick, cheap double-check alongside the AI ranking's
# own eligibility judgment - it won't catch everything, but it's a fast
# first pass over obvious cases like "must be authorized to work in the US".
_RESTRICTION_PATTERN = re.compile(
    r"(?:must (?:be |)(?:based|residing|located) in|"
    r"authorized to work in|work authorization in|"
    r"citizens? of|residents? of|based in)\s+"
    r"(the\s+)?([A-Za-z][A-Za-z .]{2,25})",
    re.IGNORECASE,
)


def _looks_country_restricted(text: str, candidate_country: str) -> bool:
    """
    Heuristic check: does this text explicitly restrict eligibility to a
    country other than the candidate's, without mentioning the
    candidate's country or worldwide/global eligibility nearby?
    """
    if not text or not candidate_country:
        return False
    text_lower = text.lower()
    if candidate_country.lower() in text_lower:
        return False
    if any(word in text_lower for word in ["worldwide", "anywhere", "global", "any location", "any country"]):
        return False
    return bool(_RESTRICTION_PATTERN.search(text))


# --- Streamlit UI -----------------------------------------------------

st.set_page_config(page_title="AI Job Hunter", page_icon="🧭", layout="wide")

st.title("🧭 AI Job Hunter")
st.write("Find AI-ranked jobs using your complete professional profile.")

if not GEMINI_API_KEY:
    st.warning(
        "GEMINI_API_KEY is not set. Add it to a .env file (or Streamlit "
        "Secrets) before searching - the AI profile, query generation, "
        "and ranking steps all depend on it."
    )

# --- Section 1: Master Resume -----------------------------------------------------

st.header("1. Master Resume")
resume_input_mode = st.radio("How would you like to provide your resume?", ["Paste text", "Upload file"])

resume_text = ""
if resume_input_mode == "Paste text":
    resume_text = st.text_area(
        "Paste your complete professional background",
        height=250,
        placeholder="Everything you can do: roles, responsibilities, tools, achievements...",
    )
else:
    uploaded_resume = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    if uploaded_resume is not None:
        try:
            resume_text = extract_resume_file(uploaded_resume)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read that file: {exc}")

# --- Section 2: Optional Preferences -----------------------------------------------------

st.header("2. Optional Preferences")
with st.expander("Set preferences (all optional)", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        work_mode = st.multiselect("Work mode", ["Remote", "Hybrid", "On-site"])
        employment_type = st.multiselect(
            "Employment type", ["Full-time", "Part-time", "Freelance", "Contract", "Internship", "Gig work"]
        )
        experience_level = st.multiselect("Level", ["Entry level", "Senior"])
        company_type = st.multiselect("Company type", ["Startup", "Large Company", "NGO", "Government"])

    with col2:
        country = st.text_input("Country preference", value="Pakistan")
        min_salary = st.text_input("Minimum salary")
        max_salary = st.text_input("Maximum salary")
        visa_sponsorship = st.checkbox("Requires visa sponsorship")
        travel_willingness = st.text_input("Travel willingness")

    with col3:
        preferred_industries = st.text_input("Preferred industries (comma-separated)")
        preferred_technologies = st.text_input("Preferred technologies (comma-separated)")
        keywords_to_avoid = st.text_input("Keywords to avoid (comma-separated)")
        max_years_experience = st.text_input("Maximum years of experience")
        languages = st.text_input("Languages")
        availability = st.text_input("Availability")

preferences = {
    "work_mode": work_mode,
    "employment_type": employment_type,
    "experience_level": experience_level,
    "company_type": company_type,
    "country": country,
    "min_salary": min_salary,
    "max_salary": max_salary,
    "visa_sponsorship": visa_sponsorship,
    "travel_willingness": travel_willingness,
    "preferred_industries": [s.strip() for s in preferred_industries.split(",") if s.strip()],
    "preferred_technologies": [s.strip() for s in preferred_technologies.split(",") if s.strip()],
    "keywords_to_avoid": [s.strip() for s in keywords_to_avoid.split(",") if s.strip()],
    "max_years_experience": max_years_experience,
    "languages": languages,
    "availability": availability,
}

# --- Find Jobs -----------------------------------------------------

st.header("3. Search")
find_clicked = st.button("Find Jobs", type="primary")

if find_clicked:
    st.session_state["ai_warnings"] = []

    if not resume_text.strip():
        st.error("Please paste or upload a Master Resume first.")
    elif not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set - AI understanding and ranking can't run without it.")
    else:
        with st.spinner("Understanding your profile..."):
            profile = understand_profile(resume_text, preferences)

        with st.spinner("Generating search queries..."):
            queries = generate_search_queries(profile, preferences)

        with st.spinner(f"Searching {len(SOURCES)} sources across {len(queries)} queries..."):
            start_time = time.time()
            jobs, transparency = collect_jobs(queries)
            total_search_time = round(time.time() - start_time, 2)

        with st.spinner("Ranking and explaining matches..."):
            ranked_jobs = rank_and_explain_jobs(jobs, profile, preferences)
            ranked_jobs.sort(key=lambda j: j.get("match_score", 0), reverse=True)

        st.session_state["profile"] = profile
        st.session_state["queries"] = queries
        st.session_state["transparency"] = transparency
        st.session_state["total_search_time"] = total_search_time
        st.session_state["ranked_jobs"] = ranked_jobs

# --- Results -----------------------------------------------------

if "ranked_jobs" in st.session_state:
    for warning in st.session_state.get("ai_warnings", []):
        st.warning(f"AI step had an issue and used a fallback: {warning}")

    profile = st.session_state["profile"]
    queries = st.session_state["queries"]
    transparency = st.session_state["transparency"]
    ranked_jobs = st.session_state["ranked_jobs"]

    # --- Candidate profile summary -----------------------------------------------------
    st.header("Candidate Profile")
    with st.expander("What the AI understood about you", expanded=False):
        st.json(profile)
        st.write("**Search queries generated:** " + ", ".join(queries))

    # --- Search transparency -----------------------------------------------------
    st.header("Search Transparency")
    st.write(f"Search completed in {st.session_state['total_search_time']} seconds.")
    transparency_df = pd.DataFrame(transparency)
    if not transparency_df.empty:
        st.dataframe(transparency_df, use_container_width=True)

    # --- Results table with filters -----------------------------------------------------
    st.header("Results")

    if not ranked_jobs:
        st.info("No jobs found. Try broadening your preferences or resume details.")
    else:
        results_df = pd.DataFrame(ranked_jobs)

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            remote_filter = st.selectbox("Remote", ["Any", "Remote only", "Non-remote only"])
        with filter_col2:
            source_filter = st.multiselect("Platform", sorted(results_df["source"].dropna().unique().tolist()))
        with filter_col3:
            company_filter = st.text_input("Company contains")
        with filter_col4:
            min_score = st.slider("Minimum match score", 0, 100, 0)

        hide_restricted = st.checkbox(
            f"Hide postings that explicitly restrict work to a country other than "
            f"{preferences.get('country') or 'your preference'}",
            value=bool(preferences.get("country")),
        )

        filtered_df = results_df.copy()
        if remote_filter == "Remote only":
            filtered_df = filtered_df[filtered_df["remote"] == True]  # noqa: E712
        elif remote_filter == "Non-remote only":
            filtered_df = filtered_df[filtered_df["remote"] == False]  # noqa: E712
        if source_filter:
            filtered_df = filtered_df[filtered_df["source"].isin(source_filter)]
        if company_filter:
            filtered_df = filtered_df[filtered_df["company"].str.contains(company_filter, case=False, na=False)]
        if hide_restricted and preferences.get("country"):
            filtered_df = filtered_df[
                ~filtered_df.apply(
                    lambda row: _looks_country_restricted(
                        f"{row.get('location', '')} {row.get('why_matches', '')} {row.get('missing_skills', '')}",
                        preferences["country"],
                    ),
                    axis=1,
                )
            ]
        filtered_df = filtered_df[filtered_df["match_score"] >= min_score]
        filtered_df = filtered_df.sort_values("match_score", ascending=False)

        display_columns = [
            "match_score", "title", "company", "location", "remote", "salary",
            "employment_type", "posting_date", "source", "why_matches",
            "missing_skills", "apply_url",
        ]
        st.dataframe(filtered_df[display_columns], use_container_width=True)

        # --- Export -----------------------------------------------------
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Export CSV",
                data=filtered_df[display_columns].to_csv(index=False),
                file_name="ai_job_hunter_results.csv",
                mime="text/csv",
            )
        with export_col2:
            st.download_button(
                "Export JSON",
                data=filtered_df[display_columns].to_json(orient="records", indent=2),
                file_name="ai_job_hunter_results.json",
                mime="application/json",
            )

        # --- AI Recommendations -----------------------------------------------------
        st.header("AI Recommendations")

        def _top(df: pd.DataFrame, by: str, ascending: bool = False, n: int = 5) -> pd.DataFrame:
            if by not in df.columns:
                return df.head(0)
            return df.sort_values(by, ascending=ascending).head(n)

        rec_tabs = st.tabs(
            ["Top 10", "Best Remote", "Highest Salary", "Fastest Applications", "Best Freelance"]
        )
        with rec_tabs[0]:
            st.dataframe(_top(filtered_df, "match_score", n=10)[display_columns], use_container_width=True)
        with rec_tabs[1]:
            remote_df = filtered_df[filtered_df["remote"] == True]  # noqa: E712
            st.dataframe(_top(remote_df, "match_score")[display_columns], use_container_width=True)
        with rec_tabs[2]:
            salaried_df = filtered_df[filtered_df["salary"].astype(str).str.strip() != ""]
            st.dataframe(salaried_df[display_columns].head(10), use_container_width=True)
        with rec_tabs[3]:
            easy_apply_df = filtered_df[filtered_df["source"].isin(["RemoteOK", "Working Nomads", "We Work Remotely"])]
            st.dataframe(_top(easy_apply_df, "match_score")[display_columns], use_container_width=True)
        with rec_tabs[4]:
            freelance_df = filtered_df[filtered_df["employment_type"].astype(str).str.contains("freelance|contract", case=False, na=False)]
            st.dataframe(_top(freelance_df, "match_score")[display_columns], use_container_width=True)
