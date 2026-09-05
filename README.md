# AI Job Hunter

A Streamlit app that reads your complete professional background, builds
an AI-inferred candidate profile (including hidden and transferable
skills), searches multiple public job platforms, and ranks the results
against your profile with an explanation of why each job fits.

## Project structure

```
app.py            Streamlit UI + AI orchestration (Gemini)
job_sources.py    One function per job platform - never imports app.py
requirements.txt  Python dependencies
.env.example       Template for your Gemini API key
```

Dependency flow is strictly one-way: `app.py` → `job_sources.py`.
`job_sources.py` never imports `app.py`, so any source function can be
copied into another project unchanged.

## How job sources work

Every platform is exactly one function in `job_sources.py`, registered
in the `SOURCES` list at the bottom of that file. To add a new platform
later, you only need to:

1. Write one `search_<platform>(query, limit)` function that returns a
   list of job dicts in the shared shape (see `_make_job`).
2. Add one line to `SOURCES`.

No other file needs to change.

**Sources with real public APIs (implemented):** RemoteOK, Working
Nomads, We Work Remotely, Greenhouse (per-company boards), Lever
(per-company boards), Adzuna (requires a free key), Jooble (requires a
free key).

Adzuna and Jooble both need API keys - see "Get free API keys" below.
Adzuna is queried across several major markets per search (not one
country) and Jooble is queried with "remote" as the location, since
this app targets worldwide remote roles rather than one region.

**Sources that are stubbed on purpose:** LinkedIn and Indeed prohibit
scraping their pages, and Wellfound / Upwork / Freelancer require an
approved partner API key that isn't publicly self-serve. Rather than
scrape sites that disallow it, those functions return an empty result
with a clear status - the architecture is ready for you to plug in a
real implementation (an official API key, or a user-provided export)
the moment you have access, without touching any other file.

## 1. Get free API keys

- **Gemini** (required - powers all AI understanding/ranking): [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Adzuna** (optional job source): sign up at [developer.adzuna.com/signup](https://developer.adzuna.com/signup) - instant, gives you an App ID and App Key
- **Jooble** (optional job source): sign up at [jooble.org/api/about](https://jooble.org/api/about) - short form, key arrives by email; free tier is a 500-request lifetime quota
- **Tavily** (optional - open-web search for NGO/government/company career pages Google-style; Google's own Custom Search API is closed to new signups and shutting down entirely on 2027-01-01): sign up at [tavily.com](https://tavily.com) - recurring monthly free tier

The app runs fine with just the Gemini key - Adzuna/Jooble simply won't
return results (empty, no error) until their keys are set.

## 2. Run locally

```bash
cd ai-job-hunter
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your real GEMINI_API_KEY

streamlit run app.py
```

## 3. Run in Google Colab (via Cloudflare Tunnel)

**Cell 1 - upload your files**
Upload `app.py`, `job_sources.py`, and `requirements.txt` into the Colab
file browser (they must all sit in the same root folder, `/content`).

**Cell 2 - install dependencies**
```python
!pip install -r requirements.txt -q
```

**Cell 3 - set your API key**
```python
import os
os.environ["GEMINI_API_KEY"] = "your_real_gemini_api_key_here"
```

**Cell 4 - sanity check**
```python
!python -c "import app"
```

**Cell 5 - start Streamlit in the background**
```python
!nohup streamlit run app.py --server.port 8501 &> streamlit_log.txt &
```

**Cell 6 - confirm it launched**
```python
import time
time.sleep(5)
!cat streamlit_log.txt
```

**Cell 7 - download and start the Cloudflare Tunnel**
```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!nohup ./cloudflared-linux-amd64 tunnel --url http://localhost:8501 &> cloudflared_log.txt &
```

**Cell 8 - grab your public URL**
```python
import time
time.sleep(6)
!grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' cloudflared_log.txt | head -n 1
```

## Deploying on Streamlit Community Cloud

- Main file path: `app.py`
- Add your key under **Settings → Secrets**:
  ```toml
  GEMINI_API_KEY = "your_real_gemini_api_key_here"
  ```

## Notes on token/rate usage

Gemini is called three times per search: once to build the candidate
profile, once to generate search queries, and once to rank/explain the
combined job results. To keep that last call bounded regardless of how
many jobs are found, only the first `MAX_JOBS_TO_RANK` (30 by default,
set in `app.py`) are sent to the AI for scoring; any beyond that are
still shown in the results table, just without an AI match score.
Lower `MAX_JOBS_TO_RANK` or `MAX_QUERIES` in `app.py` if you're on a
rate-limited Gemini tier and hit errors with a lot of results.

## Future-ready

The one-function-per-source design and the shared job-dict shape mean
these can all be added later without restructuring anything: resume
tailoring, cover letter generation, interview prep, skill-gap analysis,
job tracking, application history, email alerts, saved searches, and
automatic daily searches.
