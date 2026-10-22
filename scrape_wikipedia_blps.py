"""
Wikipedia BLP scraper -- fetches real, live Wikipedia biography pages via the official
MediaWiki API (not raw HTML scraping, which the API explicitly recommends against for
bulk access -- this follows Wikipedia's own API etiquette: identify via User-Agent,
throttle requests, and use the structured API rather than hammering page URLs directly).

This module must be run somewhere with live internet access to en.wikipedia.org and
www.wikidata.org (e.g. Google Colab, or your own machine) -- it will NOT run inside a
network-restricted sandbox.

Usage:
    python scrape_wikipedia_blps.py --num-pages 5000 --out raw_blp_data.jsonl
"""
import requests
import time
import json
import re
import os
import argparse

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "PrivacyNERResearchBot/1.0 (student research project; contact: set-your-email-here)"}
REQUEST_DELAY = 1.5  # seconds between requests -- polite rate limiting per Wikipedia API
    # etiquette. Set slightly above the documentation's bare minimum because platforms with
    # shared/pooled IP addresses (e.g. Google Colab) can trigger Wikipedia's rate limiter
    # faster than a single dedicated machine would, since the limiter is applied per-IP
    # across ALL users sharing that IP, not just this script's own request rate.
CACHE_PATH = "scrape_cache.json"


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def get_category_members(category="Category:Living_people", limit=500, cmcontinue=None):
    """Fetch a page of article titles from a Wikipedia category via the API's
    categorymembers list. Category:Living_people has ~950,000 members, so this is
    paginated (500 per request, following the 'cmcontinue' token)."""
    params = {
        "action": "query", "list": "categorymembers", "cmtitle": category,
        "cmlimit": limit, "cmnamespace": 0, "format": "json",
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue
    def _do_request():
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    data = fetch_with_retry(_do_request)
    members = data.get("query", {}).get("categorymembers", [])
    titles = [m["title"] for m in members]
    next_token = data.get("continue", {}).get("cmcontinue")
    return titles, next_token


def collect_candidate_titles(num_pages, cache):
    """Collect up to num_pages candidate BLP titles by paginating Category:Living_people."""
    titles = list(cache.get("_titles", []))
    cmcontinue = cache.get("_cmcontinue")
    while len(titles) < num_pages:
        batch, cmcontinue = get_category_members(cmcontinue=cmcontinue)
        if not batch:
            break
        titles.extend(t for t in batch if t not in titles)
        cache["_titles"] = titles
        cache["_cmcontinue"] = cmcontinue
        save_cache(cache)
        print(f"  collected {len(titles)} candidate titles so far...")
        time.sleep(REQUEST_DELAY)
        if cmcontinue is None:
            break
    return titles[:num_pages]


def fetch_page_extract_and_wikitext(title):
    """Fetch (a) the clean plain-text intro extract and (b) the raw wikitext (for infobox
    parsing) for one article, plus its Wikidata QID (for a live is-alive check), in a single
    API call using multiple props."""
    params = {
        "action": "query", "titles": title, "format": "json",
        "prop": "extracts|revisions|pageprops",
        "exintro": 1, "explaintext": 1,
        "rvprop": "content", "rvslots": "main",
        "ppprop": "wikibase_item",
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        return None
    extract = page.get("extract", "")
    wikitext = ""
    revs = page.get("revisions", [])
    if revs:
        wikitext = revs[0].get("slots", {}).get("main", {}).get("*", "")
    qid = page.get("pageprops", {}).get("wikibase_item")
    page_id = page.get("pageid")
    return {"page_id": page_id, "title": title, "extract": extract, "wikitext": wikitext, "qid": qid}


def parse_date_template(val):
    """Wikipedia commonly wraps birth/death dates in templates like
    {{birth date and age|1985|3|14}} or {{death date and age|2020|1|5|1985|3|14}}.
    Extract the first (year, month, day) triple found rather than blanket-stripping the
    template, since date fields are critical for BLP identification and DATE_OF_BIRTH."""
    m = re.search(r"\{\{\s*(?:birth|death)[\s_-]*date[^|}]*\|(?:df=[^|]*\|)?\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})", val, re.IGNORECASE)
    if m:
        year, month, day = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def parse_infobox(wikitext):
    """Extract infobox field:value pairs from raw wikitext using a bracket-depth-aware
    regex parser (handles nested {{...}} templates inside field values, which a naive
    regex would break on)."""
    m = re.search(r"\{\{\s*Infobox", wikitext, re.IGNORECASE)
    if not m:
        return {}
    start = m.start()
    depth = 0
    end = start
    for i in range(start, len(wikitext)):
        if wikitext[i:i+2] == "{{":
            depth += 1
        elif wikitext[i:i+2] == "}}":
            depth -= 1
            if depth == 0:
                end = i + 2
                break
    infobox_text = wikitext[start:end]
    fields = {}
    for line in infobox_text.split("\n"):
        m2 = re.match(r"^\s*\|\s*([a-zA-Z_0-9]+)\s*=\s*(.*)$", line)
        if m2:
            key, val = m2.group(1).strip().lower(), m2.group(2).strip()
            date_val = parse_date_template(val) if "date" in key else None
            if date_val:
                fields[key] = date_val
                continue
            val = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", val)  # [[link|text]] -> text
            val = re.sub(r"\{\{[^}]*\}\}", "", val)  # strip remaining simple inline templates
            val = re.sub(r"<[^>]+>", "", val).strip()  # strip HTML tags
            if val:
                fields[key] = val
    return fields


def is_living_via_wikidata(qid):
    """Live check: query Wikidata directly for property P570 (date of death). Absence of
    P570 is a much stronger, CURRENT signal of living status than any static-dataset proxy,
    since it reflects Wikidata's live state rather than a snapshot from years ago."""
    if not qid:
        return None  # unknown -- caller should decide how to treat this
    params = {"action": "wbgetclaims", "entity": qid, "property": "P570", "format": "json"}
    def _do_request():
        resp = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    try:
        data = fetch_with_retry(_do_request)
    except requests.RequestException:
        return None  # genuinely unreachable after retries -- caller treats as unverified
    claims = data.get("claims", {})
    return "P570" not in claims or len(claims.get("P570", [])) == 0


def fetch_with_retry(fetch_fn, *args, max_retries=5, **kwargs):
    """Wrap a fetch call with exponential backoff on 429 (Too Many Requests) responses,
    respecting the server's Retry-After header when present. This is standard, polite HTTP
    client behaviour and is what was missing before: a failed request must still wait before
    the next attempt, or failures cascade into a rapid-fire burst that gets rate-limited even
    harder (exactly what happens if the delay is skipped on the failure path)."""
    delay = REQUEST_DELAY
    for attempt in range(max_retries):
        try:
            return fetch_fn(*args, **kwargs)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"    429 rate-limited, waiting {wait:.1f}s before retry "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                delay *= 2  # exponential backoff for subsequent retries
                continue
            raise
    raise requests.RequestException(f"Exceeded max retries ({max_retries}) due to repeated 429s")


def scrape(num_pages, out_path):
    cache = load_cache()
    print(f"Collecting candidate BLP titles from Category:Living_people (target: {num_pages})...")
    titles = collect_candidate_titles(num_pages * 2, cache)  # over-collect: some will be excluded
    print(f"Collected {len(titles)} candidate titles. Fetching page content...")

    records = []
    fetched_cache = cache.get("_fetched", {})
    excluded = {"death_confirmed": 0, "no_qid": 0, "fetch_failed": 0, "too_short": 0, "unverifiable_status": 0}

    for i, title in enumerate(titles):
        if len(records) >= num_pages:
            break
        if title in fetched_cache:
            page_data = fetched_cache[title]
        else:
            try:
                page_data = fetch_with_retry(fetch_page_extract_and_wikitext, title)
            except requests.RequestException as e:
                print(f"  [{i}] fetch failed for '{title}': {e}")
                excluded["fetch_failed"] += 1
                time.sleep(REQUEST_DELAY)  # always wait, even on failure, so we never
                continue                    # accidentally burst-fire on consecutive errors
            time.sleep(REQUEST_DELAY)
            if page_data is None:
                excluded["fetch_failed"] += 1
                continue
            fetched_cache[title] = page_data
            cache["_fetched"] = fetched_cache
            if (i + 1) % 25 == 0:
                save_cache(cache)

        if not page_data or len(page_data["extract"].split()) < 25:
            excluded["too_short"] += 1
            continue

        qid = page_data.get("qid")
        if qid is None:
            excluded["no_qid"] += 1
            continue
        living = is_living_via_wikidata(qid)
        time.sleep(REQUEST_DELAY)
        if living is False:
            excluded["death_confirmed"] += 1
            continue
        if living is None:
            # Wikidata check genuinely failed even after retries -- excluded rather than
            # silently assumed-living, since we could not actually verify living status.
            excluded["unverifiable_status"] += 1
            continue

        infobox = parse_infobox(page_data["wikitext"])
        records.append({
            "wiki_id": str(page_data["page_id"]),
            "title": page_data["title"],
            "url": f"https://en.wikipedia.org/wiki/{page_data['title'].replace(' ', '_')}",
            "paragraph": page_data["extract"].strip(),
            "infobox": infobox,
            "qid": qid,
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(titles)}] scraped, {len(records)} valid BLP records so far "
                  f"(excluded: {excluded})")

    save_cache(cache)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Scraped {len(records)} BLP records -> {out_path}")
    print(f"Exclusions: {excluded}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pages", type=int, default=5000)
    parser.add_argument("--out", type=str, default="raw_blp_data.jsonl")
    args = parser.parse_args()
    scrape(args.num_pages, args.out)
