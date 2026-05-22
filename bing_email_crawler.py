#!/usr/bin/env python3
"""Simple Bing-based email crawler for a list of names.

Requires: set BING_API_KEY env var with a valid Bing Web Search key.
Reads a CSV with a `name` column (default: medicos_abn.csv) and writes found
emails to an output CSV.
"""
import argparse
import csv
import os
import random
import re
import time
import logging
from typing import List, Tuple, Set

import requests
from bs4 import BeautifulSoup

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def search_bing(api_key: str, query: str, count: int = 10) -> List[dict]:
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count}
    resp = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("webPages", {}).get("value", [])


def fetch_page(url: str, timeout: int = 10) -> Tuple[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EmailCrawler/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text, resp.headers.get("content-type", "")


def extract_emails_from_html(html: str) -> Set[str]:
    emails = set(EMAIL_RE.findall(html))
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a[href^=mailto]"):
        href = a.get("href", "")
        m = EMAIL_RE.search(href)
        if m:
            emails.add(m.group(0))
    return emails


def crawl_name(api_key: str, name: str, max_results: int = 8) -> List[Tuple[str, str]]:
    found = []
    query = f'"{name}" email OR contato OR "e-mail" OR "E-mail"'
    try:
        results = search_bing(api_key, query, count=max_results)
    except Exception as e:
        logging.warning("Bing search failed for '%s': %s", name, e)
        return found

    for r in results:
        url = r.get("url")
        snippet = r.get("snippet", "")
        emails = set(EMAIL_RE.findall(snippet))
        if emails:
            for e in emails:
                found.append((url or "", e))
            continue

        if not url:
            continue

        try:
            html, _ = fetch_page(url)
        except Exception:
            time.sleep(random.uniform(0.5, 1.5))
            continue

        emails = extract_emails_from_html(html)
        for e in emails:
            found.append((url, e))

        time.sleep(random.uniform(1.0, 2.0))

    return found


def read_names_from_csv(path: str) -> List[str]:
    names = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "name" in reader.fieldnames:
            for row in reader:
                n = row.get("name") or row.get("nome") or ""
                if n:
                    names.append(n.strip())
        else:
            f.seek(0)
            for row in csv.reader(f):
                if row:
                    names.append(row[0].strip())
    return names


def write_results(output_path: str, rows: List[Tuple[str, str, str]]):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "source_url", "email"])
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Bing email crawler for a list of names")
    parser.add_argument("--input", "-i", default="medicos_abn.csv", help="CSV input with names (column 'name' or first column)")
    parser.add_argument("--output", "-o", default="emails_found.csv", help="CSV output file")
    parser.add_argument("--bing-key", default=os.environ.get("BING_API_KEY"), help="Bing Search API key (or set BING_API_KEY)")
    parser.add_argument("--max-results", type=int, default=8, help="Max Bing results per query")
    args = parser.parse_args()

    if not args.bing_key:
        parser.error("Bing API key is required. Set BING_API_KEY or pass --bing-key")

    names = read_names_from_csv(args.input)
    logging.info("Loaded %d names from %s", len(names), args.input)

    all_rows = []
    seen = set()
    for name in names:
        logging.info("Searching for: %s", name)
        results = crawl_name(args.bing_key, name, max_results=args.max_results)
        for url, email in results:
            key = (name, email)
            if key in seen:
                continue
            seen.add(key)
            all_rows.append((name, url, email))

    write_results(args.output, all_rows)
    logging.info("Wrote %d email records to %s", len(all_rows), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
