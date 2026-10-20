#!/usr/bin/env python3
"""
Load a snapshot of TextData submissions into a LOCAL TextData instance.

Two steps, both optional:
  1. --pull : download a community's submissions from textdata.org
              (prompts for your textdata.org username/password; nothing is stored)
              and save them to --file.
  2. load   : log in to the local instance (creating the account if needed),
              create the local community if missing, and batch-upload --file into it.

Usage:
  python3 scripts/load_snapshot.py --pull          # pull CS410 from textdata.org, then load locally
  python3 scripts/load_snapshot.py                 # just (re)load data/cs410_snapshot.json

Local data is wiped whenever the Docker containers restart, so re-run this after each restart.
Standard library only, no pip install needed.
"""
import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

CS410_COMMUNITY_ID = "6a8e5f4ca8567d79c2f60b30"  # "CS410 Fall 2026" on textdata.org

# Local-only test account (exists only in your local Docker MongoDB).
LOCAL_USER = os.environ.get("LOCAL_TD_USER", "localtest")
LOCAL_EMAIL = os.environ.get("LOCAL_TD_EMAIL", "localtest@example.test")
LOCAL_PASSWORD = os.environ.get("LOCAL_TD_PASSWORD", "LocalTest-8080!")


def call(base, method, path, token=None, body=None, params=None):
    url = base.rstrip("/") + "/api/" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) TextData-snapshot-loader")
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", token)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"message": raw[:300]}


def login(base, username, password, verbose=False):
    status, resp = call(base, "POST", "users/login", body={"username": username, "password": password})
    if status != 200 and verbose:
        print(f"  login returned HTTP {status}: {resp.get('message')}")
    return resp.get("token") if status == 200 else None


# ---------- pull from textdata.org ----------
def pull(remote, community_id, out_file):
    print(f"Pulling community {community_id} from {remote}")
    user = os.environ.get("TEXTDATA_USERNAME") or input("textdata.org username or email: ").strip()
    pw = os.environ.get("TEXTDATA_PASSWORD") or getpass.getpass("textdata.org password (not shown, not saved): ")
    token = login(remote, user, pw, verbose=True)
    if not token:
        sys.exit("Login to textdata.org failed (see reason above). Run again to retry.")

    status, resp = call(remote, "GET", "search/website", token, params={
        "query": "", "community": community_id, "source": "website_community_page", "own_submissions": "False"})
    if status != 200:
        sys.exit(f"Search failed ({status}): {resp.get('message')}")
    search_id = resp["search_id"]
    print(f"  {resp.get('total_num_results')} submissions found, exporting...")

    status, resp = call(remote, "GET", "search/export", token, params={"search_id": search_id})
    if status != 200:
        sys.exit(f"Export failed ({status}): {resp.get('message')}")

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"source": remote, "community_id": community_id, "data": resp.get("data", [])}, f, indent=1)
    print(f"  saved {len(resp.get('data', []))} submissions to {out_file}")


# ---------- load into local ----------
def to_batch_items(snapshot):
    rows = snapshot["data"] if isinstance(snapshot, dict) else snapshot
    items, skipped = [], 0
    for r in rows:
        title = (r.get("title") or r.get("explanation") or "").strip()[:1000]
        if not title or title.startswith("http"):
            skipped += 1
            continue
        src = r.get("source_url") or ""
        if "textdata.org" in src:  # internal link, not an external source
            src = ""
        items.append({"title": title,
                      "description": r.get("description") or r.get("highlighted_text") or "",
                      "source_url": src})
    return items, skipped


def ensure_local_user(base):
    token = login(base, LOCAL_USER, LOCAL_PASSWORD)
    if token:
        return token
    status, resp = call(base, "POST", "users/createAccount",
                        body={"email": LOCAL_EMAIL, "username": LOCAL_USER, "password": LOCAL_PASSWORD})
    if status != 200:
        sys.exit(f"Could not create local account ({status}): {resp.get('message')}")
    print(f"Created local account '{LOCAL_USER}'")
    return resp["token"]


def ensure_community(base, token, name):
    def find():
        _, resp = call(base, "GET", "communities", token)
        for c in resp.get("community_info", []):
            if c["name"] == name:
                return c["community_id"]
    cid = find()
    if cid:
        return cid, False
    status, resp = call(base, "POST", "communities", token, body={"community_name": name, "community_description": "Snapshot loaded by scripts/load_snapshot.py"})
    if status != 200:
        sys.exit(f"Could not create community ({status}): {resp.get('message')}")
    return find(), True


def community_count(base, token, cid):
    _, resp = call(base, "GET", "search/website", token, params={
        "query": "", "community": cid, "source": "website_community_page", "own_submissions": "False"})
    return int(resp.get("total_num_results", 0) or 0)


def load(base, in_file, community_name, force, batch_size=25):
    if not os.path.exists(in_file):
        sys.exit(f"{in_file} not found. Run with --pull first.")
    with open(in_file, encoding="utf-8") as f:
        items, skipped = to_batch_items(json.load(f))
    print(f"Loading {len(items)} submissions into {base} ({skipped} skipped)")

    token = ensure_local_user(base)
    cid, created = ensure_community(base, token, community_name)
    print(f"  community '{community_name}' = {cid}{' (created)' if created else ''}")

    existing = 0 if created else community_count(base, token, cid)
    if existing and not force:
        print(f"  community already has {existing} submissions; skipping (use --force to add anyway)")
        return token, cid

    ok, failed = 0, 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        status, resp = call(base, "POST", "submission/batch", token,
                            body={"community": cid, "data": chunk, "anonymous": False})
        for k, v in resp.items():
            if k.startswith("Submission") and isinstance(v, dict):
                if v.get("status") == 200:
                    ok += 1
                else:
                    failed += 1
                    print(f"    failed: {chunk[int(k.split()[1])]['title'][:60]!r}: {v.get('message')}")
        print(f"  {min(i + batch_size, len(items))}/{len(items)} sent")
    print(f"Done: {ok} loaded, {failed} failed")
    return token, cid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pull", action="store_true", help="download the snapshot from textdata.org first")
    p.add_argument("--remote", default="https://textdata.org")
    p.add_argument("--remote-community", default=CS410_COMMUNITY_ID)
    p.add_argument("--base", default="http://localhost:8080", help="local TextData URL")
    p.add_argument("--file", default="data/cs410_snapshot.json")
    p.add_argument("--community", default="CS410 Fall 2026 (local)", help="local community name")
    p.add_argument("--force", action="store_true", help="upload even if the local community already has data")
    a = p.parse_args()

    if a.pull:
        pull(a.remote, a.remote_community, a.file)
    token, cid = load(a.base, a.file, a.community, a.force)
    n = community_count(a.base, token, cid)
    print(f"Check: local community now returns {n} submissions. Log in at {a.base} as '{LOCAL_USER}' / '{LOCAL_PASSWORD}'.")


if __name__ == "__main__":
    main()
