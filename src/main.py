"""Apify actor: Conference Signals (GLiClass). Reads a Press Monitor Google Sheet, writes a Conference_Signals tab + dataset."""
import json, re, requests, pandas as pd
from apify import Actor
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from . import signals as S

API = "https://sheets.googleapis.com/v4/spreadsheets"

class Sheet:
    def __init__(self, sid, sa_json):
        info = json.loads(sa_json)
        self.cred = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        self.sid, self.email = sid, info.get("client_email", "?")
    def _h(self):
        if not self.cred.valid: self.cred.refresh(Request())
        return {"Authorization": f"Bearer {self.cred.token}"}
    def tabs(self):
        r = requests.get(f"{API}/{self.sid}?fields=sheets.properties", headers=self._h(), timeout=60)
        if r.status_code == 403: raise RuntimeError(f"The sheet is not shared with {self.email} - share it as Editor")
        r.raise_for_status(); return [s["properties"]["title"] for s in r.json()["sheets"]]
    def read(self, tab):
        r = requests.get(f"{API}/{self.sid}/values/'{tab}'", headers=self._h(), timeout=120); r.raise_for_status()
        v = r.json().get("values", []); 
        if not v: return pd.DataFrame()
        h = v[0]; return pd.DataFrame([row + [""] * (len(h) - len(row)) for row in v[1:]], columns=h)
    def write(self, tab, df):
        if tab not in self.tabs():
            requests.post(f"{API}/{self.sid}:batchUpdate", headers=self._h(), json={"requests": [{"addSheet": {"properties": {"title": tab}}}]}, timeout=60).raise_for_status()
        requests.post(f"{API}/{self.sid}/values/'{tab}':clear", headers=self._h(), json={}, timeout=60).raise_for_status()
        values = [list(df.columns)] + df.astype(str).values.tolist()
        requests.put(f"{API}/{self.sid}/values/'{tab}'!A1?valueInputOption=RAW", headers=self._h(), json={"values": values}, timeout=120).raise_for_status()

def pick_tabs(all_tabs, spec):
    if spec.strip().lower() == "all": return [t for t in all_tabs if re.match(r"(Delivery_|Monitor_Review|Monitor_Social|Team_Review)", t)]
    if spec.strip().lower() == "latest":
        d = sorted(t for t in all_tabs if re.match(r"Delivery_\d{4}-\d{2}-\d{2}", t))
        return d[-1:] + [t for t in ("Monitor_Review", "Monitor_Social", "Team_Review") if t in all_tabs]
    return [t.strip() for t in spec.split(",") if t.strip() in all_tabs]

async def main():
    async with Actor:
        inp = await Actor.get_input() or {}
        sh = Sheet(inp["spreadsheetId"], inp["serviceAccountJson"])
        tabs = pick_tabs(sh.tabs(), inp.get("tabs", "latest")); Actor.log.info(f"reading tabs: {tabs}")
        rows = []
        for t in tabs:
            d = sh.read(t).fillna("")
            if "Review_Reason" in d:
                d = d[~d.Review_Reason.str.contains(r"no medical|namesake|describes the person|not the same person|nothing from the KOL", case=False, regex=True)]
            for _, r in d.iterrows():
                text = r.get("Press_Content") or r.get("Text") or r.get("Post_Text") or r.get("Snippet") or ""
                rows.append(dict(tab=t, KOL_ID=r.get("KOL_ID", ""), KOL_Name=r.get("KOL_Name", ""), URL=r.get("URL", ""), Title=r.get("Topic") or r.get("Title", ""),
                                 Date=r.get("Date", ""), Platform=r.get("Scientific_Platform") or r.get("Site", ""), text=str(text)))
        cands = list(S.candidates(rows)); Actor.log.info(f"{len(rows)} rows | {len(cands)} sentences name a KOL next to a conference word")
        # Haiku sorts each candidate sentence (30-Sep: GLiClass scored everything ~1.0, DeBERTa zero-shot timed out on Apify's CPU)
        key = inp.get("anthropicApiKey") or ""
        use_ai = bool(inp.get("useModel", True) and key)
        SYSTEM = ("You classify text from a news article, post or web page that names a medical doctor (the KOL), for a medical-conference intelligence team. "
                  "Answer ONLY JSON: {\"signal\":\"Presentation\"|\"Poster / abstract\"|\"Session role (speaker / chair / moderator)\"|\"Upcoming meeting\"|\"Award at a meeting\"|\"Not conference\","
                  "\"conference\":\"<meeting name or empty>\",\"confidence\":<0-1>}. Use Not conference for press conferences, the US Congress or legislative hearings, "
                  "sports, law, tax, business or political meetings, anything not about a medical/scientific meeting, or when the named person is clearly not a physician or scientist. "
                  "Use Award at a meeting ONLY when an award or honour is actually given; a registration or invitation notice is Upcoming meeting. Put the meeting name in conference, or empty if none is named.")
        def ask(kol, text):
            for _ in range(3):
                try:
                    rr = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                                       headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                                       json={"model": "claude-haiku-4-5-20251001", "max_tokens": 150, "system": SYSTEM,
                                             "messages": [{"role": "user", "content": f"KOL: {kol}\nTEXT: {text[:1500]}"}]})
                    t = "".join(c.get("text", "") for c in rr.json().get("content", []))
                    mm = re.search(r"\{[\s\S]*\}", t)
                    if mm: return json.loads(mm.group(0))
                except Exception as ex:
                    Actor.log.warning(f"AI check retry: {str(ex)[:80]}")
            return {"signal": "Not conference", "conference": "", "confidence": 0}
        if use_ai:
            for probe in ("Dr. Smith presented the phase 3 trial results at the ACR Convergence annual meeting.", "The Steelers released linebacker Smith before the roster deadline.", "Ask Congress to weigh in with CMS on the new payment rule."):
                Actor.log.info(f"self-check: {probe[:55]} -> {ask('Dr. Smith', probe)}")
        elif inp.get("useModel", True):
            Actor.log.warning("No Anthropic API key - keyword-only pass (rougher)")
        thr, out, seen = float(inp.get("threshold", 0.6)), [], set()
        for r, s, ctx in cands:
            conf = ", ".join(sorted({x.group(0) for x in S.CONF.finditer(s)}, key=str.lower))
            if use_ai:
                v = ask(r["KOL_Name"], ctx)
                try: c = float(v.get("confidence", 0) or 0)
                except Exception: c = 0.0
                if v.get("signal") in (None, "", "Not conference") or c < thr: continue
                kind, score = v.get("signal"), round(c, 2)
                cn = str(v.get("conference") or "").strip()
                if len(cn) > 3 and cn.lower().strip(".") not in ("not specified", "unknown", "none", "n/a", "drs", "aes", "unspecified"): conf = cn
            else:
                kind, score = "Keyword match", ""
            # one row per KOL and sentence, even when the same post/press release appears on several pages
            k2 = (r["KOL_ID"], re.sub(r"\W+", " ", re.sub(r"^\s*(?:\d+\s+\w+\s+ago|[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s*[Â·\-]\s*", "", s)).lower().strip()[:140])
            if k2 in seen: continue
            seen.add(k2)
            out.append({"KOL_ID": r["KOL_ID"], "KOL_Name": r["KOL_Name"], "Conference": conf, "Signal": kind, "Score": score, "Sentence": s,
                        "Article_Title": r["Title"], "Article_Date": r["Date"], "Platform": r["Platform"], "URL": r["URL"], "Source_Tab": r["tab"]})
        df = pd.DataFrame(out, columns=["KOL_ID", "KOL_Name", "Conference", "Signal", "Score", "Sentence", "Article_Title", "Article_Date", "Platform", "URL", "Source_Tab"])
        sh.write(inp.get("outputTab", "Conference_Signals"), df)
        if len(df): await Actor.push_data(df.to_dict("records"))
        await Actor.set_value("OUTPUT", {"tabsRead": tabs, "rows": len(rows), "candidates": len(cands), "signals": len(df)})
        Actor.log.info(f"{len(df)} conference signals -> tab {inp.get('outputTab', 'Conference_Signals')}")
