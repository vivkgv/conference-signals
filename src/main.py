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
        pipe = None
        if inp.get("useModel", True) and cands:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
            m = "knowledgator/gliclass-small-v1.0"
            pipe = ZeroShotClassificationPipeline(GLiClassModel.from_pretrained(m), AutoTokenizer.from_pretrained(m), classification_type="multi-label", device="cpu")
            for probe in ("Dr. Smith presented the phase 3 trial results at the ACR Convergence annual meeting.", "The Steelers released linebacker Smith before the roster deadline."):
                Actor.log.info(f"self-check: {probe[:60]} -> {[(x['label'][:28], round(x['score'], 3)) for x in pipe(probe, S.LABELS, threshold=0.0)[0]]}")
        thr, out, seen = float(inp.get("threshold", 0.45)), [], set()
        for r, s, ctx in cands:
            conf = ", ".join(sorted({x.group(0) for x in S.CONF.finditer(s)}, key=str.lower))
            if pipe:
                res = pipe(ctx, S.LABELS, threshold=0.0)[0]
                best = max((x for x in res if x["label"] != S.LABELS[5]), key=lambda x: x["score"]); none = next((x["score"] for x in res if x["label"] == S.LABELS[5]), 0)
                if best["score"] < thr or none > best["score"]: continue
                kind, score = S.SHORT[best["label"]], round(best["score"], 3)
            else:
                kind, score = "Keyword match", ""
            key = (r["KOL_ID"], r["URL"], s[:80])
            if key in seen: continue
            seen.add(key)
            out.append({"KOL_ID": r["KOL_ID"], "KOL_Name": r["KOL_Name"], "Conference": conf, "Signal": kind, "Score": score, "Sentence": s,
                        "Article_Title": r["Title"], "Article_Date": r["Date"], "Platform": r["Platform"], "URL": r["URL"], "Source_Tab": r["tab"]})
        df = pd.DataFrame(out, columns=["KOL_ID", "KOL_Name", "Conference", "Signal", "Score", "Sentence", "Article_Title", "Article_Date", "Platform", "URL", "Source_Tab"])
        sh.write(inp.get("outputTab", "Conference_Signals"), df)
        if len(df): await Actor.push_data(df.to_dict("records"))
        await Actor.set_value("OUTPUT", {"tabsRead": tabs, "rows": len(rows), "candidates": len(cands), "signals": len(df)})
        Actor.log.info(f"{len(df)} conference signals -> tab {inp.get('outputTab', 'Conference_Signals')}")
