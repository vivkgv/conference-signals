"""Conference signals for the conference team - finds sentences in press rows that point to a KOL at a conference
(presentation, poster/abstract, session/panel, upcoming meeting, award at a meeting) using GLiClass (open source, CPU).
Input : any Press Monitor export (.xlsx) - reads Delivery_*, Monitor_Review, Monitor_Social tabs - or a client delivery file.
Output: <input>_Conference_Signals.xlsx with one row per signal: KOL, conference, sentence, signal type, score, URL.
Usage : python conference_signals.py "C:\\path\\UCB_Rheum_-_Press_Monitor.xlsx"   [--no-model]  [--model knowledgator/gliclass-small-v1.0]
"""
import re, sys, argparse, pandas as pd
from pathlib import Path

CONF = re.compile(r"\b(ACR|EULAR|ASCO|ESMO|ASH|EHA|SOHO|AACR|SABCS|AUA|EAU|ASTRO|AAN|AES|ECTRIMS|AANEM|MGFA|AAD|EADV|ADA|ENDO|IMS|ISTH|ATS|ERS|ESC|AHA|DDW|UEGW|ACG|APA|SLEuro|GRAPPA|SPARTAN|Convergence|annual meeting|congress|conference|symposium|summit|scientific sessions|world congress|grand rounds|webinar|workshop)\b", re.I)
CUE = re.compile(r"\b(present(?:ed|s|ing|ation)?|poster|abstract|oral|late-?breaking|session|panel|keynote|lecture|plenary|symposium|meeting|congress|conference|award(?:ed)?|honou?red|moderat(?:ed|or|es)|chair(?:ed|s)?|speak(?:er|s|ing)?|will present|to be presented)\b", re.I)
LABELS = ["the doctor presented research at a medical conference", "a poster or abstract presented at a meeting",
          "the doctor speaks, chairs or moderates a conference session or panel", "an upcoming conference or meeting",
          "an award or honour given at a meeting", "not about a conference"]
SHORT = {LABELS[0]: "Presentation", LABELS[1]: "Poster / abstract", LABELS[2]: "Session role (speaker / chair / moderator)",
         LABELS[3]: "Upcoming meeting", LABELS[4]: "Award at a meeting", LABELS[5]: "Not conference"}

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if 25 < len(s.strip()) < 600]

def name_bits(kol):
    s = re.sub(r"\([^)]*\)", " ", str(kol)).split(",")[0].split()
    return (s[0] if s else ""), (s[-1] if s else "")

def load_rows(path):
    x = pd.ExcelFile(path); out = []
    for sh in x.sheet_names:
        if not re.match(r"(Delivery_|Monitor_Review|Monitor_Social|Team_Review)", sh): continue
        d = x.parse(sh, dtype=str).fillna("")
        if "Review_Reason" in d:   # rows the engine judged to be another person are not this KOL's conference news
            d = d[~d.Review_Reason.str.contains(r"no medical|namesake|describes the person|not the same person|nothing from the KOL", case=False, regex=True)]
        for _, r in d.iterrows():
            text = r.get("Press_Content") or r.get("Text") or r.get("Post_Text") or r.get("Snippet") or ""
            out.append(dict(tab=sh, KOL_ID=r.get("KOL_ID", ""), KOL_Name=r.get("KOL_Name", ""), URL=r.get("URL", ""), Title=r.get("Topic") or r.get("Title", ""),
                            Date=r.get("Date", ""), Platform=r.get("Scientific_Platform") or r.get("Site", ""), text=str(text)))
    return out

def candidates(rows):
    """sentences that name the KOL (surname) AND carry a conference word or cue - keeps the model's work small"""
    for r in rows:
        first, last = name_bits(r["KOL_Name"])
        sents = sentences(r["Title"] + ". " + r["text"])
        for i, s in enumerate(sents):
            ctx = " ".join(sents[max(0, i - 1): i + 2])                     # the KOL may be named in the next sentence
            if re.search(r"press conference|news conference", s, re.I): continue
            if last and re.search(r"\b" + re.escape(last) + r"\b", ctx) and CONF.search(s) and CUE.search(s):
                yield r, s, ctx

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("xlsx"); ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--model", default="knowledgator/gliclass-small-v1.0"); ap.add_argument("--threshold", type=float, default=0.45); ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load_rows(a.xlsx); cands = list(candidates(rows))
    print(f"{len(rows)} rows read | {len(cands)} sentences name a KOL next to a conference word")
    pipe = None
    if not a.no_model:
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        model = GLiClassModel.from_pretrained(a.model); tok = AutoTokenizer.from_pretrained(a.model)
        pipe = ZeroShotClassificationPipeline(model, tok, classification_type="multi-label", device="cpu")
    out, seen = [], set()
    for r, s, ctx in cands:
        conf = sorted({m.group(0) for m in CONF.finditer(s)}, key=str.lower)
        if pipe:
            res = pipe(ctx, LABELS, threshold=0.0)[0]
            best = max((x for x in res if x["label"] != LABELS[5]), key=lambda x: x["score"])
            none = next((x["score"] for x in res if x["label"] == LABELS[5]), 0)
            if best["score"] < a.threshold or none > best["score"]: continue
            kind, score = SHORT[best["label"]], round(best["score"], 3)
        else:  # keyword-only mode (no model download) - rougher
            kind = ("Upcoming meeting" if re.search(r"\b(will|upcoming|to be held|register)\b", s, re.I) else
                    "Poster / abstract" if re.search(r"\b(poster|abstract)\b", s, re.I) else
                    "Session role (speaker / chair / moderator)" if re.search(r"\b(chair|moderat|panel|keynote|speaker)", s, re.I) else
                    "Award at a meeting" if re.search(r"\baward|honou?r", s, re.I) else "Presentation")
            score = ""
        key = (r["KOL_ID"], r["URL"], s[:80])
        if key in seen: continue
        seen.add(key)
        out.append({"KOL_ID": r["KOL_ID"], "KOL_Name": r["KOL_Name"], "Conference": ", ".join(conf), "Signal": kind, "Score": score,
                    "Sentence": s, "Article_Title": r["Title"], "Article_Date": r["Date"], "Platform": r["Platform"], "URL": r["URL"], "Source_Tab": r["tab"]})
    df = pd.DataFrame(out)
    dst = Path(a.out) if a.out else Path(a.xlsx).with_name(Path(a.xlsx).stem + "_Conference_Signals.xlsx")
    try: open(dst, "ab").close()
    except OSError: dst = Path.cwd() / (Path(a.xlsx).stem + "_Conference_Signals.xlsx")
    with pd.ExcelWriter(dst) as w:
        df.to_excel(w, sheet_name="Conference_Signals", index=False)
        if len(df): df.groupby(["Conference", "Signal"]).size().reset_index(name="Signals").sort_values("Signals", ascending=False).to_excel(w, sheet_name="Summary", index=False)
    print(f"{len(df)} conference signals -> {dst}")

if __name__ == "__main__":
    main()
