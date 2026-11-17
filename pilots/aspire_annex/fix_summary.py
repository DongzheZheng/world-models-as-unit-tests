import json
from pathlib import Path
if __name__ == "__main__":
    p = Path("runs/annex/aspire_style/summary.json"); S = json.loads(p.read_text())
    S["conclusion"] = S["conclusion"].replace("(5 of 6 such casts passed within 0.06-0.13 m of its centre)",
                                              "(7 of the 8 casts of >= 0.23 m on seed 62 passed 0.054-0.135 m from its centre)")
    p.write_text(json.dumps(S, indent=1)); print(S["conclusion"][-420:])
