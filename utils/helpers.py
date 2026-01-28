import html

def render_highlighted(text: str, spans: list[dict]) -> str:
    if not text:
        return ""

    spans = [s for s in spans if 0 <= s["start"] < s["end"] <= len(text)]
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))

    out = []
    cursor = 0
    for s in spans:
        start, end = s["start"], s["end"]
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        frag = html.escape(text[start:end])

        cat = s.get("category", "")
        if cat == "urgency":
            bg = "#ffe08a"
        elif cat == "inevitability":
            bg = "#ffb3b3"
        else:
            
            bg = "#ffcc80"

        out.append(f"<span style='background-color:{bg}; padding:2px 4px; border-radius:4px;'>{frag}</span>")
        cursor = end

    out.append(html.escape(text[cursor:]))
    return "".join(out)