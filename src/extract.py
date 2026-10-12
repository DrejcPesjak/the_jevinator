"""Extract the saved Thought Anchors page without fetching external resources."""

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = ROOT / "data/_deepseekR1_blackmail_scenario2.html"
DEFAULT_JSON = DEFAULT_HTML.with_suffix(".json")
NUMBER = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def extract(path: Path) -> dict:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    prompts = soup.select("pre")
    if len(prompts) != 1 or not prompts[0].get_text().strip():
        raise ValueError("Expected exactly one nonempty scenario prompt")
    sentences = []
    for node in soup.select("[data-step-id]"):
        children = node.find_all("div", recursive=False)
        labels = children[1].find_all("span")
        sentences.append({
            "ix": int(node["data-step-id"]),
            "text": children[2].get_text(),
            "classification": labels[0].get_text(strip=True),
            "importance": float(labels[1].get_text().split(":", 1)[1]),
        })
    if not sentences or [s["ix"] for s in sentences] != list(range(len(sentences))):
        raise ValueError("Expected unique, contiguous step indices starting at zero")

    # SVG links have coordinates rather than step IDs. Match their endpoints to nodes.
    positions = {}
    for node in soup.select("g.nodes > g"):
        xy = tuple(map(float, re.findall(NUMBER, node["transform"])))
        positions[xy] = int(node.find("text").get_text())
    connections = []
    for link in soup.select("g.links path"):
        match = re.fullmatch(rf"M({NUMBER}),({NUMBER})L({NUMBER}),({NUMBER})", link["d"])
        if not match:
            raise ValueError(f"Unsupported SVG connection geometry: {link['d']}")
        x1, y1, x2, y2 = map(float, match.groups())
        source, target = positions[(x1, y1)], positions[(x2, y2)]
        marker = link.get("marker-end", "")
        if "sequential" in marker:
            kind = "sequential"
        elif "causal" in marker:
            kind = "causal"
        else:
            raise ValueError(f"Unknown connection type: {marker}")
        connections.append({
            "source_ix": source, "target_ix": target, "type": kind,
            "rendered_opacity": float(link.get("opacity", 1)),
        })
    return {
        "source_html": path.name,
        "prompt_note": "The export labels this Scenario prompt; original message roles are not separated.",
        "system_prompt": prompts[0].get_text(),
        "sentences": sentences,
        "connections": connections,
        "annotation_note": "Importance has the displayed 3-decimal precision. Connection opacity is a visual attribute, not a recovered causal weight.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="?", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = extract(args.html)
    output = args.output or args.html.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(document['sentences'])} steps and {len(document['connections'])} connections to {output}")


if __name__ == "__main__":
    main()
