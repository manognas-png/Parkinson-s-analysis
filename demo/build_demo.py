"""Assemble demo/gait_demo.html: one self-contained file (no external files, works offline).

Takes demo/gait_demo_src.html and replaces its placeholders with
  /*__ENGINE_JS__*/       -> demo/engine.js       (the validated analysis engine)
  /*__TEMPLATE_JSON__*/   -> demo/template.json   (healthy-stride templates)
  /*__DEMO_DATA_JSON__*/  -> demo/demo_data.json  (pairs, clips, group statistics)
  /*__APP_JS__*/          -> demo/app.js          (the interactive page)

Run from the project root (after slurm/demo_data.sbatch has produced demo_data.json):
    python demo/build_demo.py
"""

from pathlib import Path

DEMO = Path(__file__).resolve().parent
PARTS = {
    "/*__ENGINE_JS__*/": "engine.js",
    "/*__TEMPLATE_JSON__*/": "template.json",
    "/*__DEMO_DATA_JSON__*/": "demo_data.json",
    "/*__APP_JS__*/": "app.js",
}


def main():
    page = (DEMO / "gait_demo_src.html").read_text()
    start, end = page.index("<!--"), page.index("-->") + 3
    page = page[:start] + ("<!-- Built by demo/build_demo.py from demo/gait_demo_src.html, engine.js, app.js,\n"
                           "     template.json and demo_data.json. Edit those files, not this one. -->") + page[end:]
    for placeholder, filename in PARTS.items():
        path = DEMO / filename
        if not path.exists():
            raise SystemExit(f"ERROR: missing {path}")
        if page.count(placeholder) != 1:
            raise SystemExit(f"ERROR: placeholder {placeholder} not found exactly once")
        content = path.read_text()
        if "</script" in content.lower():
            raise SystemExit(f"ERROR: {filename} contains '</script', which would break the page")
        page = page.replace(placeholder, content)
    output = DEMO / "gait_demo.html"
    output.write_text(page)
    print(f"wrote {output} ({output.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
