# PRLO Web UI

## XML Builder

This repo includes a lightweight XML builder page that turns the example PRLO RAVEN input decks into insertable XML blocks.

### Run (development)

From the repository root:

```bash
pip install -r webui/requirements.txt
python -m webui.app --list
python -m webui.app --reload
```

Then open:

- `http://127.0.0.1:8750/xml-builder`

The builder uses:

- `GET /api/xml-builder/catalog` to discover snippets from `plugins/PRLO/examples/**.xml` that contain a `<Simulation>` root.
- `GET /api/xml-builder/example?path=<path>` to fetch full example decks to use as a starting point.
