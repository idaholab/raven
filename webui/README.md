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

### Launch modes

The server auto-detects a launch mode, or you can override it.

- Local (default): `python -m webui.app --mode local` (binds to 127.0.0.1)
- Private LAN/HPC: `python -m webui.app --mode private` (binds to 0.0.0.0)
- Public: `python -m webui.app --mode public` (binds to 0.0.0.0; put behind a proxy)

Auto mode honors `PRLO_WEBUI_MODE=local|private|public`. When connected via SSH,
auto defaults to local so you can use SSH tunneling without opening ports.
