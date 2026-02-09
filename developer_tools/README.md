# Developer Tools

This folder contains scripts and utilities used to develop and maintain RAVEN.

## InputSpecs/XSD synchronization

When you add or change `getInputSpecification()` or any `returnInputParameter` wiring,
the InputData audit test will fail if the baseline JSON is out of date.

To refresh the baseline (and optionally regenerate XSDs), run:

```bash
python3 developer_tools/sync_input_specs.py
```

If you only want to update the audit baseline JSON (no XSD regeneration), run:

```bash
python3 developer_tools/sync_input_specs.py --skip-xsd
```

If the sync script fails due to missing optional dependencies, rerun with
`--skip-xsd` or install the missing dependency set.

