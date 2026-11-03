# Developer Tools

This folder contains scripts and utilities used to develop and maintain RAVEN.

## InputSpecs/XSD synchronization

`developer_tools/audit_input_specs.py` audits InputData spec coverage by
**live registry introspection**: for each entity it imports the real factory,
enumerates `factory.knownTypes()`, resolves each type with
`factory.returnClass()`, and checks `getInputSpecification` coverage with a
live `hasattr`/MRO test. Any registered class that exposes a secondary
registry dict (e.g. `knownAlgorithms` on the multi-objective genetic algorithm)
has its sub-registered names folded in too, so algorithms registered via
`registerAlgorithm(...)` are not invisible to the guardrail. An AST scan is kept
only as a fallback for classes that cannot be resolved live.

### What the guardrail asserts

The `TestXSD/test_input_spec_audit` test compares only the **stable coverage
facts** against the committed baseline `developer_tools/audit_input_specs.json`:

- per entity, the set of registered classes missing `getInputSpecification`;
- per entity, factory-import / class-resolve errors;
- the `xsd_diff` missing/extra-in-XSD sets.

It deliberately does **not** freeze the full type rosters or the manual-parsing
path map. Consequences:

- Adding a well-formed entity (a new type that has a spec and appears in the
  XSD) does **not** fail the test — the baseline is unchanged.
- Removing a `getInputSpecification`, breaking XSD coverage, introducing a
  manual-parse class, or an entity that no longer imports **does** fail it.

Because the facts contain no rosters and no absolute paths, the baseline is
portable across branches, merges, and machines.

### Refreshing the baseline

When you make an intentional change to spec coverage (add/remove a spec, change
XSD wiring, register a new entity), refresh the baseline (and optionally
regenerate XSDs):

```bash
python3 developer_tools/sync_input_specs.py
```

If you only want to update the coverage-facts baseline JSON (no XSD
regeneration), run:

```bash
python3 developer_tools/sync_input_specs.py --skip-xsd
```

If the sync script fails due to missing optional dependencies, rerun with
`--skip-xsd` or install the missing dependency set.

The audit tool can also be run directly:

```bash
# human-readable report
python3 developer_tools/audit_input_specs.py
# full audit JSON (includes type rosters + manual-parsing scan)
python3 developer_tools/audit_input_specs.py --json /tmp/audit_full.json --no-print
# minimal coverage-facts baseline (what the test compares against)
python3 developer_tools/audit_input_specs.py --baseline-json developer_tools/audit_input_specs.json --no-print
```

