"""
MAESTRO viz — Streamlit dashboard for exploring experiment results.

Read-only consumer of the experiment SQLite database (and, where useful, the
JSON outputs of ``maestro.analysis``). Launch with:

    streamlit run src/maestro/viz/app.py

This package provides the navigation shell, read-only DB access, settings
panel, and empty-state handling. Chart and metric logic live in the design
system and the individual view modules.
"""
