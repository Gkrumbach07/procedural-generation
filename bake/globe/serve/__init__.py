"""Serving baked worlds and running bakes behind them.

:mod:`globe.serve.jobs` is the queue the servers run work in (one job at a
time: a bake uses every core), :mod:`globe.serve.worlds` reads what a
directory of worlds holds, and :mod:`globe.serve.hub` is the handler for
``scripts/serve_planets.py`` -- the page that lists the planets, makes new
ones and watches them bake.
"""
