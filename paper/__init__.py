"""Table and figure pipeline of the RARe paper. Everything here runs on CPU."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
