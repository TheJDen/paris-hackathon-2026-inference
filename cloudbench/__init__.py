"""cloudbench — provider-independent benchmark harness.

BENCH selects *what* to run; PROVIDER selects *where* to run it.
The harness provisions hardware, syncs code (or pulls a Docker image),
runs the benchmark, downloads logs/results, and destroys the resource
by default.
"""

__version__ = "0.1.0"
