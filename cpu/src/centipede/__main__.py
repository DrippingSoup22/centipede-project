"""Allow ``python -m centipede PLAN.toml``.

Spawned worker processes do not re-run a package's ``__main__`` module, so the
command-line imports below never load inside environment workers.
"""

if __name__ == "__main__":
    from centipede.cli import main

    raise SystemExit(main())
