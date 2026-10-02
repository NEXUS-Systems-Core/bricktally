"""BrickTally desktop entry point."""

import sys


def main() -> None:
    if "--selftest" in sys.argv[1:]:
        from bricktally.selftest import run_selftest

        raise SystemExit(run_selftest())
    from bricktally_ui.window import run

    run()


if __name__ == "__main__":
    main()
