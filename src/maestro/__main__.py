"""Entry point for python -m maestro."""

from maestro import __version__


def main() -> None:
    print(f"MAESTRO v{__version__}")


if __name__ == "__main__":
    main()