"""Echo a string, optionally with a prefix when --verbose is set.

Usage:
    python echo.py --text "hello world" --verbose
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Echo a string.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-t", "--text", type=str, required=True,
                        help="A string to print")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Adds a descriptive prefix")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.verbose:
        print(f"The entered string was: {args.text}")
    else:
        print(args.text)

if __name__ == "__main__":
    main()