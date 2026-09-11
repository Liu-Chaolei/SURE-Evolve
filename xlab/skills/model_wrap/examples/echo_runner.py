#!/usr/bin/env python3
import json
import sys


def main() -> None:
    value = json.load(sys.stdin)
    print(json.dumps({"echo": value}))


if __name__ == "__main__":
    main()
