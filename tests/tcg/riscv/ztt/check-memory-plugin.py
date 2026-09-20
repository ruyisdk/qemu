#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check each byte of the two labeled raw AME accesses in libmem's output."""

import argparse
import csv
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("output")
    args = parser.parse_args()
    symbols = {}
    output = subprocess.check_output([args.nm, args.binary], text=True)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    expected = {}
    for insn, region, direction in (("memory_load", "memory_input", "load"),
                                    ("memory_store", "memory_output", "store")):
        pc, base = symbols[insn], symbols[region]
        expected[pc] = [(base + i, base + i, 8, direction, i)
                        for i in range(16)]
    actual = {pc: [] for pc in expected}
    with open(args.output, encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != 7 or not row[0].startswith("0x"):
                continue
            pc = int(row[0], 16)
            if pc in actual:
                actual[pc].append((int(row[2], 16), int(row[3], 16),
                                   int(row[4]), row[5], int(row[6], 16)))
    if actual != expected:
        raise SystemExit(f"AME plugin mismatch: expected={expected}, "
                         f"actual={actual}")
    print("PASS: both AME accesses reported all 16 byte addresses and values")


if __name__ == "__main__":
    main()
