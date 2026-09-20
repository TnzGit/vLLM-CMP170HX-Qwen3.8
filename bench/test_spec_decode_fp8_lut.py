"""Exhaustive E4M3FN LUT semantics check for the SM80 verifier patch."""

from __future__ import annotations

import math
import sys


def build_lut() -> list[float]:
    values: list[float] = []
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            value = 0.0
        elif exponent == 0:
            value = mantissa * (2.0**-9)
        else:
            value = (1.0 + mantissa * 0.125) * (2.0 ** (exponent - 7))
        values.append(sign * value)
    return values


def reference(byte: int) -> float:
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return 0.0
    sign = -1.0 if byte & 0x80 else 1.0
    if exponent == 0:
        return sign * mantissa * (2.0**-9)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def main() -> int:
    lut = build_lut()
    mismatches = [
        (byte, got, reference(byte))
        for byte, got in enumerate(lut)
        if not math.isclose(got, reference(byte), rel_tol=0.0, abs_tol=0.0)
    ]
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches; first={mismatches[0]}")
        return 1
    assert lut[0x7F] == 0.0 and lut[0xFF] == 0.0
    print("PASS: all 256 E4M3FN codes match; NaN encodings fail closed to zero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
