r"""Write runtime/test/bcpack_vectors.h: BCPack chunks with what tools/bcpack.py makes of them (stdlib only).

bcpack_test (runtime/test/bcpack_test.cpp) decodes every chunk here with the runtime's decoder (src/bcpack.cpp) and
compares with the Python decoder's answer, so the C++ port is checked against the reference without Python at test
time. Every chunk is synthetic: tests/test_bcpack.py's encoder over random lanes (no game files or texture bytes).

- Lane cases: random lanes of all six modes, some corrupted, cut short or with a byte appended, as
  tests/test_bcpack_native.py checks the Rust port. Expected: the lanes `_decode_lanes_raw` returns (with nothing
  but zeros after the bytes it used), or a refusal.
- Block cases: modes 1, 3, 4 and 5 with history distances that point at no block, at an earlier block or at a
  built-in entry (confirmed or not), and now and then a non-zero lane 1. Expected: `bcpack.decode`'s strict output
  (the default), or the error class it raises.

The pure-Python decoder is the reference (LEONIDA_BCPACK_NATIVE=0), so the Rust library does not take part.

Usage:
    python -I runtime/test/bcpack_vectors.py            # rewrite runtime/test/bcpack_vectors.h
    python -I runtime/test/bcpack_vectors.py --check    # exit 1 if the header is not what this writes
"""
import argparse
import os
import random
import sys

os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import bcpack  # noqa: E402
import test_bcpack as synth  # noqa: E402

OUT = os.path.join(HERE, "bcpack_vectors.h")
SEED = 20261124

# runtime/src/bcpack.h Status, in declaration order.
OK, UNSUPPORTED, BAD_SIZE, BAD_STREAM, TRAILING, LANE1, BAD_SYMBOL, UNCONFIRMED, BAD_BLOCK = range(9)


def _lanes_reference(raw, codec, nsym):
    """`_decode_lanes_raw`'s lanes for `raw`, or None where it raises or leaves non-zero bytes."""
    try:
        lanes, used = bcpack._decode_lanes_raw(memoryview(raw), bcpack.LANE_WIDTHS[codec], nsym)
    except (bcpack.BcpackError, ZeroDivisionError, IndexError):
        return None
    return None if any(raw[used:]) else lanes


def _corrupt(rng, raw):
    raw = bytearray(raw)
    kind = rng.random()
    if kind < 0.25:
        for _ in range(rng.randint(1, 3)):
            raw[rng.randrange(len(raw))] = rng.randrange(256)
    elif kind < 0.35:
        del raw[rng.randrange(len(raw)):]
    elif kind < 0.42:
        raw += bytes([rng.randrange(1, 256)])
    return bytes(raw)


def _random_lanes(rng, codec, nsym):
    lanes = []
    for w in bcpack.LANE_WIDTHS[codec]:
        rate = rng.random() + 0.05
        lanes.append([min((1 << w) - 1, int(rng.expovariate(rate))) if rng.random() < 0.9
                      else rng.randrange(1 << w) for _ in range(nsym)])
    return lanes


def lane_cases(rng, count):
    out = []
    for _ in range(count):
        codec = rng.choice(sorted(bcpack.LANE_WIDTHS))
        nsym = rng.choice([1, 2, 7, 16]) * bcpack.BLOCK_BYTES[codec] // bcpack.ELEMENT_BYTES
        lanes = _random_lanes(rng, codec, nsym)
        q = rng.randrange(16) if rng.random() < 0.7 else 0
        raw = _corrupt(rng, synth.encode_lanes(lanes, codec, q=q))
        out.append((codec, nsym, raw, _lanes_reference(raw, codec, nsym)))
    return out


def _distance_code(rng, i):
    """A history distance code for block i: none, an earlier block, or a built-in entry (d > i)."""
    r = rng.random()
    if r < 0.5:
        d = 0
    elif r < 0.96:
        d = rng.randint(1, min(i, 255)) if i > 0 else 0
    else:
        d = rng.randint(i + 1, 255) if i < 255 else 0
    # The 8-bit code dist() and dist8() read back as d: even 2d up to 127, odd 2 * (256 - d) - 1 above.
    return 2 * d if d <= 127 else 2 * (256 - d) - 1


def block_cases(rng, count):
    out = []
    for _ in range(count):
        codec = rng.choice([1, 3, 4, 5])
        nblocks = rng.choice([1, 3, 8, 16])
        two = bcpack.BLOCK_BYTES[codec] == 16
        nsym = nblocks * (2 if two else 1)
        lanes = _random_lanes(rng, codec, nsym)
        lanes[1] = [0] * nsym if rng.random() > 0.06 else [rng.randrange(1, 16)] + [0] * (nsym - 1)
        for i in range(nblocks):
            code = _distance_code(rng, i)
            if two:  # lane 0 holds the code's even bits in element a and its odd bits in element b (`_inter8`)
                lanes[0][2 * i] = sum(((code >> (2 * k)) & 1) << k for k in range(4))
                lanes[0][2 * i + 1] = sum(((code >> (2 * k + 1)) & 1) << k for k in range(4))
            else:
                lanes[0][i] = code
        raw = synth.encode_lanes(lanes, codec, q=rng.randrange(16) if rng.random() < 0.7 else 0)
        if rng.random() < 0.12:
            raw = _corrupt(rng, raw)
        size = nblocks * bcpack.BLOCK_BYTES[codec]
        out.append((codec, raw, size) + _reference(raw, codec, size) + (None,))
    return out


def _reference(raw, codec, size):
    """(status, blocks) of the strict Python decode of `raw`."""
    try:
        return OK, bcpack.decode(raw, codec, size)
    except bcpack.BcpackProvisionalError:
        return UNCONFIRMED, None
    except bcpack.BcpackError as e:
        text = str(e)
        # "lane 10 ... input ran out" is a stream error: match the rebuilders' own message, not "lane 1".
        if "lane 1 is non-zero" in text:
            return LANE1, None
        if "left over" in text:
            return TRAILING, None
        if (text.startswith("BC7 signature") or "signature check" in text or "consistency check" in text
                or "mode symbol" in text or "inconsistent index" in text):
            return BAD_BLOCK, None
        return BAD_STREAM, None
    except (ZeroDivisionError, IndexError):
        return BAD_STREAM, None


def _split8(code):
    """Two 4-bit symbols whose `_inter8` is `code`: its even bits, then its odd bits."""
    return (sum(((code >> (2 * k)) & 1) << k for k in range(4)),
            sum(((code >> (2 * k + 1)) & 1) << k for k in range(4)))


def _zig(v):
    """The symbol `_unzig` turns into v (-128..127)."""
    return 2 * v if v >= 0 else -2 * v - 1


def _small_index_symbol(rng):
    """An index-row symbol whose even and odd halves both decode to 0 or 1, so an anchor index stays representable
    when the reference is the zero state."""
    return _split8_inverse(rng.choice((0, 2)), rng.choice((0, 2)))


def _split8_inverse(even, odd):
    return sum((((even >> k) & 1) << (2 * k)) | (((odd >> k) & 1) << (2 * k + 1)) for k in range(4))


def bc7_cases(rng, count):
    """BC7 chunks: one signature per block (now and then an inferred one), endpoint and index distances within the
    chunk or, rarely, into the built-in history (all unknown), random endpoints and mostly representable indices."""
    import bcpack_tables_bptc as bptc  # noqa: PLC0415
    static = bptc._bc7_static()
    cells = {rank: (a, b) for a in range(32) for b in range(16)
             if (rank := static["rank_of"][a * 16 + b]) >= 0}
    valid = [r for r, cfg in enumerate(static["config"]) if cfg[0] <= 7 and cfg[1] != 255]
    plain = [r for r in valid if r not in static["inferred"]]
    widths = bcpack.LANE_WIDTHS[7]
    out = []
    for _ in range(count):
        nblocks = rng.choice([1, 2, 4, 8, 16])
        nsym = 2 * nblocks
        lanes = _random_lanes(rng, 7, nsym)
        for i in range(nblocks):
            a, b = 2 * i, 2 * i + 1
            rank = rng.choice(valid if rng.random() < 0.04 else plain)
            lanes[15][a], lanes[15][b] = cells[rank]
            mode = static["config"][rank][0]

            def distance():
                r = rng.random()
                if r < 0.55 or (i == 0 and r < 0.97):
                    return 0
                if r < 0.97:
                    return rng.randint(1, i)
                return rng.randint(i + 1, 255)

            di, de = distance(), distance()
            lanes[1][a], lanes[1][b] = _split8(di)
            delta = ((de - di + 128) % 256) - 128
            lanes[0][a], lanes[0][b] = _split8(_zig(delta))
            rows = [11, 12, 13, 14] + ([6, 7, 8, 10] if mode in (4, 5) else [])
            for lane in rows:
                for e in (a, b):
                    lanes[lane][e] = _small_index_symbol(rng) & ((1 << widths[lane]) - 1)
        raw = synth.encode_lanes(lanes, 7, q=rng.randrange(16) if rng.random() < 0.7 else 0)
        if rng.random() < 0.06:
            raw = _corrupt(rng, raw)
        size = 16 * nblocks
        status, expected = _reference(raw, 7, size)
        try:
            lenient = bcpack.decode(raw, 7, size, allow_provisional=True)
        except (bcpack.BcpackError, ZeroDivisionError, IndexError):
            lenient = None
        out.append((7, raw, size, status, expected, lenient))
    return out


def bc6h_cases(rng, count):
    """BC6H chunks: a mode symbol 0..13 per block (now and then one past 13), a partition, endpoint and index
    distances within the chunk or, rarely, into the built-in history (all unknown), random endpoint deltas, and index
    deltas of -1 or -2, which keep the anchors representable from the no-reference index base (8)."""
    out = []
    for _ in range(count):
        nblocks = rng.choice([1, 2, 4, 8, 16])
        nsym = 2 * nblocks
        lanes = _random_lanes(rng, 6, nsym)
        for i in range(nblocks):
            a, b = 2 * i, 2 * i + 1
            lanes[15][a] = rng.randrange(14) if rng.random() > 0.02 else rng.randrange(14, 32)
            lanes[15][b] = rng.randrange(32)

            def distance():
                r = rng.random()
                if r < 0.55 or (i == 0 and r < 0.97):
                    return 0
                if r < 0.97:
                    return rng.randint(1, i)
                return rng.randint(i + 1, 255)

            di, de = distance(), distance()
            lanes[1][a], lanes[1][b] = _split8(di)
            delta = ((de - di + 128) % 256) - 128
            lanes[0][a], lanes[0][b] = _split8(_zig(delta))
            for lane in (11, 12, 13, 14):
                for e in (a, b):
                    lanes[lane][e] = _split8_inverse(rng.choice((1, 3)), rng.choice((1, 3)))
        raw = synth.encode_lanes(lanes, 6, q=rng.randrange(16) if rng.random() < 0.7 else 0)
        if rng.random() < 0.06:
            raw = _corrupt(rng, raw)
        size = 16 * nblocks
        status, expected = _reference(raw, 6, size)
        try:
            lenient = bcpack.decode(raw, 6, size, allow_provisional=True)
        except (bcpack.BcpackError, ZeroDivisionError, IndexError):
            lenient = None
        out.append((6, raw, size, status, expected, lenient))
    return out


def _bytes(name, data, ctype="std::uint8_t"):
    body = ", ".join(f"0x{b:02x}" for b in data) if data else "0"
    return f"inline constexpr {ctype} {name}[] = {{{body}}};"


def render():
    rng = random.Random(SEED)
    lanes = lane_cases(rng, 120)
    blocks = block_cases(rng, 90) + bc7_cases(rng, 80) + bc6h_cases(rng, 80)
    lines = [
        "// Generated by runtime/test/bcpack_vectors.py. Do not edit by hand (tests/test_bcpack_vectors.py checks it).",
        "// Synthetic BCPack chunks (tests/test_bcpack.py's encoder over random lanes; no game data) and what the",
        "// Python reference decoder, tools/bcpack.py, makes of them.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace bcpack_vectors {",
        "",
        "struct LaneCase {",
        "    std::uint32_t mode;",
        "    std::size_t count;  // symbols per lane",
        "    const std::uint8_t* raw;",
        "    std::size_t raw_size;",
        "    bool ok;                       // the Python lane model accepts the stream",
        "    const std::uint16_t* symbols;  // lane-major, count per lane, when ok",
        "};",
        "",
        "struct BlockCase {",
        "    std::uint32_t mode;",
        "    const std::uint8_t* raw;",
        "    std::size_t raw_size;",
        "    std::size_t out_size;",
        "    int status;                 // leonida::bcpack::Status, as an int",
        "    const std::uint8_t* blocks;  // out_size bytes when status is 0",
        "    bool lenient_known;          // BC7: Python's provisional decode succeeded",
        "    const std::uint8_t* lenient_blocks;  // its out_size bytes",
        "};",
        "",
    ]
    for k, (codec, nsym, raw, ref) in enumerate(lanes):
        lines.append(_bytes(f"lane_raw_{k}", raw))
        flat = [s for lane in ref for s in lane] if ref is not None else []
        body = ", ".join(str(s) for s in flat) if flat else "0"
        lines.append(f"inline constexpr std::uint16_t lane_sym_{k}[] = {{{body}}};")
    lines += ["", f"inline constexpr LaneCase kLaneCases[] = {{"]
    for k, (codec, nsym, raw, ref) in enumerate(lanes):
        lines.append(f"    {{{codec}, {nsym}, lane_raw_{k}, {len(raw)}, {'true' if ref is not None else 'false'}, "
                     f"lane_sym_{k}}},")
    lines += ["};", ""]
    for k, (codec, raw, size, status, expected, lenient) in enumerate(blocks):
        lines.append(_bytes(f"block_raw_{k}", raw))
        lines.append(_bytes(f"block_out_{k}", expected or b""))
        if lenient is not None and lenient != expected:
            lines.append(_bytes(f"block_lenient_{k}", lenient))
    lines += ["", "inline constexpr BlockCase kBlockCases[] = {"]
    for k, (codec, raw, size, status, expected, lenient) in enumerate(blocks):
        if lenient is None:
            tail = "false, nullptr"
        else:
            tail = f"true, {'block_out' if lenient == expected else 'block_lenient'}_{k}"
        lines.append(f"    {{{codec}, block_raw_{k}, {len(raw)}, {size}, {status}, block_out_{k}, {tail}}},")
    lines += ["};", "", "}  // namespace bcpack_vectors", ""]
    return "\n".join(lines), lanes, blocks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    text, lanes, blocks = render()
    if args.check:
        try:
            with open(args.out, encoding="utf-8", newline="") as f:
                current = f.read()
        except OSError as e:
            print(f"ERROR: cannot read {args.out}: {e}")
            return 1
        if current != text:
            print(f"ERROR: {args.out} is out of date; run python -I runtime/test/bcpack_vectors.py")
            return 1
        print("bcpack_vectors.h is up to date")
        return 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    accepted = sum(ref is not None for *_, ref in lanes)
    by_status = {}
    for _, _, _, status, _, _ in blocks:
        by_status[status] = by_status.get(status, 0) + 1
    print(f"wrote {args.out}: {len(lanes)} lane cases ({accepted} accepted), {len(blocks)} block cases "
          f"(by status {dict(sorted(by_status.items()))})")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
