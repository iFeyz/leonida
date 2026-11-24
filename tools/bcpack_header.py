r"""Write runtime/src/bcpack_tables.h, the C++ copy of the BCPack constant tables (stdlib only).

The runtime's BCPack decoder (runtime/src/bcpack.cpp, a port of tools/bcpack.py) needs the same codec constants the
Python decoder reads from tools/bcpack_tables.py: the rANS probability tables, the symbol-rank lookup tables and the
confirmed built-in history entries. This tool writes them as C++ arrays so the two decoders cannot drift apart by
hand copying. Provisional history entries are not written: the runtime decodes strictly, as `bcpack.decode` does by
default, and takes anything else from the user's own calibrated history (tools/bcpack_history_calib.py).

Usage:
    python -I tools/bcpack_header.py            # rewrite runtime/src/bcpack_tables.h
    python -I tools/bcpack_header.py --check    # exit 1 if the header is not what this tool would write

tests/test_bcpack_header.py runs --check, so a change to tools/bcpack_tables.py without a new header fails.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bcpack  # noqa: E402
import bcpack_tables as t  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "runtime", "src", "bcpack_tables.h")
MAX_WIDTH = bcpack.MAX_WIDTH
SIDE = MAX_WIDTH + 1

# codec: (confirmed history table name, values per entry)
HISTORY = {1: ("BC1_HISTORY_CONFIRMED", 6), 3: ("BC3_HISTORY_CONFIRMED", 8), 4: ("BC4_HISTORY_CONFIRMED", 2),
           5: ("BC5_HISTORY_CONFIRMED", 4)}


def _rows(values, per_line=16, width=6):
    vals = [str(v) for v in values]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(v.rjust(width) for v in vals[i:i + per_line]) + ",")
    return "\n".join(lines)


def _prob_flat():
    flat = [-1] * (SIDE * 16 * SIDE)
    for (w, q), freq in sorted(t.PROB_TABLES.items()):
        if not (0 <= w <= MAX_WIDTH and 0 <= q < 16 and len(freq) == w + 1):
            raise SystemExit(f"PROB_TABLES[{(w, q)}] does not have the expected shape")
        flat[(w * 16 + q) * SIDE:(w * 16 + q) * SIDE + w + 1] = freq
    return flat


def _history_flat(name, k):
    flat = [-1] * (256 * k)
    for entry, values in sorted(getattr(t, name).items()):
        if not (0 <= entry < 256 and len(values) == k):
            raise SystemExit(f"{name}[{entry}] does not have {k} values")
        flat[entry * k:entry * k + k] = values
    return flat


def _bc7_parts():
    """BC7 (BCPack mode 7): signature configurations, the rank of every lane-15 symbol pair, the public D3D11 BC7
    partition and anchor tables, and the ranks whose configuration is only inferred. The built-in BC7 history is
    not written: tools/bcpack_tables_bptc.py knows no bit of it, so it comes from the user's calibrated tables only."""
    s = bptc._bc7_static()
    config = [v for row in s["config"] for v in row]
    if len(s["config"]) != 286 or any(not 0 <= v <= 255 for v in config):
        raise SystemExit("BC7_CONFIG does not have the expected shape")
    for k in range(256):
        if bptc.load_bc7(None)["history_endpoint_bits"][k] != (0,) * 24:
            raise SystemExit("the built-in BC7 history now has known bits: write it into the header too")
    return [
        "",
        "// BC7_CONFIG: per signature rank, (mode, partition, rotation, index selection); mode 255 is invalid.",
        f"inline constexpr std::array<std::array<std::uint8_t, 4>, {len(s['config'])}> kBc7Config = {{{{",
        "\n".join("    {" + ", ".join(str(v) for v in row) + "}," for row in s["config"]),
        "}};",
        "",
        "// The signature rank of lane-15 symbols (a, b) at a * 16 + b; -1 where no rank is assigned.",
        f"inline constexpr std::array<std::int16_t, {len(s['rank_of'])}> kBc7Rank = {{",
        _rows(s["rank_of"], 16, 4),
        "};",
        "",
        "// D3D11 BC7 two-subset partitions, packed: bit p is the subset of pixel p.",
        f"inline constexpr std::array<std::uint16_t, {len(bptc.BC7_PART2_PACKED)}> kBc7Partition2 = {{",
        _rows(bptc.BC7_PART2_PACKED, 8, 6),
        "};",
        "",
        "// D3D11 BC7 three-subset partitions, packed: bits 2p..2p+1 are the subset of pixel p.",
        f"inline constexpr std::array<std::uint32_t, {len(bptc.BC7_PART3_PACKED)}> kBc7Partition3 = {{",
        _rows([f"{v}u" for v in bptc.BC7_PART3_PACKED], 6, 11),
        "};",
        "",
        "// Anchor pixel of subset 1 (two subsets), and of subsets 1 and 2 (three subsets), per partition.",
        f"inline constexpr std::array<std::uint8_t, 64> kBc7Anchor2 = {{",
        _rows(bptc.BC7_ANCHOR2, 16, 3),
        "};",
        f"inline constexpr std::array<std::uint8_t, 64> kBc7Anchor3a = {{",
        _rows(bptc.BC7_ANCHOR3A, 16, 3),
        "};",
        f"inline constexpr std::array<std::uint8_t, 64> kBc7Anchor3b = {{",
        _rows(bptc.BC7_ANCHOR3B, 16, 3),
        "};",
        "",
        "// Ranks whose mode-2 partition is inferred from anchor positions and order only (strict decoding refuses them).",
        f"inline constexpr std::array<std::uint16_t, {len(s['inferred'])}> kBc7Inferred = {{"
        + ", ".join(str(v) for v in sorted(s["inferred"])) + "};",
    ]


def _bc6h_parts():
    """BC6H (BCPack mode 6): per mode index 1..14 (lane-15 symbol + 1) its D3D11 mode code, transform, base and delta
    bits, region count and header size, and its header bit plan as `_bc6h_plan` computes it from the D3D11 layouts:
    (channel, endpoint quantity, bit, stream position), channel -1 for the partition field. The built-in BC6H history
    is not written: tools/bcpack_tables_bptc.py knows none of it."""
    eh, ih, eclass, iclass, _, _ = bptc.load_bc6h(None, with_confirmed_bits=True)
    if any(eclass) or any(iclass):
        raise SystemExit("the built-in BC6H history now has known entries: write it into the header too")
    modes, plan = [], []
    for mode in range(1, 15):
        code = bcpack._BC6H_CODE[mode]
        trans, base, deltas, regions = bcpack._BC6H_INFO[code]
        steps, header = bcpack._bc6h_plan(code)
        modes.append((code, trans, base, *deltas, regions, header, len(plan), len(steps)))
        plan += steps
    return [
        "",
        "// BC6H modes 1..14 (index 0 unused): code, transformed, base bits, delta bits (r, g, b), regions, header",
        "// bits, first plan step, plan steps.",
        "struct Bc6hMode {",
        "    std::uint8_t code, transformed, base, delta[3], regions, header;",
        "    std::uint16_t plan_start, plan_size;",
        "};",
        "inline constexpr std::array<Bc6hMode, 15> kBc6hModes = {{",
        "    {0, 0, 0, {0, 0, 0}, 0, 0, 0, 0},",
        "\n".join(f"    {{{m[0]}, {m[1]}, {m[2]}, {{{m[3]}, {m[4]}, {m[5]}}}, {m[6]}, {m[7]}, {m[8]}, {m[9]}}},"
                  for m in modes),
        "}};",
        "",
        "// One header bit: endpoint quantity `q` (0 w, 1 x, 2 y, 3 z) of channel `c` (-1: the partition field), its bit",
        "// `bit`, at stream position `pos`.",
        "struct Bc6hStep {",
        "    std::int8_t c, q;",
        "    std::uint8_t bit, pos;",
        "};",
        f"inline constexpr std::array<Bc6hStep, {len(plan)}> kBc6hPlan = {{{{",
        _rows([f"{{{c}, {q}, {b}, {p}}}" for c, q, b, p in plan], 6, 16),
        "}};",
    ]


def render():
    parts = [
        "// Generated by tools/bcpack_header.py from tools/bcpack_tables.py. Do not edit by hand: change the Python",
        "// tables and run `python -I tools/bcpack_header.py` (tests/test_bcpack_header.py checks the two agree).",
        "//",
        "// Codec-level constants of the clean-room BCPack decoder (no game data): rANS probability tables, symbol-rank",
        "// lookup tables and the confirmed built-in history entries. Provisional entries are left out on purpose.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace leonida::bcpack::tables {",
        "",
        f"inline constexpr int kMaxWidth = {MAX_WIDTH};",
        "",
        "// Lane widths per codec (BCPack mode 1, 3, 4, 5, 6, 7), in lane order.",
    ]
    for codec in sorted(bcpack.LANE_WIDTHS):
        widths = bcpack.LANE_WIDTHS[codec]
        parts.append(f"inline constexpr std::array<std::uint8_t, {len(widths)}> kLaneWidths{codec} = "
                     "{" + ", ".join(str(w) for w in widths) + "};")
    flat = _prob_flat()
    parts += [
        "",
        "// prob[(w * 16 + q) * 11 + p]: the frequency of band p of the model for symbol width w and parameter q; a",
        "// row whose first value is -1 has no model. Band 0 is symbol 0, band p >= 1 covers [2^(p-1), 2^p); a",
        "// present row sums to 65536 with each band weighted by its symbol count.",
        f"inline constexpr std::array<std::int32_t, {len(flat)}> kProb = {{",
        _rows(flat),
        "};",
    ]
    luts = [
        ("kBc1Quadrant", list(t.BC1_QUADRANT), "BC1_QUADRANT: a symbol's two index pairs of one quadrant"),
        ("kBc3AlphaPair", [v for p in t.BC3_ALPHA_PAIR for v in p], "BC3_ALPHA_PAIR, flattened: 64 rows of two "
         "3-bit indices"),
        ("kBc3ColourQuadrant", list(t.BC3_COLOUR_QUADRANT), "BC3_COLOUR_QUADRANT"),
        ("kBc4Pair", [v for p in t.BC4_PAIR for v in p], "BC4_PAIR, flattened: 64 rows of two 3-bit indices"),
        ("kBc5Pair", [v for p in t.BC5_PAIR for v in p], "BC5_PAIR, flattened: 64 rows of two 3-bit indices"),
    ]
    for name, values, what in luts:
        parts += ["", f"// {what}.", f"inline constexpr std::array<std::uint8_t, {len(values)}> {name} = {{",
                  _rows(values, 16, 3), "};"]
    for codec, (name, k) in sorted(HISTORY.items()):
        flat = _history_flat(name, k)
        parts += ["", f"// {name}: entry e at [e * {k}, e * {k} + {k}); -1 where the entry is not confirmed.",
                  f"inline constexpr std::array<std::int16_t, {len(flat)}> kBc{codec}History = {{",
                  _rows(flat, k * (16 // k) if k <= 16 else k, 4), "};"]
    parts += _bc7_parts()
    parts += _bc6h_parts()
    parts += ["", "}  // namespace leonida::bcpack::tables", ""]
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    ap.add_argument("--out", default=OUT, help="header path (default: runtime/src/bcpack_tables.h)")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        try:
            with open(args.out, encoding="utf-8", newline="") as f:
                current = f.read()
        except OSError as e:
            print(f"ERROR: cannot read {args.out}: {e}")
            return 1
        if current != text:
            print(f"ERROR: {args.out} is out of date; run python -I tools/bcpack_header.py")
            return 1
        print("bcpack_tables.h is up to date")
        return 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
