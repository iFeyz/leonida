// BCPack decoder for the DirectStorage worker: a port of tools/bcpack.py (docs/textures.md, "What BCPack is").
//
// BCPack is the console's entropy coder for BCn texture data: DirectStorage inflates a request (zlib), then decodes
// BCPack into BCn blocks, then swizzles (docs/directstorage.md). This file is the middle step only, portable (no
// windows.h), so it builds and tests on every OS. The lane layer is a line-by-line port of
// tools/bcpack_calibration/src/model.rs (itself a port of `_decode_lanes_raw`); the block rebuilds follow
// tools/bcpack_calibration/src/blocks.rs and `_bc1_blocks` .. `_bc5_blocks`, `_bc6h_blocks` for BC6H (its header
// bit plans generated from `_bc6h_plan`), and `_bc7_check` / `_bc7_rebuild` for BC7 (with the structure of DadaCap's
// prototype port). The tables come from tools/bcpack_tables*.py through
// tools/bcpack_header.py (bcpack_tables.h).
//
// Two policies:
//  - strict (the default), as `bcpack.decode`: a block that refers to a built-in history entry nobody has confirmed,
//    a BC7 signature whose configuration is only inferred, or a BC7 block that fails its anchor / p-bit consistency
//    check refuses the whole chunk. The test vectors hold this decoder to the Python one.
//  - lenient, what a title needs: such a block is decoded anyway, from the zero state where the history is unknown
//    (what `bcpack.decode(..., allow_provisional=True)` does for BC6H and BC7, whose built-in history is all unknown), and
//    counted in Result::guessed together with every block that predicts from it. The texture then has a few wrong
//    blocks instead of missing.
//
// Done: all six BCPack modes, 1 (BC1), 3 (BC3), 4 (BC4), 5 (BC5), 6 (BC6H) and 7 (BC7).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace leonida::bcpack {

enum class Status {
    ok,
    unsupported_codec,    // not a BCPack mode this decoder rebuilds
    bad_size,             // the output is not a whole number of blocks of the mode
    bad_stream,           // the lane layer refuses the stream (corrupt, cut short, a missing model, ...)
    trailing_data,        // non-zero bytes after the last lane finished
    lane1_nonzero,        // lane 1 carries data (unexpected for BC1..BC5)
    bad_symbol,           // a symbol outside its lookup table
    unconfirmed_history,  // strict: a block needs a history entry or signature nobody has confirmed
    bad_block,            // BC6H/BC7: a mode or signature outside the table, or (strict) an inconsistent anchor or p-bit
};

const char* status_name(Status status);

// The lane widths of a BCPack mode (1, 3, 4, 5, 6 or 7); empty for another value.
std::span<const std::uint8_t> lane_widths(std::uint32_t mode);
// Bytes per block of a BCPack mode: 8 for BC1 and BC4, 16 for the others; 0 for another value.
std::uint32_t block_bytes(std::uint32_t mode);

// Entropy layer: the symbols of every lane, lane-major (`count` per lane), and the bytes of `in` it used.
struct Lanes {
    std::vector<std::uint16_t> symbols;
    std::size_t count = 0;
    std::size_t used = 0;
    std::uint16_t at(std::size_t lane, std::size_t i) const { return symbols[lane * count + i]; }
};
// `count` symbols per lane for `widths`; Status::bad_stream where tools/bcpack.py raises.
Status decode_lanes(std::span<const std::uint8_t> in, std::span<const std::uint8_t> widths, std::size_t count,
                    Lanes& out);

// Confirmed history entries from the user's calibrated tables (tools/bcpack_history_calib.py), to use besides the
// built-in ones: entry e of mode m at values[m][e * K .. e * K + K) with K = 6, 8, 2, 4 for modes 1, 3, 4, 5; -1 or a
// short vector means "not given". An entry given here wins over the built-in one (`_resolve_bc_history`'s merge).
// BC6H's and BC7's calibrated history is not read yet: their built-in entries are all unknown, so strict decoding
// refuses a chunk that refers to one and lenient decoding takes zeros.
struct History {
    std::array<std::vector<std::int16_t>, 8> values;
};

struct Options {
    const History* history = nullptr;
    bool lenient = false;
};

struct Result {
    Status status = Status::ok;
    std::size_t guessed = 0;  // lenient: blocks decoded from an unknown history entry, or that predict from one
};

// Decodes one BCPack chunk of `mode` into `out`, whose size is the decoded size (a whole number of blocks). On any
// status but ok, `out` is left in an unspecified state.
Result decode(std::span<const std::uint8_t> in, std::uint32_t mode, std::span<std::uint8_t> out,
              const Options& options);
// The strict decode of `in` (Options{history}).
Status decode(std::span<const std::uint8_t> in, std::uint32_t mode, std::span<std::uint8_t> out,
              const History* history = nullptr);

}  // namespace leonida::bcpack
