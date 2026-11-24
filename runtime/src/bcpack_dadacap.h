#pragma once

// BCPack (DSTORAGE_BCPACK_MODE), the console's BCn entropy coding: a chunk of 12 or 16 interleaved 16-bit rANS lanes
// back to BC1, BC3, BC4, BC5, BC6H or BC7 blocks. A port of tools/bcpack.py, whose notes (docs/textures.md, "What BCPack
// is") describe the stream; the tables in bcpack_tables.h are generated from tools/bcpack_tables*.py by
// tools/gen_bcpack_tables.py, so neither side drifts.
//
// Where the tool's strict mode refuses a chunk that names an entry of the codec's built-in history nobody has measured,
// this decoder does what a title needs: it takes a provisional entry, and where there is none (all of BC6H's and BC7's,
// which the repository does not carry) the zero-distance predictor. Such a block comes out with a wrong colour instead
// of the whole texture missing. `guessed` counts those blocks, with every block that predicts from one.
//
// Plain integers, standard library only.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "bcpack_dadacap_tables.h"

namespace leonida::bcpack_dc {

inline constexpr int kMaxLanes = 16;
inline constexpr std::uint32_t kHome = 1u << 24;  // the rANS lower bound, every fragment's end state

inline std::uint32_t block_bytes(unsigned codec) {
    switch (codec) {
    case 1: case 4: return 8;
    case 3: case 5: case 6: case 7: return 16;
    default: return 0;
    }
}

inline std::basic_string_view<std::uint8_t> lane_widths(unsigned codec) {
    static constexpr std::uint8_t w1[] = {8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8};
    static constexpr std::uint8_t w3[] = {4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8};
    static constexpr std::uint8_t w4[] = {8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6};
    static constexpr std::uint8_t w5[] = {4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6};
    static constexpr std::uint8_t w6[] = {4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5};
    static constexpr std::uint8_t w7[] = {4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5};
    switch (codec) {
    case 1: return {w1, 12};
    case 3: return {w3, 12};
    case 4: return {w4, 12};
    case 5: return {w5, 12};
    case 6: return {w6, 16};
    case 7: return {w7, 16};
    default: return {};
    }
}

// One (symbol width, q) frequency model: a probability per bit length of the symbol.
struct Model {
    std::uint32_t freq[tables::kMaxWidth + 1] = {}, base[tables::kMaxWidth + 1] = {};
    int width = -1;  // -1: the table is not known

    std::uint32_t lookup(std::uint32_t cf) const {
        int p = width;
        while (p > 0 && cf < base[p]) --p;
        return p == 0 ? 0 : (1u << (p - 1)) + (cf - base[p]) / freq[p];
    }
    std::uint64_t advance(std::uint32_t s, std::uint64_t x) const {
        const int p = int(std::bit_width(s));
        std::uint32_t f = freq[p], c = base[p] + f * (s - (p == 0 ? 0 : 1u << (p - 1)));
        if (c == 0) --f, c = 1;
        return std::uint64_t{f} * (x >> 16) + (x & 0xFFFF) - c;
    }
};

inline const Model* model_for(unsigned width, unsigned q) {
    static const auto models = [] {
        auto all = std::make_unique<std::array<Model, (tables::kMaxWidth + 1) * 16>>();
        for (unsigned w = 1; w <= tables::kMaxWidth; ++w)
            for (unsigned k = 0; k < 16; ++k) {
                const std::int32_t* f = tables::kProbability[w * 16 + k];
                Model& m = (*all)[w * 16 + k];
                if (f[0] < 0) continue;
                std::uint32_t sum = 0;
                for (unsigned p = 0; p <= w; ++p) {
                    m.freq[p] = std::uint32_t(f[p]);
                    m.base[p] = sum;
                    sum += m.freq[p] * (p == 0 ? 1u : 1u << (p - 1));
                }
                if (sum == 65536) m.width = int(w);
            }
        return all;
    }();
    if (width < 1 || width > tables::kMaxWidth) return nullptr;
    const Model& m = (*models)[width * 16 + q];
    return m.width < 0 ? nullptr : &m;
}

using Lanes = std::array<std::vector<std::uint16_t>, kMaxLanes>;

// The lane array, cycle by cycle as the hardware fetches: at most 8 input bytes a cycle, in lane order, so which byte
// belongs to which lane depends on every lane's progress. False, with `why`, for a stream that breaks a rule.
inline bool decode_lanes(const std::uint8_t* raw, std::size_t size, std::basic_string_view<std::uint8_t> widths,
                         std::size_t symbols, Lanes& out, const char*& why) {
    enum : std::uint8_t { header0, header1, header2, process };
    struct Lane {
        std::uint8_t control = header0, flush = 0, long_length = 0, q = 0, zeros = 0, input = 0;
        bool done = false, full = false, held = false, slot_full = false, stall = false;
        int width = 0;
        std::uint32_t symbol = 0, slot = 0, input_left = 0;
        std::size_t symbols_left = 0;
        std::uint64_t p0 = 0, p1 = 0;
        const Model* model = nullptr;
    };
    const int count = int(widths.size());
    Lane lanes[kMaxLanes];
    for (int i = 0; i < count; ++i) {
        lanes[i].symbols_left = symbols;
        lanes[i].done = symbols == 0;
        out[i].clear();
        out[i].reserve(symbols);
    }
    const auto fail = [&](const char* reason) {
        why = reason;
        return false;
    };
    std::size_t at = 0;
    // Every cycle of a good stream takes a byte or moves a symbol in some lane; this bounds a bad one.
    for (std::size_t cycle = 0, limit = 64 + 8 * (size + symbols * count); cycle < limit; ++cycle) {
        bool finished = true;
        for (int i = 0; i < count; ++i) finished &= lanes[i].done && !lanes[i].slot_full;
        if (finished) return true;

        // State update, prefetch and symbol release.
        int budget = 8;
        for (int i = 0; i < count; ++i) {
            Lane& l = lanes[i];
            if (l.done || (l.held && l.slot_full)) {
                l.stall = true;
                continue;
            }
            l.stall = false;
            if (l.control == process) l.p0 = l.held ? l.model->advance(l.symbol, l.p1) : l.p1;
            if (!l.full && (l.control != process || l.input_left > 0)) {
                if (budget == 0) {
                    l.stall = true;
                    continue;
                }
                if (at >= size) return fail("the input ran out");
                l.input = raw[at++];
                l.full = true;
                --budget;
            }
            if (l.held) {
                l.slot_full = true;
                l.slot = l.symbol;
                l.held = false;
            }
        }

        // Headers, byte merge and symbol lookup.
        for (int i = 0; i < count; ++i) {
            Lane& l = lanes[i];
            if (l.done || l.stall) continue;
            if (l.control != process) {
                if (!l.full) continue;
                l.full = false;
                if (l.control == header0) {
                    l.q = l.input & 15;
                    l.flush = (l.input >> 4) & 1;
                    l.long_length = (l.input >> 5) & 1;
                    l.width = int(widths[i]) - (l.input >> 6);
                    l.control = header1;
                    l.zeros = 0;
                } else if (l.control == header1) {
                    l.model = model_for(l.width > 0 ? unsigned(l.width) : 0, l.q);
                    if (!l.model) return fail("no probability table for a lane's width and q");
                    l.input_left = l.long_length ? l.input : l.input + 1u;
                    l.control = l.long_length ? header2 : process;
                } else {
                    l.input_left = ((std::uint32_t{l.input} << 8) | (l.input_left & 0xFF)) + 257;
                    l.control = process;
                }
                continue;
            }
            std::uint64_t state = l.p0;
            const auto pad = [&] {
                while (state < kHome && state != 0 && l.zeros < 8) {
                    state <<= 8;
                    ++l.zeros;
                }
            };
            if (state < kHome) {
                if (l.input_left > 0) {
                    if (l.full) {
                        state = (state << 8) | l.input;
                        --l.input_left;
                        l.full = false;
                        if (l.input_left == 0 && l.flush) pad();
                    }
                } else if (l.flush) {
                    if (state == 0 || l.zeros >= 8) return fail("a fragment did not end at its home state");
                    pad();
                } else {
                    l.p1 = state;
                    if (l.symbols_left > 0) l.control = header0;
                    else l.done = true;
                    continue;
                }
            }
            const bool next = state > kHome && l.symbols_left > 0;
            if (next) {
                l.symbol = l.model->lookup(std::uint32_t(state & 0xFFFF));
                --l.symbols_left;
                if (l.symbol >> l.width) return fail("a symbol is outside its width");
            }
            l.held = next;
            l.p1 = state;
            if (l.input_left == 0) {
                if (l.flush) {
                    if (state == kHome) {
                        if (l.symbols_left > 0) {
                            l.control = header0;
                            l.p0 = l.p1 = 0;
                        } else {
                            l.done = true;
                        }
                    } else if (state > kHome && !next) {
                        return fail("a fragment did not end at its home state");
                    }
                } else if (l.symbols_left == 0 && !next) {
                    l.done = true;
                } else if (state == kHome) {
                    return fail("a fragment did not end at its home state");
                }
            } else if (state == kHome || (l.symbols_left == 0 && !next && state > kHome)) {
                return fail("a fragment did not end at its home state");
            }
        }

        // One element once every active lane has its symbol.
        bool ready = true, any = false;
        for (int i = 0; i < count && ready; ++i) {
            if (lanes[i].slot_full) any = true;
            else if (!(lanes[i].done && !lanes[i].held)) ready = false;
        }
        if (ready && any)
            for (int i = 0; i < count; ++i)
                if (lanes[i].slot_full) {
                    out[i].push_back(std::uint16_t(lanes[i].slot));
                    lanes[i].slot_full = false;
                }
        for (int i = 0; i < count; ++i) {
            Lane& l = lanes[i];
            if (l.done && l.held && !l.slot_full) {
                l.slot_full = true;
                l.slot = l.symbol;
                l.held = false;
            }
        }
    }
    return fail("the stream does not end");
}

inline int unzig(int v) { return v & 1 ? -((v + 1) >> 1) : v >> 1; }
inline int interleave(int a, int b, int bits) {
    int d = 0;
    for (int k = 0; k < bits; ++k) d |= ((a >> k) & 1) << (2 * k) | ((b >> k) & 1) << (2 * k + 1);
    return d;
}
// The even (odd = 0) or odd bits of an 8-bit symbol.
inline int half(int s, int odd) {
    s >>= odd;
    return (s & 1) | (s >> 1 & 2) | (s >> 2 & 4) | (s >> 3 & 8);
}

// A 128-bit block written from its low bit up.
struct Bits {
    std::uint64_t word[2] = {};
    unsigned at = 0;
    void put(std::uint64_t value, unsigned bits) {
        for (unsigned b = 0; b < bits; ++b, ++at)
            if (at < 128) word[at >> 6] |= (value >> b & 1) << (at & 63);
    }
    void set(unsigned position) { word[position >> 6] |= std::uint64_t{1} << (position & 63); }
};

// The four rows of a BC4-style index block (two 3-bit indices per pair symbol).
inline void alpha_indices(std::uint8_t* to, const std::uint8_t (*pairs)[2], const int rows[8]) {
    std::uint64_t bits = 0;
    for (int r = 0; r < 4; ++r) {
        const std::uint8_t *left = pairs[rows[2 * r] & 63], *right = pairs[rows[2 * r + 1] & 63];
        bits |= std::uint64_t(left[0] | left[1] << 3 | right[0] << 6 | right[1] << 9) << (12 * r);
    }
    for (int k = 0; k < 6; ++k) to[k] = std::uint8_t(bits >> (8 * k));
}
inline void colour_block(std::uint8_t* to, unsigned c0, unsigned c1, const std::uint8_t* quadrant, const int q[4]) {
    const unsigned a = quadrant[q[0] & 255], b = quadrant[q[1] & 255], c = quadrant[q[2] & 255], d = quadrant[q[3] & 255];
    const std::uint8_t bytes[8] = {std::uint8_t(c0), std::uint8_t(c0 >> 8), std::uint8_t(c1), std::uint8_t(c1 >> 8),
                                   std::uint8_t((a & 15) | (b & 15) << 4), std::uint8_t(a >> 4 | (b >> 4) << 4),
                                   std::uint8_t((c & 15) | (d & 15) << 4), std::uint8_t(c >> 4 | (d >> 4) << 4)};
    std::memcpy(to, bytes, 8);
}

// BC1, BC3, BC4, BC5: each block's endpoints are a delta from an earlier block's, up to 255 back, through the built-in
// history before the chunk's first block.
template <std::size_t Values>
struct History {
    std::vector<std::array<int, Values>> seen;
    std::vector<bool> lost;
    const std::int16_t (*built_in)[Values];
    std::array<int, Values> none;
    std::size_t guessed = 0;

    std::array<int, Values> reference(std::size_t block, int distance) {
        bool unknown = false;
        std::array<int, Values> base = none;
        if (distance && std::size_t(distance) <= block) {
            base = seen[block - distance];
            unknown = lost[block - distance];
        } else if (distance) {
            const std::int16_t* entry = built_in[256 + block - distance];
            unknown = entry[0] < 0;
            if (!unknown) std::copy(entry, entry + Values, base.begin());
        }
        lost.push_back(unknown);
        guessed += unknown;
        return base;
    }
};

inline std::size_t rebuild_bc1(const Lanes& l, std::size_t blocks, std::uint8_t* out) {
    static constexpr int mods[6] = {32, 64, 32, 32, 64, 32};
    History<6> history{{}, {}, tables::kBc1History, {}};
    for (std::size_t i = 0; i < blocks; ++i) {
        const int v = l[0][i];
        auto e = history.reference(i, v & 1 ? 256 - ((v + 1) >> 1) : v >> 1);
        for (int k = 0; k < 6; ++k) e[k] = (e[k] + unzig(l[2 + k][i])) & (mods[k] - 1);
        history.seen.push_back(e);
        const int q[4] = {l[8][i], l[9][i], l[10][i], l[11][i]};
        colour_block(out + 8 * i, e[0] << 11 | e[1] << 5 | e[2], e[3] << 11 | e[4] << 5 | e[5], tables::kBc1Quadrant, q);
    }
    return history.guessed;
}

inline std::size_t rebuild_bc3(const Lanes& l, std::size_t blocks, std::uint8_t* out) {
    static constexpr int mods[8] = {256, 256, 32, 64, 32, 32, 64, 32};
    History<8> history{{}, {}, tables::kBc3History, {128, 128, 0, 0, 0, 0, 0, 0}};
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::size_t a = 2 * i, b = a + 1;
        auto e = history.reference(i, unzig(interleave(l[0][a], l[0][b], 4)) & 255);
        const int delta[8] = {l[2][a], l[2][b], l[7][a], l[8][a], l[9][a], l[7][b], l[8][b], l[9][b]};
        for (int k = 0; k < 8; ++k) e[k] = (e[k] + unzig(delta[k])) & (mods[k] - 1);
        history.seen.push_back(e);
        std::uint8_t* to = out + 16 * i;
        to[0] = std::uint8_t(e[0]);
        to[1] = std::uint8_t(e[1]);
        const int rows[8] = {l[3][a], l[3][b], l[4][a], l[4][b], l[5][a], l[5][b], l[6][a], l[6][b]};
        alpha_indices(to + 2, tables::kBc3AlphaPair, rows);
        const int q[4] = {l[10][a], l[10][b], l[11][a], l[11][b]};
        colour_block(to + 8, e[2] << 11 | e[3] << 5 | e[4], e[5] << 11 | e[6] << 5 | e[7], tables::kBc3ColourQuadrant, q);
    }
    return history.guessed;
}

inline std::size_t rebuild_bc4(const Lanes& l, std::size_t blocks, std::uint8_t* out) {
    History<2> history{{}, {}, tables::kBc4History, {128, 128}};
    for (std::size_t i = 0; i < blocks; ++i) {
        const int v = l[0][i];
        auto e = history.reference(i, v & 1 ? 256 - ((v + 1) >> 1) : v >> 1);
        for (int k = 0; k < 2; ++k) e[k] = (e[k] + unzig(l[2 + k][i])) & 255;
        history.seen.push_back(e);
        std::uint8_t* to = out + 8 * i;
        to[0] = std::uint8_t(e[0]);
        to[1] = std::uint8_t(e[1]);
        int rows[8];
        for (int k = 0; k < 8; ++k) rows[k] = l[4 + k][i];
        alpha_indices(to + 2, tables::kBc4Pair, rows);
    }
    return history.guessed;
}

inline std::size_t rebuild_bc5(const Lanes& l, std::size_t blocks, std::uint8_t* out) {
    History<4> history{{}, {}, tables::kBc5History, {128, 128, 128, 128}};
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::size_t a = 2 * i, b = a + 1;
        auto e = history.reference(i, unzig(interleave(l[0][a], l[0][b], 4)) & 255);
        const int delta[4] = {l[2][a], l[2][b], l[3][a], l[3][b]};
        for (int k = 0; k < 4; ++k) e[k] = (e[k] + unzig(delta[k])) & 255;
        history.seen.push_back(e);
        for (int channel = 0; channel < 2; ++channel) {
            std::uint8_t* to = out + 16 * i + 8 * channel;
            to[0] = std::uint8_t(e[2 * channel]);
            to[1] = std::uint8_t(e[2 * channel + 1]);
            int rows[8];
            for (int r = 0; r < 4; ++r) rows[2 * r] = l[4 + 4 * channel + r][a], rows[2 * r + 1] = l[4 + 4 * channel + r][b];
            alpha_indices(to + 2, tables::kBc5Pair, rows);
        }
    }
    return history.guessed;
}

// The reference distances BC6H and BC7 share: lane 1 the index distance, lane 0 the endpoint distance from it.
inline void distances(const Lanes& l, std::size_t block, int& endpoint, int& index) {
    const std::size_t a = 2 * block, b = a + 1;
    index = interleave(l[1][a], l[1][b], 4);
    endpoint = (index + unzig(interleave(l[0][a], l[0][b], 4))) & 255;
}
// The four index deltas of row `y` of a lane.
inline void row_deltas(int sa, int sb, int y, int d[4]) {
    const int first = y < 3;  // the last row's pairs are the other way round
    d[0] = unzig(half(sa, first));
    d[1] = unzig(half(sa, !first));
    d[2] = unzig(half(sb, first));
    d[3] = unzig(half(sb, !first));
}

inline bool rebuild_bc6h(const Lanes& l, std::size_t blocks, std::uint8_t* out, std::size_t& guessed, const char*& why) {
    static constexpr std::uint8_t codes[15] = {0, 0x00, 0x01, 0x02, 0x06, 0x0A, 0x0E, 0x12, 0x16, 0x1A, 0x1E, 0x03, 0x07, 0x0B, 0x0F};
    struct Mode { std::uint8_t transformed, base, delta[3], regions; };
    static constexpr Mode modes[15] = {{}, {1, 10, {5, 5, 5}, 2}, {1, 7, {6, 6, 6}, 2}, {1, 11, {5, 4, 4}, 2},
                                       {1, 11, {4, 5, 4}, 2}, {1, 11, {4, 4, 5}, 2}, {1, 9, {5, 5, 5}, 2}, {1, 8, {6, 5, 5}, 2},
                                       {1, 8, {5, 6, 5}, 2}, {1, 8, {5, 5, 6}, 2}, {0, 6, {6, 6, 6}, 2}, {0, 10, {10, 10, 10}, 1},
                                       {1, 11, {9, 9, 9}, 1}, {1, 12, {8, 8, 8}, 1}, {1, 16, {4, 4, 4}, 1}};
    using Endpoints = std::array<std::array<int, 4>, 3>;
    constexpr Endpoints no_endpoints{{{0x8000, 0x200, 0x20, 0x20}, {0x8000, 0x200, 0x20, 0x20}, {0x8000, 0x200, 0x20, 0x20}}};
    std::vector<Endpoints> seen_e(blocks);
    std::vector<std::array<std::uint8_t, 16>> seen_i(blocks);
    std::vector<bool> lost_e(blocks), lost_i(blocks);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::size_t a = 2 * i, b = a + 1;
        if (l[15][a] > 13) {
            why = "a BC6H mode is outside 0..13";
            return false;
        }
        const int mode = l[15][a] + 1, partition = std::min<int>(l[15][b], 31);
        const Mode& m = modes[mode];
        int de, di;
        distances(l, i, de, di);
        Endpoints ref = no_endpoints;
        if (de && std::size_t(de) <= i) ref = seen_e[i - de], lost_e[i] = lost_e[i - de];
        else if (de) lost_e[i] = true;  // the built-in history is not known
        std::array<std::uint8_t, 16> iref;
        iref.fill(8);
        if (di && std::size_t(di) <= i) iref = seen_i[i - di], lost_i[i] = lost_i[i - di];
        else if (di) lost_i[i] = true;
        guessed += lost_e[i] || lost_i[i];

        int field[3][4];
        for (int c = 0; c < 3; ++c) {
            const int dc = m.delta[c], mask = (1 << dc) - 1, shift = dc < 6 ? 6 - dc : 0, w_shift = 16 - m.base;
            const int dz0 = unzig(interleave(l[2 + c][a], l[2 + c][b], 8)), dz1 = unzig(interleave(l[5 + c][a], l[5 + c][b], 5));
            const auto& r = ref[c];
            const int w = ((r[0] >> w_shift) + dz0) & ((1 << m.base) - 1), x = ((r[1] >> (10 - dc)) + dz1) & mask;
            const int y = ((r[2] >> shift) + unzig(l[8 + c][a])) & mask, z = ((r[3] >> shift) + unzig(l[8 + c][b])) & mask;
            const int x10 = x << (10 - dc);
            int stream_w = w;
            if (mode == 13) stream_w = (w & 0x3FF) | (w >> 10 & 1) << 11 | (w >> 11 & 1) << 10;
            if (mode == 14) {
                stream_w = w & 0x3FF;
                for (int k = 10; k < 16; ++k) stream_w |= (w >> k & 1) << (25 - k);
            }
            if (m.regions == 2) {
                seen_e[i][c] = {w << w_shift, x10, y << shift, z << shift};
            } else {
                const int v = (w & mask) ^ (m.transformed ? 1 << (dc - 1) : 0);
                seen_e[i][c] = {w << w_shift, x10, dc >= 6 ? v >> (dc - 6) : v << shift, x10 >> 4};
            }
            field[c][0] = stream_w;
            field[c][1] = x;
            field[c][2] = m.regions == 2 ? y : 0;
            field[c][3] = m.regions == 2 ? z : 0;
        }
        const int bits = m.regions == 2 ? 3 : 4;
        int index[16];
        for (int y = 0; y < 4; ++y) {
            int d[4];
            row_deltas(l[11 + y][a], l[11 + y][b], y, d);
            for (int x = 0; x < 4; ++x) {
                const int p = 4 * y + x;
                index[p] = ((iref[p] >> (4 - bits)) + d[x]) & ((1 << bits) - 1);
                seen_i[i][p] = std::uint8_t(index[p] << (4 - bits));
            }
        }
        Bits block;
        block.word[0] = codes[mode];
        const auto& plan = tables::kBc6hPlan[mode];
        for (int k = 0; k < plan.count; ++k) {
            const auto& f = plan.bit[k];
            const int value = f.channel < 0 ? partition : field[f.channel][f.slot];
            if (value >> f.bit & 1) block.set(f.position);
        }
        block.at = plan.header_bits;
        const int second = m.regions == 2 ? tables::kAnchor2[partition] : -1;
        for (int p = 0; p < 16; ++p) {
            const int n = p == 0 || p == second ? bits - 1 : bits;
            block.put(std::uint64_t(index[p] & ((1 << n) - 1)), unsigned(n));
        }
        std::memcpy(out + 16 * i, block.word, 16);
    }
    return true;
}

inline bool rebuild_bc7(const Lanes& l, std::size_t blocks, std::uint8_t* out, std::size_t& guessed, const char*& why) {
    static constexpr std::uint8_t subsets[8] = {3, 2, 3, 2, 1, 1, 1, 2}, colour[8] = {4, 6, 5, 7, 5, 7, 7, 5},
                                  alpha[8] = {0, 0, 0, 0, 6, 8, 7, 5}, p_bits[8] = {1, 2, 0, 1, 0, 0, 1, 1},
                                  index_bits[8] = {3, 3, 2, 2, 3, 2, 4, 2}, partition_bits[8] = {4, 6, 6, 6, 0, 0, 0, 6};
    static constexpr std::int8_t lane_of[3][4] = {{2, 3, 4, 5}, {6, 7, 8, 9}, {10, 5, 9, -1}};
    static constexpr std::uint8_t second_rows[4] = {6, 7, 8, 10};
    std::vector<std::array<std::uint8_t, 24>> seen_e(blocks);
    std::vector<std::array<std::uint8_t, 32>> seen_i(blocks);
    std::vector<bool> lost_e(blocks), lost_i(blocks);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::size_t a = 2 * i, b = a + 1;
        const int sa = l[15][a], sb = l[15][b];
        const int rank = sa < 32 && sb < 16 ? tables::kBc7Rank[sa * 16 + sb] : -1;
        if (rank < 0 || tables::kBc7Config[rank][0] > 7) {
            why = "a BC7 signature is outside the table";
            return false;
        }
        const int mode = tables::kBc7Config[rank][0], partition = tables::kBc7Config[rank][1],
                  rotation = tables::kBc7Config[rank][2], selection = tables::kBc7Config[rank][3];
        int de, di;
        distances(l, i, de, di);
        std::array<std::uint8_t, 24> ref{};
        std::array<std::uint8_t, 32> iref{};
        if (de && std::size_t(de) <= i) ref = seen_e[i - de], lost_e[i] = lost_e[i - de];
        else if (de) lost_e[i] = true;  // the built-in history is not known
        if (di && std::size_t(di) <= i) iref = seen_i[i - di], lost_i[i] = lost_i[i - di];
        else if (di) lost_i[i] = true;
        guessed += lost_e[i] || lost_i[i];

        const int ns = subsets[mode], cb = colour[mode], ab = alpha[mode], pk = p_bits[mode];
        int width[4] = {cb, cb, cb, ab};
        if (rotation) width[rotation - 1] = ab, width[3] = cb;
        int value[3][2][4] = {}, p_bit[3][2] = {};
        auto& ends = seen_e[i];
        for (int s = 0; s < ns; ++s) {
            for (int c = 0; c < 4; ++c) {
                const int lane = lane_of[s][c];
                if (!width[c] || lane < 0) continue;
                const int n = width[c] + (pk && c == 0), shift = 8 - n, mask = (1 << n) - 1;
                for (int e = 0; e < 2; ++e) {
                    value[s][e][c] = ((ref[8 * s + 4 * e + c] >> shift) + unzig(l[lane][e ? b : a])) & mask;
                    ends[8 * s + 4 * e + c] = std::uint8_t(value[s][e][c] << shift);
                }
            }
            for (int e = 0; e < 2; ++e) p_bit[s][e] = value[s][pk == 2 ? 0 : e][0] & 1;
        }
        for (int s = ns; s < 3; ++s) std::copy_n(ends.begin(), 8, ends.begin() + 8 * s);

        const int b1 = index_bits[mode];
        const bool two = mode == 4 || mode == 5;
        int s1[16], s2[16] = {};
        bool empty = mode == 0 && partition == 0;
        for (int y = 0; y < 4; ++y) {
            int d[4];
            row_deltas(l[11 + y][a], l[11 + y][b], y, d);
            for (int x = 0; x < 4; ++x) {
                const int p = 4 * y + x;
                s1[p] = ((iref[p] >> (4 - b1)) + d[x]) & ((1 << b1) - 1);
                seen_i[i][p] = std::uint8_t(s1[p] << (4 - b1));
                empty = empty && !s1[p];
            }
            if (!two) continue;
            row_deltas(l[second_rows[y]][a], l[second_rows[y]][b], y, d);
            for (int x = 0; x < 4; ++x) {
                const int p = 4 * y + x;
                s2[p] = ((iref[16 + p] >> 2) + d[x]) & 3;
                seen_i[i][16 + p] = std::uint8_t(s2[p] << 2);
            }
        }
        std::uint8_t* to = out + 16 * i;
        if (empty && std::all_of(ends.begin(), ends.end(), [](int v) { return v == 0; })) {
            std::memset(to, 0, 16);  // the reserved all-zero block
            continue;
        }
        Bits block;
        block.put(std::uint64_t{1} << mode, unsigned(mode) + 1);
        block.put(unsigned(partition), partition_bits[mode]);
        if (two) block.put(unsigned(rotation), 2);
        if (mode == 4) block.put(unsigned(selection), 1);
        for (int c = 0; c < 3; ++c) {
            const int from = rotation && c == rotation - 1 ? 3 : c;
            const int drop = pk && from == 0;
            for (int s = 0; s < ns; ++s)
                for (int e = 0; e < 2; ++e) block.put(unsigned(value[s][e][from] >> drop), unsigned(cb));
        }
        if (ab) {
            const int from = rotation ? rotation - 1 : 3;
            const int drop = pk && from == 0;
            for (int s = 0; s < ns; ++s)
                for (int e = 0; e < 2; ++e) block.put(unsigned(value[s][e][from] >> drop), unsigned(ab));
        }
        if (pk)
            for (int s = 0; s < ns; ++s)
                for (int e = 0; e < (pk == 1 ? 2 : 1); ++e) block.put(unsigned(p_bit[s][e]), 1);
        const int *first = mode == 4 ? s2 : s1, *other = mode == 4 ? s1 : s2;
        const int n1 = mode == 4 ? 2 : b1, n2 = !two ? 0 : mode == 4 ? 3 : 2;
        for (int p = 0; p < 16; ++p) {
            int subset = 0, anchor = 0;
            if (ns == 2) {
                subset = tables::kPartition2[partition] >> p & 1;
                anchor = subset ? tables::kAnchor2[partition] : 0;
            } else if (ns == 3) {
                subset = int(tables::kPartition3[partition] >> (2 * p) & 3);
                anchor = subset == 0 ? 0 : subset == 1 ? tables::kAnchor3a[partition] : tables::kAnchor3b[partition];
            }
            block.put(unsigned(first[p]), unsigned(p == anchor ? n1 - 1 : n1));
        }
        if (n2)
            for (int p = 0; p < 16; ++p) block.put(unsigned(other[p]), unsigned(p == 0 ? n2 - 1 : n2));
        std::memcpy(to, block.word, 16);
    }
    return true;
}

// Decodes a chunk into exactly `out_size` bytes of BCn blocks, in the order the surface stores them. `guessed`: how many
// blocks rest on an unknown built-in history entry. False, with `why`, for a stream that is not valid BCPack.
inline bool decode(const std::uint8_t* raw, std::size_t size, unsigned codec, std::uint8_t* out, std::size_t out_size,
                   std::size_t& guessed, const char*& why) {
    guessed = 0;
    const std::uint32_t bytes = block_bytes(codec);
    if (!bytes || out_size % bytes) {
        why = "the mode or the decoded size is not one of BCPack's";
        return false;
    }
    Lanes lanes;
    const auto widths = lane_widths(codec);
    const std::size_t symbols = out_size / 8;
    if (!decode_lanes(raw, size, widths, symbols, lanes, why)) return false;
    for (std::size_t k = 0; k < widths.size(); ++k)
        if (lanes[k].size() != symbols) {
            why = "a lane ended early";
            return false;
        }
    const std::size_t blocks = out_size / bytes;
    switch (codec) {
    case 1: guessed = rebuild_bc1(lanes, blocks, out); return true;
    case 3: guessed = rebuild_bc3(lanes, blocks, out); return true;
    case 4: guessed = rebuild_bc4(lanes, blocks, out); return true;
    case 5: guessed = rebuild_bc5(lanes, blocks, out); return true;
    case 6: return rebuild_bc6h(lanes, blocks, out, guessed, why);
    default: return rebuild_bc7(lanes, blocks, out, guessed, why);
    }
}

}  // namespace leonida::bcpack_dc
