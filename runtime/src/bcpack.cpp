// BCPack decoder: see bcpack.h. The lane layer below follows tools/bcpack_calibration/src/model.rs statement by
// statement (that file is a line-by-line port of `_decode_lanes_raw` in tools/bcpack.py), so the three decoders
// accept the same streams and give the same symbols. Where model.rs returns None, this returns Status::bad_stream.
//
// model.rs keeps the coder state in an i128 because the Python model works on unbounded integers. The values stay
// far inside 64 bits: a state below RANS_L (2^24) is shifted left by one byte at a time until it reaches RANS_L, so it
// stays below 2^32, and the advance step is at most freq (<= 2^16) times the state's top 16 bits. A negative state
// (only a corrupt stream makes one) is refused before it is shifted, as model.rs does.
#include "bcpack.h"

#include "bcpack_tables.h"

#include <algorithm>
#include <bit>

namespace leonida::bcpack {
namespace {

constexpr std::int64_t kRansL = std::int64_t{1} << 24;
constexpr int kMaxWidth = tables::kMaxWidth;
constexpr int kSide = kMaxWidth + 1;
constexpr std::uint32_t kPrefetchBudget = 8;
constexpr std::uint32_t kMaxVirtualZeros = 8;
constexpr std::size_t kElementBytes = 8;  // bytes one symbol tuple of every lane stands for

enum Ctrl : std::uint8_t { hdr0, hdr1, hdr2, proc };

// The rANS models: freq/base per band for every (width, q), built once from tables::kProb as model.rs's Tables::new.
struct Models {
    struct Model {
        bool present = false;
        std::array<std::int64_t, kSide> freq{};
        std::array<std::int64_t, kSide> base{};
    };
    std::array<Model, kSide * 16> m{};
    bool valid = true;

    Models() {
        for (int w = 0; w <= kMaxWidth; ++w)
            for (int q = 0; q < 16; ++q) {
                const std::int32_t* row = &tables::kProb[static_cast<std::size_t>((w * 16 + q) * kSide)];
                Model& model = m[static_cast<std::size_t>(w * 16 + q)];
                if (row[0] < 0) continue;
                std::int64_t acc = 0;
                for (int p = 0; p <= w; ++p) {
                    model.freq[p] = row[p];
                    model.base[p] = acc;
                    acc += model.freq[p] * (p == 0 ? 1 : (std::int64_t{1} << (p - 1)));
                }
                model.present = true;
                if (acc != 65536) valid = false;
            }
    }

    // The symbol whose band holds cumulative frequency `cf`; -1 where model.rs's lookup returns None.
    std::int64_t lookup(std::size_t index, std::int64_t cf) const {
        const Model& model = m[index];
        int p = static_cast<int>(index / 16);
        while (p > 0 && cf < model.base[p]) --p;
        if (p == 0) return 0;
        if (model.freq[p] == 0) return -1;
        return (std::int64_t{1} << (p - 1)) + (cf - model.base[p]) / model.freq[p];
    }

    std::int64_t advance(std::size_t index, std::int64_t s, std::int64_t x) const {
        const Model& model = m[index];
        const int p = static_cast<int>(std::bit_width(static_cast<std::uint64_t>(s)));
        std::int64_t f = model.freq[p];
        std::int64_t c = model.base[p] + f * (s - (p == 0 ? 0 : (std::int64_t{1} << (p - 1))));
        if (c == 0) {
            f -= 1;
            c = 1;
        }
        return f * (x >> 16) + (x & 0xFFFF) - c;
    }
};

const Models& models() {
    static const Models instance;
    return instance;
}

std::int64_t unzig(std::uint16_t v) {
    const std::int64_t x = v;
    return (x & 1) ? -((x + 1) >> 1) : (x >> 1);
}

// The history distance of BC1 and BC4 (one element per block): 0 for none, 1..255 back.
std::int64_t dist(std::uint16_t v) {
    const std::int64_t x = v;
    return (x & 1) ? 256 - ((x + 1) >> 1) : (x >> 1);
}

// The 8-bit history distance of the two-element codecs (`_inter8`, then zigzag modulo 256).
std::int64_t dist8(std::uint16_t a, std::uint16_t b) {
    std::uint16_t code = 0;
    for (int k = 0; k < 4; ++k)
        code = static_cast<std::uint16_t>(code | (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1)));
    return unzig(code) & 255;
}

std::int64_t rem(std::int64_t v, std::int64_t m) {
    const std::int64_t r = v % m;
    return r < 0 ? r + m : r;
}

// The decode in progress: the policy, and per block whether it rests on an unknown history entry (`lost`).
struct Ctx {
    bool lenient = false;
    std::vector<std::uint8_t> lost;
    std::size_t guessed() const { return static_cast<std::size_t>(std::count(lost.begin(), lost.end(), 1)); }
};

// The base of block i: the zero state, an earlier block, or a confirmed history entry (external first). An entry
// nobody has confirmed refuses the chunk (strict) or gives the zero state and marks the block lost (lenient); a block
// that predicts from a lost block is lost too.
template <std::size_t K>
Status base_of(std::size_t i, std::int64_t d, const std::array<std::int64_t, K>& zero,
               const std::vector<std::array<std::int64_t, K>>& hist, std::span<const std::int16_t> builtin,
               std::span<const std::int16_t> external, Ctx& ctx, std::array<std::int64_t, K>& out) {
    if (d == 0) {
        out = zero;
        return Status::ok;
    }
    if (static_cast<std::int64_t>(i) - d >= 0) {
        out = hist[i - static_cast<std::size_t>(d)];
        ctx.lost[i] = ctx.lost[i - static_cast<std::size_t>(d)];
        return Status::ok;
    }
    const std::size_t entry = static_cast<std::size_t>(256 + static_cast<std::int64_t>(i) - d);
    for (std::span<const std::int16_t> table : {external, builtin}) {
        if (table.size() < (entry + 1) * K || table[entry * K] < 0) continue;
        for (std::size_t k = 0; k < K; ++k) out[k] = table[entry * K + k];
        return Status::ok;
    }
    if (!ctx.lenient) return Status::unconfirmed_history;
    out = zero;
    ctx.lost[i] = 1;
    return Status::ok;
}

std::span<const std::int16_t> external_for(const History* history, std::uint32_t mode) {
    if (!history || mode >= history->values.size()) return {};
    return history->values[mode];
}

Status pair_bits(std::span<const std::uint8_t> pairs, std::uint16_t left, std::uint16_t right, std::uint64_t& bits) {
    if (left >= 64 || right >= 64) return Status::bad_symbol;
    bits = static_cast<std::uint64_t>(pairs[2u * left]) | static_cast<std::uint64_t>(pairs[2u * left + 1]) << 3 |
           static_cast<std::uint64_t>(pairs[2u * right]) << 6 | static_cast<std::uint64_t>(pairs[2u * right + 1]) << 9;
    return Status::ok;
}

void put16(std::uint8_t* o, std::int64_t v) {
    o[0] = static_cast<std::uint8_t>(v & 0xFF);
    o[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
}

void put48(std::uint8_t* o, std::uint64_t bits) {
    for (int k = 0; k < 6; ++k) o[k] = static_cast<std::uint8_t>(bits >> (8 * k));
}

bool lane_is_zero(const Lanes& lanes, std::size_t lane) {
    for (std::size_t i = 0; i < lanes.count; ++i)
        if (lanes.at(lane, i)) return false;
    return true;
}

Status bc1(const Lanes& l, std::span<const std::int16_t> external, Ctx& ctx, std::span<std::uint8_t> out) {
    if (!lane_is_zero(l, 1)) return Status::lane1_nonzero;
    static constexpr std::array<std::int64_t, 6> mods{32, 64, 32, 32, 64, 32};
    const std::size_t n = l.count;
    std::vector<std::array<std::int64_t, 6>> hist(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<std::int64_t, 6> b{};
        if (const Status s = base_of<6>(i, dist(l.at(0, i)), {}, hist, tables::kBc1History, external, ctx, b);
            s != Status::ok)
            return s;
        std::array<std::int64_t, 6> e{};
        for (std::size_t k = 0; k < 6; ++k) e[k] = rem(b[k] + unzig(l.at(2 + k, i)), mods[k]);
        hist[i] = e;
        for (std::size_t lane = 8; lane < 12; ++lane)
            if (l.at(lane, i) > 255) return Status::bad_symbol;
        const auto q = [&](std::size_t lane) { return tables::kBc1Quadrant[l.at(lane, i)]; };
        const std::uint8_t a = q(8), bb = q(9), c = q(10), dd = q(11);
        std::uint8_t* o = out.data() + i * 8;
        put16(o, (e[0] << 11) | (e[1] << 5) | e[2]);
        put16(o + 2, (e[3] << 11) | (e[4] << 5) | e[5]);
        o[4] = static_cast<std::uint8_t>((a & 15) | ((bb & 15) << 4));
        o[5] = static_cast<std::uint8_t>((a >> 4) | ((bb >> 4) << 4));
        o[6] = static_cast<std::uint8_t>((c & 15) | ((dd & 15) << 4));
        o[7] = static_cast<std::uint8_t>((c >> 4) | ((dd >> 4) << 4));
    }
    return Status::ok;
}

Status bc4(const Lanes& l, std::span<const std::int16_t> external, Ctx& ctx, std::span<std::uint8_t> out) {
    if (!lane_is_zero(l, 1)) return Status::lane1_nonzero;
    const std::size_t n = l.count;
    std::vector<std::array<std::int64_t, 2>> hist(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<std::int64_t, 2> b{};
        if (const Status s = base_of<2>(i, dist(l.at(0, i)), {128, 128}, hist, tables::kBc4History, external, ctx, b);
            s != Status::ok)
            return s;
        const std::int64_t a0 = (b[0] + unzig(l.at(2, i))) & 255, a1 = (b[1] + unzig(l.at(3, i))) & 255;
        hist[i] = {a0, a1};
        std::uint64_t bits = 0;
        for (std::size_t r = 0; r < 4; ++r) {
            std::uint64_t v = 0;
            if (const Status s = pair_bits(tables::kBc4Pair, l.at(4 + 2 * r, i), l.at(5 + 2 * r, i), v);
                s != Status::ok)
                return s;
            bits |= v << (12 * r);
        }
        std::uint8_t* o = out.data() + i * 8;
        o[0] = static_cast<std::uint8_t>(a0);
        o[1] = static_cast<std::uint8_t>(a1);
        put48(o + 2, bits);
    }
    return Status::ok;
}

// BC3 and BC5 carry two symbols of every lane per block, so the lane count is twice the block count.
Status bc3(const Lanes& l, std::span<const std::int16_t> external, Ctx& ctx, std::span<std::uint8_t> out) {
    if (l.count % 2) return Status::bad_size;
    if (!lane_is_zero(l, 1)) return Status::lane1_nonzero;
    static constexpr std::array<std::int64_t, 8> mods{256, 256, 32, 64, 32, 32, 64, 32};
    const std::size_t blocks = l.count / 2;
    std::vector<std::array<std::int64_t, 8>> hist(blocks);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::size_t a = 2 * i, b = 2 * i + 1;
        std::array<std::int64_t, 8> base{};
        if (const Status s = base_of<8>(i, dist8(l.at(0, a), l.at(0, b)), {128, 128, 0, 0, 0, 0, 0, 0}, hist,
                                        tables::kBc3History, external, ctx, base);
            s != Status::ok)
            return s;
        const std::array<std::uint16_t, 8> dz{l.at(2, a), l.at(2, b), l.at(7, a), l.at(8, a),
                                              l.at(9, a), l.at(7, b), l.at(8, b), l.at(9, b)};
        std::array<std::int64_t, 8> e{};
        for (std::size_t k = 0; k < 8; ++k) e[k] = rem(base[k] + unzig(dz[k]), mods[k]);
        hist[i] = e;
        std::uint64_t bits = 0;
        for (std::size_t r = 0; r < 4; ++r) {
            std::uint64_t v = 0;
            if (const Status s = pair_bits(tables::kBc3AlphaPair, l.at(3 + r, a), l.at(3 + r, b), v); s != Status::ok)
                return s;
            bits |= v << (12 * r);
        }
        for (std::size_t lane = 10; lane < 12; ++lane)
            if (l.at(lane, a) > 255 || l.at(lane, b) > 255) return Status::bad_symbol;
        const auto q = [&](std::size_t lane, std::size_t at) { return tables::kBc3ColourQuadrant[l.at(lane, at)]; };
        const std::uint8_t qa = q(10, a), qb = q(10, b), qc = q(11, a), qd = q(11, b);
        std::uint8_t* o = out.data() + i * 16;
        o[0] = static_cast<std::uint8_t>(e[0]);
        o[1] = static_cast<std::uint8_t>(e[1]);
        put48(o + 2, bits);
        put16(o + 8, (e[2] << 11) | (e[3] << 5) | e[4]);
        put16(o + 10, (e[5] << 11) | (e[6] << 5) | e[7]);
        o[12] = static_cast<std::uint8_t>((qa & 15) | ((qb & 15) << 4));
        o[13] = static_cast<std::uint8_t>((qa >> 4) | ((qb >> 4) << 4));
        o[14] = static_cast<std::uint8_t>((qc & 15) | ((qd & 15) << 4));
        o[15] = static_cast<std::uint8_t>((qc >> 4) | ((qd >> 4) << 4));
    }
    return Status::ok;
}

Status bc5(const Lanes& l, std::span<const std::int16_t> external, Ctx& ctx, std::span<std::uint8_t> out) {
    if (l.count % 2) return Status::bad_size;
    if (!lane_is_zero(l, 1)) return Status::lane1_nonzero;
    const std::size_t blocks = l.count / 2;
    std::vector<std::array<std::int64_t, 4>> hist(blocks);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::size_t a = 2 * i, b = 2 * i + 1;
        std::array<std::int64_t, 4> base{};
        if (const Status s = base_of<4>(i, dist8(l.at(0, a), l.at(0, b)), {128, 128, 128, 128}, hist,
                                        tables::kBc5History, external, ctx, base);
            s != Status::ok)
            return s;
        const std::array<std::int64_t, 4> e{(base[0] + unzig(l.at(2, a))) & 255, (base[1] + unzig(l.at(2, b))) & 255,
                                            (base[2] + unzig(l.at(3, a))) & 255, (base[3] + unzig(l.at(3, b))) & 255};
        hist[i] = e;
        std::uint8_t* o = out.data() + i * 16;
        for (std::size_t ch = 0; ch < 2; ++ch) {
            std::uint64_t bits = 0;
            for (std::size_t r = 0; r < 4; ++r) {
                std::uint64_t v = 0;
                if (const Status s = pair_bits(tables::kBc5Pair, l.at(4 + 4 * ch + r, a), l.at(4 + 4 * ch + r, b), v);
                    s != Status::ok)
                    return s;
                bits |= v << (12 * r);
            }
            o[8 * ch] = static_cast<std::uint8_t>(e[2 * ch]);
            o[8 * ch + 1] = static_cast<std::uint8_t>(e[2 * ch + 1]);
            put48(o + 8 * ch + 2, bits);
        }
    }
    return Status::ok;
}

// BC7 --------------------------------------------------------------------------------------------------------------
// `_bc7_check` then `_bc7_rebuild` of tools/bcpack.py, in that order, so a chunk is refused for the same reason. The
// built-in BC7 history is all unknown (tools/bcpack_tables_bptc.py has no bit of it), so a reference before the
// chunk's first block is refused (strict) or decoded from the zero state, which is what Python's provisional mode
// rebuilds too (lenient). Per-mode constants: `_B7_*`.
constexpr std::array<int, 8> kB7Subsets{3, 2, 3, 2, 1, 1, 1, 2};
constexpr std::array<int, 8> kB7Colour{4, 6, 5, 7, 5, 7, 7, 5};
constexpr std::array<int, 8> kB7Alpha{0, 0, 0, 0, 6, 8, 7, 5};
constexpr std::array<int, 8> kB7PBits{1, 2, 0, 1, 0, 0, 1, 1};  // 1: per endpoint, 2: shared per subset
constexpr std::array<int, 8> kB7IndexBits{3, 3, 2, 2, 3, 2, 4, 2};
constexpr std::array<int, 8> kB7PartitionBits{4, 6, 6, 6, 0, 0, 0, 6};
constexpr int kB7Lane[3][4] = {{2, 3, 4, 5}, {6, 7, 8, 9}, {10, 5, 9, -1}};

// The 8-bit value `_inter8` makes of two 4-bit symbols.
int inter8(std::uint16_t a, std::uint16_t b) {
    int code = 0;
    for (int k = 0; k < 4; ++k) code |= (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1));
    return code;
}

// The even (odd = false) or odd bits of an 8-bit symbol, packed (`_DEINT`).
int deint(int s, bool odd) {
    const int o = odd ? 1 : 0;
    return ((s >> o) & 1) | (((s >> (2 + o)) & 1) << 1) | (((s >> (4 + o)) & 1) << 2) | (((s >> (6 + o)) & 1) << 3);
}

// A 128-bit BC7 block written from its low bit up.
struct Bits128 {
    std::uint64_t word[2] = {0, 0};
    int pos = 0;
    void put(std::uint64_t value, int bits) {
        if (bits <= 0) return;
        value &= (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
        for (int k = 0; k < bits; ++k, ++pos)
            if (pos < 128 && ((value >> k) & 1)) word[pos >> 6] |= std::uint64_t{1} << (pos & 63);
    }
};

int bc7_rank(std::uint16_t a, std::uint16_t b) {
    return a < 32 && b < 16 ? tables::kBc7Rank[a * 16u + b] : -1;
}

bool bc7_inferred(int rank) {
    return std::find(tables::kBc7Inferred.begin(), tables::kBc7Inferred.end(), rank) != tables::kBc7Inferred.end();
}

Status bc7(const Lanes& l, Ctx& ctx, std::span<std::uint8_t> out) {
    const std::size_t n = l.count / 2;
    // _bc7_check: every signature in the table, none inferred, no reference into the built-in history.
    for (std::size_t i = 0; i < n; ++i) {
        const int rank = bc7_rank(l.at(15, 2 * i), l.at(15, 2 * i + 1));
        if (rank < 0 || tables::kBc7Config[rank][0] > 7 || tables::kBc7Config[rank][1] == 255) return Status::bad_block;
    }
    if (!ctx.lenient) {
        for (std::size_t i = 0; i < n; ++i)
            if (bc7_inferred(bc7_rank(l.at(15, 2 * i), l.at(15, 2 * i + 1)))) return Status::unconfirmed_history;
        for (std::size_t i = 0; i < std::min<std::size_t>(n, 256); ++i) {
            const int di = inter8(l.at(1, 2 * i), l.at(1, 2 * i + 1));
            const int de = (di + static_cast<int>(unzig(static_cast<std::uint16_t>(inter8(l.at(0, 2 * i),
                                                                                          l.at(0, 2 * i + 1)))))) & 255;
            if (static_cast<std::size_t>(de) > i || static_cast<std::size_t>(di) > i)
                return Status::unconfirmed_history;
        }
    }

    // _bc7_rebuild.
    std::vector<std::array<int, 24>> eps(n);
    std::vector<std::array<int, 32>> ixs(n);
    std::vector<std::uint8_t> lost_e(n, 0), lost_i(n, 0);
    bool inconsistent = false;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t a = 2 * i, b = 2 * i + 1;
        const int rank = bc7_rank(l.at(15, a), l.at(15, b));
        const auto& cfg = tables::kBc7Config[static_cast<std::size_t>(rank)];
        const int mode = cfg[0], part = cfg[1], rot = cfg[2], isel = cfg[3];
        const int di = inter8(l.at(1, a), l.at(1, b));
        const int de = (di + static_cast<int>(unzig(static_cast<std::uint16_t>(inter8(l.at(0, a), l.at(0, b)))))) & 255;
        std::array<int, 24> rep{};
        std::array<int, 32> rix{};
        if (de && static_cast<std::size_t>(de) <= i) {
            rep = eps[i - static_cast<std::size_t>(de)];
            lost_e[i] = lost_e[i - static_cast<std::size_t>(de)];
        } else if (de) {
            lost_e[i] = 1;  // the built-in entry is unknown: the zero state
        }
        if (di && static_cast<std::size_t>(di) <= i) {
            rix = ixs[i - static_cast<std::size_t>(di)];
            lost_i[i] = lost_i[i - static_cast<std::size_t>(di)];
        } else if (di) {
            lost_i[i] = 1;
        }
        ctx.lost[i] = lost_e[i] || lost_i[i];

        const int ns = kB7Subsets[mode], cb = kB7Colour[mode], ab = kB7Alpha[mode], pk = kB7PBits[mode];
        int w[4] = {cb, cb, cb, ab};
        if (rot) {
            w[rot - 1] = ab;
            w[3] = cb;
        }
        int val[3][2][4] = {};
        int pb[3][2] = {};
        bool bad = false;
        std::array<int, 24> sep{};
        for (int sub = 0; sub < ns; ++sub) {
            for (int c = 0; c < 4; ++c) {
                const int lane = kB7Lane[sub][c];
                if (!w[c] || lane < 0) continue;
                const int bits = w[c] + ((pk && c == 0) ? 1 : 0), shift = 8 - bits, mask = (1 << bits) - 1;
                const int v0 = ((rep[sub * 8 + c] >> shift) + static_cast<int>(unzig(l.at(static_cast<std::size_t>(lane), a)))) & mask;
                const int v1 = ((rep[sub * 8 + 4 + c] >> shift) + static_cast<int>(unzig(l.at(static_cast<std::size_t>(lane), b)))) & mask;
                val[sub][0][c] = v0;
                val[sub][1][c] = v1;
                sep[sub * 8 + c] = v0 << shift;
                sep[sub * 8 + 4 + c] = v1 << shift;
            }
            if (pk == 1) {
                pb[sub][0] = val[sub][0][0] & 1;
                pb[sub][1] = val[sub][1][0] & 1;
            } else if (pk == 2) {
                pb[sub][0] = pb[sub][1] = val[sub][0][0] & 1;
                if ((val[sub][1][0] & 1) != pb[sub][0]) bad = true;
            }
        }
        for (int sub = ns; sub < 3; ++sub)
            for (int k = 0; k < 8; ++k) sep[sub * 8 + k] = sep[k];
        eps[i] = sep;

        const int b1 = kB7IndexBits[mode], b1_shift = 4 - b1, b1_mask = (1 << b1) - 1;
        const bool two = mode == 4 || mode == 5;
        int s1[16] = {}, s2[16] = {};
        std::array<int, 32> six{};
        static constexpr std::size_t rows1[4] = {11, 12, 13, 14}, rows2[4] = {6, 7, 8, 10};
        for (int y = 0; y < 4; ++y) {
            const bool first_odd = y < 3;  // the last row's pairs are the other way round
            const int sa = l.at(rows1[y], a), sb = l.at(rows1[y], b);
            const int d1[4] = {deint(sa, first_odd), deint(sa, !first_odd), deint(sb, first_odd), deint(sb, !first_odd)};
            for (int x = 0; x < 4; ++x) {
                const int p = y * 4 + x;
                s1[p] = ((rix[p] >> b1_shift) + static_cast<int>(unzig(static_cast<std::uint16_t>(d1[x])))) & b1_mask;
                six[p] = s1[p] << b1_shift;
            }
            if (!two) continue;
            const int ta = l.at(rows2[y], a), tb = l.at(rows2[y], b);
            const int d2[4] = {deint(ta, first_odd), deint(ta, !first_odd), deint(tb, first_odd), deint(tb, !first_odd)};
            for (int x = 0; x < 4; ++x) {
                const int p = y * 4 + x;
                s2[p] = ((rix[16 + p] >> 2) + static_cast<int>(unzig(static_cast<std::uint16_t>(d2[x])))) & 3;
                six[16 + p] = s2[p] << 2;
            }
        }
        ixs[i] = six;

        std::uint8_t* to = out.data() + 16 * i;
        std::fill(to, to + 16, std::uint8_t{0});
        const bool empty_endpoints = std::all_of(sep.begin(), sep.end(), [](int v) { return v == 0; });
        const bool empty_indices = std::all_of(std::begin(s1), std::end(s1), [](int v) { return v == 0; });
        if (mode == 0 && part == 0 && empty_endpoints && empty_indices) {
            inconsistent = inconsistent || bad;
            continue;  // the reserved all-zero block
        }

        int anchors[3] = {0, 0, 0};
        if (ns == 2) {
            anchors[1] = tables::kBc7Anchor2[static_cast<std::size_t>(part)];
        } else if (ns == 3) {
            anchors[1] = tables::kBc7Anchor3a[static_cast<std::size_t>(part)];
            anchors[2] = tables::kBc7Anchor3b[static_cast<std::size_t>(part)];
        }
        const auto subset_of = [&](int p) -> int {
            if (ns == 2) return (tables::kBc7Partition2[static_cast<std::size_t>(part)] >> p) & 1;
            if (ns == 3) return static_cast<int>((tables::kBc7Partition3[static_cast<std::size_t>(part)] >> (2 * p)) & 3);
            return 0;
        };
        const int* pri = s1;
        const int* sec = nullptr;
        int ib = b1, ib2 = 0;
        if (mode == 4) {
            pri = s2, ib = 2, sec = s1, ib2 = 3;
        } else if (mode == 5) {
            pri = s1, ib = 2, sec = s2, ib2 = 2;
        }

        Bits128 block;
        block.put(std::uint64_t{1} << mode, mode + 1);
        if (kB7PartitionBits[mode]) block.put(static_cast<std::uint64_t>(part), kB7PartitionBits[mode]);
        if (two) block.put(static_cast<std::uint64_t>(rot), 2);
        if (mode == 4) block.put(static_cast<std::uint64_t>(isel), 1);
        for (int c = 0; c < 3; ++c) {
            const int tc = (rot && c == rot - 1) ? 3 : c;
            const bool drop = pk && tc == 0;
            for (int sub = 0; sub < ns; ++sub)
                for (int e = 0; e < 2; ++e) block.put(static_cast<std::uint64_t>(drop ? val[sub][e][tc] >> 1 : val[sub][e][tc]), cb);
        }
        if (ab) {
            const int tc = rot ? rot - 1 : 3;
            const bool drop = pk && tc == 0;
            for (int sub = 0; sub < ns; ++sub)
                for (int e = 0; e < 2; ++e) block.put(static_cast<std::uint64_t>(drop ? val[sub][e][tc] >> 1 : val[sub][e][tc]), ab);
        }
        if (pk == 1) {
            for (int sub = 0; sub < ns; ++sub) {
                block.put(static_cast<std::uint64_t>(pb[sub][0]), 1);
                block.put(static_cast<std::uint64_t>(pb[sub][1]), 1);
            }
        } else if (pk == 2) {
            for (int sub = 0; sub < ns; ++sub) block.put(static_cast<std::uint64_t>(pb[sub][0]), 1);
        }
        for (int p = 0; p < 16; ++p) {
            const int pv = pri[p];
            if (p == anchors[subset_of(p)]) {
                if (pv >> (ib - 1)) bad = true;
                block.put(static_cast<std::uint64_t>(pv), ib - 1);
            } else {
                block.put(static_cast<std::uint64_t>(pv), ib);
            }
        }
        if (sec) {
            for (int p = 0; p < 16; ++p) {
                const int sv = sec[p];
                if (p == 0) {
                    if (sv >> (ib2 - 1)) bad = true;
                    block.put(static_cast<std::uint64_t>(sv), ib2 - 1);
                } else {
                    block.put(static_cast<std::uint64_t>(sv), ib2);
                }
            }
        }
        if (block.pos != 128) return Status::bad_block;  // `_B7_SIGNATURE`: refused in every mode
        for (int k = 0; k < 2; ++k)
            for (int byte = 0; byte < 8; ++byte) to[8 * k + byte] = static_cast<std::uint8_t>(block.word[k] >> (8 * byte));
        inconsistent = inconsistent || bad;
        if (bad && ctx.lenient) ctx.lost[i] = 1;
    }
    if (inconsistent && !ctx.lenient) return Status::bad_block;  // the anchor / p-bit consistency check
    return Status::ok;
}

// BC6H -------------------------------------------------------------------------------------------------------------
// `_bc6h_blocks` of tools/bcpack.py. The header bit layout of each mode comes from `_bc6h_plan` through the table
// header (kBc6hPlan), so the D3D11 layout strings are not parsed twice. The built-in BC6H history is all unknown: a
// reference before the chunk's first block is refused (strict) or read as zeros, as Python's provisional mode does
// (lenient). Refusals in Python's order: a mode symbol past 13 (at once), then the history, then the anchor indices.

// `_inter(a, b, n)`: the bits of a and b interleaved, a's first.
int inter(int a, int b, int n) {
    int d = 0;
    for (int k = 0; k < n; ++k) d |= (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1));
    return d;
}

// `_bc6h_stream_w`: BCPack's base endpoint in the D3D11 bit order of modes 13 and 14.
std::int64_t bc6h_stream_w(int mode, std::int64_t w) {
    if (mode == 13) return (w & 0x3FF) | (((w >> 10) & 1) << 11) | (((w >> 11) & 1) << 10);
    if (mode == 14) {
        std::int64_t out = w & 0x3FF;
        for (int b = 10; b < 16; ++b) out |= ((w >> b) & 1) << (25 - b);
        return out;
    }
    return w;
}

Status bc6h(const Lanes& l, Ctx& ctx, std::span<std::uint8_t> out) {
    using Endpoints = std::array<std::array<std::int64_t, 4>, 3>;
    static constexpr std::array<std::int64_t, 4> kNoneChannel{0x8000, 0x200, 0x20, 0x20};
    const std::size_t n = l.count / 2;
    std::vector<Endpoints> st_e(n);
    std::vector<std::array<int, 16>> st_i(n);
    std::vector<std::uint8_t> lost_e(n, 0), lost_i(n, 0);
    bool unknown_reference = false;
    std::size_t violations = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t a = 2 * i, b = 2 * i + 1;
        const int m_sym = l.at(15, a);
        if (m_sym > 13) return Status::bad_block;
        const int mode = m_sym + 1;
        const tables::Bc6hMode& md = tables::kBc6hModes[static_cast<std::size_t>(mode)];
        const int part = std::min<int>(l.at(15, b), 31);
        const int di = inter8(l.at(1, a), l.at(1, b));
        const int de = (di + static_cast<int>(unzig(static_cast<std::uint16_t>(inter8(l.at(0, a), l.at(0, b)))))) & 255;

        Endpoints eref{};
        if (de == 0) {
            eref = {kNoneChannel, kNoneChannel, kNoneChannel};
        } else if (static_cast<std::size_t>(de) <= i) {
            eref = st_e[i - static_cast<std::size_t>(de)];
            lost_e[i] = lost_e[i - static_cast<std::size_t>(de)];
        } else {
            lost_e[i] = 1;  // the built-in entry is unknown: zeros
            unknown_reference = true;
        }
        std::array<int, 16> iref{};
        if (di == 0) {
            iref.fill(8);
        } else if (static_cast<std::size_t>(di) <= i) {
            iref = st_i[i - static_cast<std::size_t>(di)];
            lost_i[i] = lost_i[i - static_cast<std::size_t>(di)];
        } else {
            lost_i[i] = 1;
            unknown_reference = true;
        }

        const int regions = md.regions, base = md.base, w_shift = 16 - base;
        const std::int64_t mw = (std::int64_t{1} << base) - 1;
        std::int64_t f[3][4] = {};
        Endpoints st{};
        for (int c = 0; c < 3; ++c) {
            const int dc = md.delta[c];
            const std::int64_t m_d = (std::int64_t{1} << dc) - 1;
            const int sh = dc < 6 ? 6 - dc : 0;
            const std::int64_t dz0 = unzig(static_cast<std::uint16_t>(inter(l.at(2 + c, a), l.at(2 + c, b), 8)));
            const std::int64_t dz1 = unzig(static_cast<std::uint16_t>(inter(l.at(5 + c, a), l.at(5 + c, b), 5)));
            const std::int64_t dz2 = unzig(l.at(8 + c, a)), dz3 = unzig(l.at(8 + c, b));
            const auto& r = eref[static_cast<std::size_t>(c)];
            const std::int64_t w = ((r[0] >> w_shift) + dz0) & mw;
            const std::int64_t x = ((r[1] >> (10 - dc)) + dz1) & m_d;
            const std::int64_t y = ((r[2] >> sh) + dz2) & m_d;
            const std::int64_t z = ((r[3] >> sh) + dz3) & m_d;
            const std::int64_t x10 = x << (10 - dc);
            if (regions == 2) {
                st[static_cast<std::size_t>(c)] = {w << w_shift, x10, y << sh, z << sh};
                f[c][0] = bc6h_stream_w(mode, w), f[c][1] = x, f[c][2] = y, f[c][3] = z;
            } else {
                const std::int64_t v = (w & m_d) ^ (md.transformed ? (std::int64_t{1} << (dc - 1)) : 0);
                const std::int64_t low_y = dc >= 6 ? v >> (dc - 6) : v << sh;
                st[static_cast<std::size_t>(c)] = {w << w_shift, x10, low_y, x10 >> 4};
                f[c][0] = bc6h_stream_w(mode, w), f[c][1] = x;
            }
        }
        st_e[i] = st;

        const int ibits = regions == 2 ? 3 : 4, imask = (1 << ibits) - 1, ishift = 4 - ibits;
        std::array<int, 16> ix{}, si{};
        for (int y = 0; y < 4; ++y) {
            const bool first_odd = y < 3;  // the last row's pairs are the other way round
            const int sa = l.at(static_cast<std::size_t>(11 + y), a), sb = l.at(static_cast<std::size_t>(11 + y), b);
            const int d[4] = {deint(sa, first_odd), deint(sa, !first_odd), deint(sb, first_odd), deint(sb, !first_odd)};
            for (int x = 0; x < 4; ++x) {
                const std::size_t p = static_cast<std::size_t>(y * 4 + x);
                ix[p] = ((iref[p] >> ishift) + static_cast<int>(unzig(static_cast<std::uint16_t>(d[x])))) & imask;
                si[p] = ix[p] << ishift;
            }
        }
        st_i[i] = si;

        const int second = regions == 2 ? tables::kBc7Anchor2[static_cast<std::size_t>(part)] : -1;
        bool bad = false;
        if (ix[0] & (regions == 2 ? 4 : 8)) ++violations, bad = true;
        if (regions == 2 && (ix[static_cast<std::size_t>(second)] & 4)) ++violations, bad = true;

        std::uint64_t word[2] = {md.code, 0};
        const auto set = [&](int pos) { word[pos >> 6] |= std::uint64_t{1} << (pos & 63); };
        for (std::size_t k = md.plan_start; k < static_cast<std::size_t>(md.plan_start) + md.plan_size; ++k) {
            const tables::Bc6hStep& step = tables::kBc6hPlan[k];
            const std::int64_t value = step.c < 0 ? part : f[step.c][step.q];
            if ((value >> step.bit) & 1) set(step.pos);
        }
        int at = md.header;
        for (int p = 0; p < 16; ++p) {
            const int nb = (p == 0 || p == second) ? ibits - 1 : ibits;
            for (int k = 0; k < nb; ++k, ++at)
                if ((ix[static_cast<std::size_t>(p)] >> k) & 1) set(at);
        }
        std::uint8_t* to = out.data() + 16 * i;
        for (int k = 0; k < 2; ++k)
            for (int byte = 0; byte < 8; ++byte) to[8 * k + byte] = static_cast<std::uint8_t>(word[k] >> (8 * byte));
        ctx.lost[i] = lost_e[i] || lost_i[i] || (bad && ctx.lenient);
    }
    if (!ctx.lenient) {
        if (unknown_reference) return Status::unconfirmed_history;
        if (violations) return Status::bad_block;
    }
    return Status::ok;
}

}  // namespace

const char* status_name(Status status) {
    switch (status) {
        case Status::ok: return "ok";
        case Status::unsupported_codec: return "unsupported BCPack mode";
        case Status::bad_size: return "decoded size is not a whole number of blocks";
        case Status::bad_stream: return "corrupt or truncated BCPack stream";
        case Status::trailing_data: return "non-zero data after the last lane finished";
        case Status::lane1_nonzero: return "lane 1 is non-zero";
        case Status::bad_symbol: return "symbol outside its lookup table";
        case Status::unconfirmed_history: return "block refers to an unconfirmed built-in history entry";
        case Status::bad_block: return "BC7 block fails its signature or consistency check";
    }
    return "unknown status";
}

std::span<const std::uint8_t> lane_widths(std::uint32_t mode) {
    switch (mode) {
        case 1: return tables::kLaneWidths1;
        case 3: return tables::kLaneWidths3;
        case 4: return tables::kLaneWidths4;
        case 5: return tables::kLaneWidths5;
        case 6: return tables::kLaneWidths6;
        case 7: return tables::kLaneWidths7;
        default: return {};
    }
}

std::uint32_t block_bytes(std::uint32_t mode) {
    switch (mode) {
        case 1:
        case 4: return 8;
        case 3:
        case 5:
        case 6:
        case 7: return 16;
        default: return 0;
    }
}

Status decode_lanes(std::span<const std::uint8_t> raw, std::span<const std::uint8_t> widths, std::size_t nsym,
                    Lanes& result) {
    const Models& t = models();
    if (!t.valid) return Status::bad_stream;
    const std::size_t nl = widths.size();
    std::vector<std::uint8_t> ctrl(nl, hdr0);
    std::vector<std::size_t> out_rem(nl, nsym), w(nl, 0), q(nl, 0), model(nl, SIZE_MAX);
    std::vector<std::uint8_t> h3(nl, 0), flush(nl, 0), done(nl, nsym == 0), full(nl, 0), held(nl, 0), slotfull(nl, 0),
        stall(nl, 0);
    std::vector<std::uint32_t> vz(nl, 0);
    std::vector<std::int64_t> inbuf(nl, 0), sym(nl, 0), in_rem(nl, 0), p0(nl, 0), p1(nl, 0), slotval(nl, 0);
    std::vector<std::vector<std::uint16_t>> out(nl);
    for (auto& lane : out) lane.reserve(nsym);
    std::size_t pos = 0;
    std::uint32_t idle = 0;  // model.rs hands a stream on which nothing moves for 2^16 cycles back to Python
    constexpr std::size_t kBadWidth = SIZE_MAX;

    for (;;) {
        bool finished = true;
        for (std::size_t i = 0; i < nl; ++i)
            if (!done[i] || slotfull[i]) finished = false;
        if (finished) {
            result.symbols.clear();
            result.symbols.reserve(nl * nsym);
            for (const auto& lane : out) {
                if (lane.size() != nsym) return Status::bad_stream;
                result.symbols.insert(result.symbols.end(), lane.begin(), lane.end());
            }
            result.count = nsym;
            result.used = pos;
            return Status::ok;
        }
        std::size_t produced = 0;
        for (const auto& lane : out) produced += lane.size();
        const std::size_t before_pos = pos, before_out = produced;
        bool moved = false;

        std::uint32_t left = kPrefetchBudget;
        for (std::size_t i = 0; i < nl; ++i) {
            if (done[i]) {
                stall[i] = 1;
                continue;
            }
            if (held[i] && slotfull[i]) {
                stall[i] = 1;
                continue;
            }
            stall[i] = 0;
            if (ctrl[i] == proc) p0[i] = held[i] ? t.advance(model[i], sym[i], p1[i]) : p1[i];
            if (!full[i] && (ctrl[i] != proc || in_rem[i] > 0)) {
                if (left == 0) {
                    stall[i] = 1;
                    continue;
                }
                if (pos >= raw.size()) return Status::bad_stream;
                inbuf[i] = raw[pos];
                ++pos;
                full[i] = 1;
                --left;
            }
            if (held[i]) {
                slotfull[i] = 1;
                slotval[i] = sym[i];
                held[i] = 0;
                moved = true;
            }
        }

        for (std::size_t i = 0; i < nl; ++i) {
            if (done[i] || stall[i]) continue;
            switch (ctrl[i]) {
                case hdr0: {
                    if (!full[i]) continue;
                    const std::int64_t h = inbuf[i];
                    q[i] = static_cast<std::size_t>(h & 15);
                    flush[i] = (h >> 4) & 1;
                    h3[i] = (h >> 5) & 1;
                    const std::int64_t wi = static_cast<std::int64_t>(widths[i]) - (h >> 6);
                    w[i] = (wi < 1 || wi > kMaxWidth) ? kBadWidth : static_cast<std::size_t>(wi);
                    full[i] = 0;
                    ctrl[i] = hdr1;
                    vz[i] = 0;
                    moved = true;
                    break;
                }
                case hdr1: {
                    if (!full[i]) continue;
                    full[i] = 0;
                    if (w[i] == kBadWidth) return Status::bad_stream;
                    const std::size_t index = w[i] * 16 + q[i];
                    if (!t.m[index].present) return Status::bad_stream;
                    model[i] = index;
                    if (h3[i]) {
                        in_rem[i] = inbuf[i];
                        ctrl[i] = hdr2;
                    } else {
                        in_rem[i] = inbuf[i] + 1;
                        ctrl[i] = proc;
                    }
                    moved = true;
                    break;
                }
                case hdr2: {
                    if (!full[i]) continue;
                    in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257;
                    ctrl[i] = proc;
                    full[i] = 0;
                    moved = true;
                    break;
                }
                default: {
                    std::int64_t st = p0[i];
                    if (st < 0) return Status::bad_stream;  // only a corrupt stream gets here
                    if (st < kRansL) {
                        if (in_rem[i] > 0) {
                            if (full[i]) {
                                st = (st << 8) | inbuf[i];
                                --in_rem[i];
                                full[i] = 0;
                                if (in_rem[i] == 0 && flush[i])
                                    while (st < kRansL && st != 0 && vz[i] < kMaxVirtualZeros) {
                                        st <<= 8;
                                        ++vz[i];
                                    }
                            }
                        } else if (flush[i]) {
                            if (st == 0 || vz[i] >= kMaxVirtualZeros) return Status::bad_stream;
                            while (st < kRansL && st != 0 && vz[i] < kMaxVirtualZeros) {
                                st <<= 8;
                                ++vz[i];
                            }
                        } else {
                            p1[i] = st;
                            if (out_rem[i] > 0)
                                ctrl[i] = hdr0;
                            else
                                done[i] = 1;
                            moved = true;
                            continue;
                        }
                    }
                    const bool nxt = st > kRansL && out_rem[i] > 0;
                    if (nxt) {
                        const std::int64_t s = t.lookup(model[i], st & 0xFFFF);
                        if (s < 0) return Status::bad_stream;
                        sym[i] = s;
                        --out_rem[i];
                        if ((s >> w[i]) != 0) return Status::bad_stream;
                    }
                    if (nxt != static_cast<bool>(held[i]) || p1[i] != st) moved = true;
                    held[i] = nxt;
                    p1[i] = st;
                    if (in_rem[i] == 0) {
                        if (flush[i]) {
                            if (st == kRansL) {
                                if (out_rem[i] > 0) {
                                    ctrl[i] = hdr0;
                                    p0[i] = 0;
                                    p1[i] = 0;
                                } else {
                                    done[i] = 1;
                                }
                                moved = true;
                            } else if (st > kRansL && !nxt) {
                                return Status::bad_stream;
                            }
                        } else if (out_rem[i] == 0 && !nxt) {
                            done[i] = 1;
                            moved = true;
                        }
                    } else if (st == kRansL || (out_rem[i] == 0 && !nxt && st > kRansL)) {
                        return Status::bad_stream;
                    }
                }
            }
        }

        bool ready = true, anyfull = false;
        for (std::size_t i = 0; i < nl; ++i) {
            if (slotfull[i]) {
                anyfull = true;
            } else if (!(done[i] && !held[i])) {
                ready = false;
                break;
            }
        }
        if (ready && anyfull)
            for (std::size_t i = 0; i < nl; ++i)
                if (slotfull[i]) {
                    out[i].push_back(static_cast<std::uint16_t>(slotval[i]));
                    slotfull[i] = 0;
                }
        for (std::size_t i = 0; i < nl; ++i)
            if (done[i] && held[i] && !slotfull[i]) {
                slotfull[i] = 1;
                slotval[i] = sym[i];
                held[i] = 0;
                moved = true;
            }
        produced = 0;
        for (const auto& lane : out) produced += lane.size();
        if (pos != before_pos || produced != before_out || moved) {
            idle = 0;
        } else if (++idle > (1u << 16)) {
            return Status::bad_stream;
        }
    }
}

Result decode(std::span<const std::uint8_t> in, std::uint32_t mode, std::span<std::uint8_t> out,
              const Options& options) {
    const std::uint32_t bytes = block_bytes(mode);
    if (block_bytes(mode) == 0) return {Status::unsupported_codec};
    if (out.size() % bytes) return {Status::bad_size};
    Lanes lanes;
    if (const Status s = decode_lanes(in, lane_widths(mode), out.size() / kElementBytes, lanes); s != Status::ok)
        return {s};
    if (std::any_of(in.begin() + static_cast<std::ptrdiff_t>(lanes.used), in.end(), [](std::uint8_t b) { return b; }))
        return {Status::trailing_data};
    const std::span<const std::int16_t> external = external_for(options.history, mode);
    Ctx ctx;
    ctx.lenient = options.lenient;
    ctx.lost.assign(out.size() / bytes, 0);
    Status s;
    switch (mode) {
        case 1: s = bc1(lanes, external, ctx, out); break;
        case 3: s = bc3(lanes, external, ctx, out); break;
        case 4: s = bc4(lanes, external, ctx, out); break;
        case 5: s = bc5(lanes, external, ctx, out); break;
        case 6: s = bc6h(lanes, ctx, out); break;
        default: s = bc7(lanes, ctx, out); break;
    }
    return {s, s == Status::ok ? ctx.guessed() : 0};
}

Status decode(std::span<const std::uint8_t> in, std::uint32_t mode, std::span<std::uint8_t> out,
              const History* history) {
    return decode(in, mode, out, Options{history, false}).status;
}

}  // namespace leonida::bcpack
