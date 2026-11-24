// The runtime's BCPack decoder (src/bcpack.cpp) against the Python reference (tools/bcpack.py): every synthetic
// chunk of bcpack_vectors.h (runtime/test/bcpack_vectors.py) must give the lanes, blocks or refusal Python gives.
// No game files, no GPU.
#include "bcpack.h"
#include "bcpack_vectors.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

using leonida::bcpack::Status;

}  // namespace

int main() {
    namespace bp = leonida::bcpack;

    // The lane layer: the same streams accepted, the same symbols.
    int lane_ok = 0, lane_refused = 0;
    for (std::size_t k = 0; k < std::size(bcpack_vectors::kLaneCases); ++k) {
        const auto& c = bcpack_vectors::kLaneCases[k];
        bp::Lanes lanes;
        Status s = bp::decode_lanes({c.raw, c.raw_size}, bp::lane_widths(c.mode), c.count, lanes);
        // Python's decode_lanes also refuses non-zero bytes after the last lane; the vectors fold that in.
        if (s == Status::ok)
            for (std::size_t i = lanes.used; i < c.raw_size; ++i)
                if (c.raw[i]) s = Status::trailing_data;
        char what[96];
        std::snprintf(what, sizeof what, "lane case %zu (mode %u): accepted as Python does", k, c.mode);
        check((s == Status::ok) == c.ok, what);
        if (s == Status::ok && c.ok) {
            const std::size_t lanes_n = bp::lane_widths(c.mode).size();
            std::snprintf(what, sizeof what, "lane case %zu (mode %u): the same symbols", k, c.mode);
            check(lanes.symbols.size() == lanes_n * c.count &&
                      std::memcmp(lanes.symbols.data(), c.symbols, lanes.symbols.size() * sizeof(std::uint16_t)) == 0,
                  what);
        }
        (s == Status::ok ? lane_ok : lane_refused)++;
    }

    // Whole chunks: the same blocks, or the same class of refusal.
    int block_ok = 0, lenient_checked = 0, lenient_guessed = 0;
    for (std::size_t k = 0; k < std::size(bcpack_vectors::kBlockCases); ++k) {
        const auto& c = bcpack_vectors::kBlockCases[k];
        std::vector<std::uint8_t> out(c.out_size);
        const Status s = bp::decode({c.raw, c.raw_size}, c.mode, out);
        char what[128];
        std::snprintf(what, sizeof what, "block case %zu (mode %u): status %d (%s), Python %d", k, c.mode,
                      static_cast<int>(s), bp::status_name(s), c.status);
        check(static_cast<int>(s) == c.status, what);
        if (s == Status::ok && c.status == 0) {
            std::snprintf(what, sizeof what, "block case %zu (mode %u): the same blocks", k, c.mode);
            check(std::memcmp(out.data(), c.blocks, c.out_size) == 0, what);
            ++block_ok;
        }
        // Lenient: where Python's provisional decode succeeds, the same bytes; and lenient never refuses a chunk the
        // strict decode accepts, nor changes its bytes, nor counts a guessed block in it.
        std::vector<std::uint8_t> loose(c.out_size);
        const bp::Result r = bp::decode({c.raw, c.raw_size}, c.mode, loose, bp::Options{nullptr, true});
        if (c.lenient_known) {
            std::snprintf(what, sizeof what, "block case %zu (mode %u): lenient gives Python's provisional blocks (%s)",
                          k, c.mode, bp::status_name(r.status));
            check(r.status == Status::ok && std::memcmp(loose.data(), c.lenient_blocks, c.out_size) == 0, what);
            ++lenient_checked;
        }
        if (c.status == 0) {
            std::snprintf(what, sizeof what, "block case %zu (mode %u): lenient agrees with strict", k, c.mode);
            check(r.status == Status::ok && r.guessed == 0 && loose == out, what);
        }
        if (r.status == Status::ok && r.guessed) ++lenient_guessed;
    }

    // Refusals that need no stream.
    std::vector<std::uint8_t> out(32);
    const std::uint8_t none[1] = {0};
    check(bp::decode(none, 2, out) == Status::unsupported_codec, "mode 2 (BC2) has no BCPack form");
    check(bp::decode(none, 8, out) == Status::unsupported_codec, "mode 8 does not exist");
    std::vector<std::uint8_t> odd(12);
    check(bp::decode(none, 1, odd) == Status::bad_size, "12 bytes are not whole BC1 blocks");
    check(bp::block_bytes(1) == 8 && bp::block_bytes(5) == 16 && bp::block_bytes(9) == 0, "block sizes");
    std::vector<std::uint8_t> empty;
    check(bp::decode({}, 1, empty) == Status::ok, "an empty BC1 chunk decodes to nothing");

    // External history: an entry the built-in table lacks makes a refused chunk decode when the user's table has it.
    bool tried = false;
    for (const auto& c : bcpack_vectors::kBlockCases) {
        if (c.status != static_cast<int>(Status::unconfirmed_history) || c.mode != 1) continue;
        bp::History history;
        history.values[1].assign(256 * 6, 0);  // every entry known, all zero
        std::vector<std::uint8_t> blocks(c.out_size);
        const Status s = bp::decode({c.raw, c.raw_size}, c.mode, blocks, &history);
        check(s == Status::ok, "a BC1 chunk refused for its history decodes with a complete external table");
        tried = true;
        break;
    }
    check(tried, "the vectors hold a BC1 chunk refused for its history");

    check(lenient_guessed > 0, "the vectors hold a chunk lenient decoding takes with guessed blocks");
    std::printf("bcpack: %d lane cases accepted, %d refused; %d block cases decoded, %zu checked; lenient: %d "
                "against Python's provisional decode, %d with guessed blocks\n",
                lane_ok, lane_refused, block_ok, std::size(bcpack_vectors::kBlockCases), lenient_checked,
                lenient_guessed);
    if (g_failures) std::printf("%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
