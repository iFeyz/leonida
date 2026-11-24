// The BCPack decoder (src/bcpack.h) against tools/bcpack.py: synthetic chunks made by the test encoder of
// tests/test_bcpack.py and the bytes the Python decoder rebuilds from them. No game data.
//
//   bcpack_test                                   the fixtures
//   bcpack_test CODEC OUT_SIZE INPUT OUTPUT       decode one file, for a comparison with the Python decoder
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/bcpack_dadacap.h"

namespace {

int failures = 0, checks = 0;
void check(bool condition, const char* what) {
    ++checks;
    if (!condition) ++failures;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
}

std::vector<std::uint8_t> bytes(const std::string& hex) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) out.push_back(std::uint8_t(std::strtoul(hex.substr(i, 2).c_str(), nullptr, 16)));
    return out;
}

struct Fixture {
    unsigned codec;
    const char *packed, *blocks;
};
const Fixture fixtures[] = {
    {1,
     "171b101710141b1d0404030403040705bb02800e807501015887214f10000009041700c50413ffb8db0392c67a62ecbc18aa6366ffff"
     "1919dc06061916010804065fdc0406ffffd62baad804d404fc2bb1b610baf8ff51ffb1e5f2ae",
     "2007c0022b11ba5a20bf833c8aea0a2ef9f62cba1aca7c58"},
    {3,
     "141a1714191915120403080709080706650742271c1d3416b62e154821e6ff922e6c7508fffa722243863c65e341eab2277292ffd20e"
     "c415e90612151c0106090dae0f06340de898a3bb04dfff149c2cc09ca2ff52ff218aab8844d7dcffc37bffc741dfaa1aff396ed7c264"
     "3416b4fffe6e7e028ea8c478e208",
     "757880579d9ce81c000800a85d58b02380e3d21d9200e08cf1bda00f056309c0ed7180010053ff41f4cd9712545ea6b7"},
    {4,
     "18131e15111d101b04050705050704055c0f010304010402227e62d5410000960701ff3405ff9805ec1bb81211eb94b5907bfff3d2ff"
     "9cff58df13141c1905050506030e01010884c4458e43ff00e5aad26e137162f05d4678efff01bc59e5fce4ffda",
     "dc7989222746bb240879d6ca4d6093e4dc79956f8d893f1d"},
    {5,
     "1b1517141417161e04040707070707090614323f27040d0129d1c08b52975f053d33feba9efbfb8d4ed3a50da3a6528c605b5b4164ff"
     "608a561211161d07070608060b845f903832de49affbffc92e62f767c446ff2f099036425789f28ec58bba414943c4ffbf49c6bfaead"
     "1c99ef69c1c68787",
     "0e805cb3287802bc4db4c34901b8bd830eb7800f928607587fb4c0401c8009440e552f001c1980024d104b2002710d84"},
    {6,
     "171b19111d1c1415040308080c0907070111ec670101151069a7b87e12842b950051e3a8fdff1ff97c536ae74fecb9e285177d7dffbc"
     "7014ee07151011181d144507070809090714b701108001089906f0e10e3680501d6642fafb9eacff9e1b80ddc2c72a89b9ad90504f43"
     "ffcdf6d85cf5ffffffe8bd00689bed390d98d1d2ff8ca5924aff3d6efcbe92ceff05e2ffa749e63d090f6571005d4069ff6ae816117e"
     "897d331a",
     "e73a31f7e24bbfdc8a32495b7ded6cbf964ac5183fa0025082dd43347024f9c69a8ac784bedef0927229c233e46932ba"},
    {7,
     "131113171315131705050909080908080126014a0a050102d95454f35139470f1f03b29cbeef610ccc86b9b6566c1027bd5d8726c4e1"
     "3f1e191e191d10171e1a0a09080b07090b090301e40201d401187c0260e3012a633ffffffbff1afcfffffffaf9c84d53af5dff20e6ff"
     "ba61ff4b42003f777a025bf2ccff39ffc924fff5c720ffb58df67f9e2ef224faaa14710cd1cd00700237ffff109bfefe3c99eb72fff0"
     "e10c224e9db9bcb74830bc36590b9c2fe4eeff4c0d",
     "848f408002b8c046dd4f901b48352d0520f75c6b0412b04335d00e60ed110bc27e17963b34f036ed0a00c804f657d6b3"},
};

int decode_file(char** argv) {
    std::FILE* in = std::fopen(argv[3], "rb");
    if (!in) return 2;
    std::vector<std::uint8_t> raw, out(std::strtoull(argv[2], nullptr, 10));
    for (int c; (c = std::fgetc(in)) != EOF;) raw.push_back(std::uint8_t(c));
    std::fclose(in);
    std::size_t guessed = 0;
    const char* why = "";
    if (!leonida::bcpack_dc::decode(raw.data(), raw.size(), unsigned(std::atoi(argv[1])), out.data(), out.size(), guessed, why)) {
        std::printf("refused: %s\n", why);
        return 1;
    }
    std::printf("%zu guessed\n", guessed);
    std::FILE* to = std::fopen(argv[4], "wb");
    if (!to) return 2;
    std::fwrite(out.data(), 1, out.size(), to);
    std::fclose(to);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 5) return decode_file(argv);
    std::size_t guessed = 0;
    const char* why = nullptr;
    for (const Fixture& f : fixtures) {
        const auto raw = bytes(f.packed), want = bytes(f.blocks);
        std::vector<std::uint8_t> out(want.size());
        check(leonida::bcpack_dc::decode(raw.data(), raw.size(), f.codec, out.data(), out.size(), guessed, why) && out == want &&
                  guessed == 0,
              ("mode " + std::to_string(f.codec) + " decodes to the same blocks as the Python decoder").c_str());
        bool refused = true;
        for (std::size_t n = 0; n < raw.size(); ++n)  // a cut stream runs out of input, whatever lane it is in
            refused &= !leonida::bcpack_dc::decode(raw.data(), n, f.codec, out.data(), out.size(), guessed, why);
        check(refused, "every shorter stream is refused");
        for (std::size_t n = 0; n < raw.size(); ++n) {  // a damaged one ends, in either way
            auto bad = raw;
            bad[n] ^= 0x5A;
            leonida::bcpack_dc::decode(bad.data(), bad.size(), f.codec, out.data(), out.size(), guessed, why);
        }
        check(!leonida::bcpack_dc::decode(raw.data(), raw.size(), f.codec, out.data(), out.size() - 4, guessed, why),
              "a size that is not whole blocks is refused");
    }
    std::uint8_t one = 0;
    check(!leonida::bcpack_dc::decode(&one, 1, 2, &one, 0, guessed, why), "mode 2 (BC2) is not supported");
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
