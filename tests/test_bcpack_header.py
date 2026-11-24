"""runtime/src/bcpack_tables.h and runtime/test/bcpack_vectors.h are what their generators write today.

The runtime's BCPack decoder (runtime/src/bcpack.cpp) reads the codec tables from bcpack_tables.h, which
tools/bcpack_header.py writes from tools/bcpack_tables.py, and bcpack_test compares it with tools/bcpack.py through
the chunks of bcpack_vectors.h (runtime/test/bcpack_vectors.py). A change to the Python tables or decoder without new
headers would leave the C++ decoder checked against stale answers; these two checks catch that. No game files.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _check(script):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, *script), "--check"],
                          capture_output=True, text=True, timeout=600)


class BcpackHeadersTest(unittest.TestCase):
    def test_tables_header_is_current(self):
        r = _check(("tools", "bcpack_header.py"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_vectors_header_is_current(self):
        r = _check(("runtime", "test", "bcpack_vectors.py"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_tables_header_keeps_provisional_entries_out(self):
        # Strict decoding: an entry only the provisional tables hold must stay -1 in the header.
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import bcpack_tables as t  # noqa: PLC0415
        with open(os.path.join(ROOT, "runtime", "src", "bcpack_tables.h"), encoding="utf-8") as f:
            text = f.read()
        for codec, k in ((1, 6), (3, 8), (4, 2), (5, 4)):
            confirmed = getattr(t, f"BC{codec}_HISTORY_CONFIRMED")
            provisional = getattr(t, f"BC{codec}_HISTORY_PROVISIONAL")
            start = text.index(f"kBc{codec}History = {{") + len(f"kBc{codec}History = {{")
            values = [int(v) for v in text[start:text.index("};", start)].replace("\n", " ").split(",") if v.strip()]
            for entry in provisional:
                if entry not in confirmed:
                    with self.subTest(codec=codec, entry=entry):
                        self.assertEqual(values[entry * k:entry * k + k], [-1] * k)


if __name__ == "__main__":
    unittest.main()
