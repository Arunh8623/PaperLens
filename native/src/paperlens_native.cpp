// PaperLens optional C++ accelerator. Every function here has a pure-Python twin in
// src/paperlens/native.py with the SAME arguments and results; tests/test_native.py compares them.
//
// What it does and why (plain language):
//   profile (docs/evidence/profile-20261101-*.json) showed 57% of search time inside the Python varint decoder.
//   decode_postings        : read (gap, tf) varint pairs -> [(doc, tf)]            (same as text/varint.py)
//   bm25_accumulate        : decode a postings list AND add each document's BM25 contribution into a dense
//                            score array, skipping deleted / filtered-out documents  (the hot loop of search)
//   level_dot              : for packed 1-4 bit vector codes, dot(levels, query) without ever materialising
//                            the (n x dim) float matrix
// NOT implemented (honest): block-max WAND (our postings have no block-max metadata yet) and a popcount scan
// with a quantised query (it would change results relative to the Python reference).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

constexpr int MAX_VARINT_BYTES = 10;

// Same behaviour and messages as text/varint.py::decode_uvarint.
inline uint64_t read_uvarint(const uint8_t* p, size_t n, size_t& pos) {
    uint64_t result = 0;
    unsigned shift = 0;
    for (int i = 0; i < MAX_VARINT_BYTES; i++) {
        if (pos + static_cast<size_t>(i) >= n) throw std::runtime_error("truncated varint");
        uint8_t b = p[pos + i];
        uint64_t low = b & 0x7F;
        if (!(b & 0x80)) {
            if (shift == 63 && low > 1) throw std::runtime_error("varint exceeds 64 bits");
            result |= low << shift;
            pos += static_cast<size_t>(i) + 1;
            return result;
        }
        if (shift == 63) break;  // 10th byte still says "more": too long (Python also stops after 10 groups)
        result |= low << shift;
        shift += 7;
    }
    throw std::runtime_error("varint too long");
}

struct Buf {
    py::buffer_info info;
    const uint8_t* data;
    size_t size;
    explicit Buf(const py::buffer& b) : info(b.request()) {
        if (info.itemsize != 1 || info.ndim != 1) throw std::invalid_argument("expected a 1-D bytes-like object");
        data = static_cast<const uint8_t*>(info.ptr);
        size = static_cast<size_t>(info.shape[0]);
    }
};

py::tuple decode_postings(const py::buffer& buf, size_t pos, size_t count) {
    Buf b(buf);
    py::list out;
    int64_t prev = -1;
    for (size_t i = 0; i < count; i++) {
        uint64_t gap = read_uvarint(b.data, b.size, pos);
        uint64_t tf = read_uvarint(b.data, b.size, pos);
        if (gap < 1 || tf < 1) throw std::runtime_error("corrupt postings (gap or tf is zero)");
        prev += static_cast<int64_t>(gap);
        out.append(py::make_tuple(prev, tf));
    }
    return py::make_tuple(out, pos);
}

// out[doc] += idf*tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)), written in EXACTLY the operation order of the
// Python reference so the doubles are bit-identical.
size_t bm25_accumulate(const py::buffer& postings, size_t pos, size_t count, const py::buffer& doclens,
                       py::object dead_obj, py::object mask_obj, double idf, double k1, double b, double avgdl,
                       py::array_t<double, py::array::c_style> out) {
    Buf pb(postings);
    py::buffer_info dl = doclens.request();
    if (dl.itemsize != 4 || dl.ndim != 1) throw std::invalid_argument("doclens must be a 1-D array of uint32");
    const uint32_t* doclen = static_cast<const uint32_t*>(dl.ptr);
    const int64_t n = dl.shape[0];
    if (out.ndim() != 1 || out.shape(0) != n) throw std::invalid_argument("out must be a float64 array with one entry per document");
    double* acc = out.mutable_data();

    py::buffer_info dead_info, mask_info;
    const uint8_t* dead = nullptr;
    const uint8_t* mask = nullptr;
    if (!dead_obj.is_none()) {
        dead_info = py::cast<py::buffer>(dead_obj).request();
        if (dead_info.itemsize != 1 || dead_info.shape[0] != n) throw std::invalid_argument("dead mask has the wrong shape");
        dead = static_cast<const uint8_t*>(dead_info.ptr);
    }
    if (!mask_obj.is_none()) {
        mask_info = py::cast<py::buffer>(mask_obj).request();
        if (mask_info.itemsize != 1 || mask_info.shape[0] != n) throw std::invalid_argument("filter mask has the wrong shape");
        mask = static_cast<const uint8_t*>(mask_info.ptr);
    }
    int64_t prev = -1;
    size_t scored = 0;
    for (size_t i = 0; i < count; i++) {
        uint64_t gap = read_uvarint(pb.data, pb.size, pos);
        uint64_t tf = read_uvarint(pb.data, pb.size, pos);
        if (gap < 1 || tf < 1) throw std::runtime_error("corrupt postings (gap or tf is zero)");
        prev += static_cast<int64_t>(gap);
        if (prev >= n) throw std::out_of_range("document id out of range");
        if ((dead && dead[prev]) || (mask && !mask[prev])) continue;
        double dtf = static_cast<double>(tf);
        double ddl = static_cast<double>(doclen[prev]);
        double s = idf * dtf * (k1 + 1.0) / (dtf + k1 * (1.0 - b + b * ddl / avgdl));
        acc[prev] += s;
        scored++;
    }
    return scored;
}

// levels u = 2c + 1 - 2^bits where c is the `bits`-bit code of each dimension, bit j of c (LSB first) stored at
// flat bit position d*bits + j of a most-significant-bit-first bit stream (numpy.packbits order).
py::array_t<float> level_dot(py::array_t<uint8_t, py::array::c_style> packed, py::array_t<float, py::array::c_style> q,
                             int dim, int bits) {
    if (bits < 1 || bits > 4) throw std::invalid_argument("bits must be 1, 2, 3 or 4");
    if (packed.ndim() != 2 || q.ndim() != 1 || q.shape(0) != dim) throw std::invalid_argument("bad shapes");
    const int64_t n = packed.shape(0);
    const int64_t row_bytes = packed.shape(1);
    if (row_bytes < (static_cast<int64_t>(dim) * bits + 7) / 8) throw std::invalid_argument("packed rows are too short");
    const uint8_t* p = packed.data();
    const float* qv = q.data();
    py::array_t<float> out(n);
    float* o = out.mutable_data();
    const int offset = 1 - (1 << bits);
    for (int64_t r = 0; r < n; r++) {
        const uint8_t* row = p + r * row_bytes;
        double acc = 0.0;
        for (int d = 0; d < dim; d++) {
            int c = 0;
            for (int j = 0; j < bits; j++) {
                int64_t bitpos = static_cast<int64_t>(d) * bits + j;
                c |= ((row[bitpos >> 3] >> (7 - (bitpos & 7))) & 1) << j;
            }
            acc += static_cast<double>(2 * c + offset) * static_cast<double>(qv[d]);
        }
        o[r] = static_cast<float>(acc);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "PaperLens C++ accelerator (see src/paperlens/native.py for the Python twins)";
    m.def("decode_postings", &decode_postings, py::arg("buf"), py::arg("pos"), py::arg("count"));
    m.def("bm25_accumulate", &bm25_accumulate, py::arg("postings"), py::arg("pos"), py::arg("count"),
          py::arg("doclens"), py::arg("dead"), py::arg("mask"), py::arg("idf"), py::arg("k1"), py::arg("b"),
          py::arg("avgdl"), py::arg("out"));
    m.def("level_dot", &level_dot, py::arg("packed"), py::arg("q"), py::arg("dim"), py::arg("bits"));
    m.attr("__version__") = "0.1.0";
}
