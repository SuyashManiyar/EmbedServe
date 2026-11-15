// C++ LRU hot tier for TierManager.
//
// Rows live in one contiguous float buffer of capacity * dim floats. An
// open-addressing hash map takes a (table, row id) key to a slot, and an
// intrusive doubly linked list over slots keeps LRU order. lookup() runs the
// whole per-ID pass of a batch in C++ for any mix of tables: hot hits are copied
// into the output, and each distinct missing key is recorded once with the
// output positions it must fill. Python fetches the misses from the cold tier
// and hands them back through put_batch().
//
// Semantics match the Python HotTier + TierManager exactly: a repeated key that
// missed earlier in the same batch counts as a hit, a hot hit moves the row to
// most recent, and misses are inserted in first-seen order after the fetch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kRowBits = 40;    // row ids < 2^40
constexpr int kTableBits = 23;  // table ids < 2^23, so a key never equals kEmpty
constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
constexpr std::int32_t kNone = -1;

std::uint64_t make_key(std::int64_t table, std::int64_t row) {
    if (table < 0 || table >= (std::int64_t{1} << kTableBits))
        throw std::out_of_range("table id out of range");
    if (row < 0 || row >= (std::int64_t{1} << kRowBits))
        throw std::out_of_range("row id out of range");
    return (static_cast<std::uint64_t>(table) << kRowBits) | static_cast<std::uint64_t>(row);
}

// Linear-probing map from key to a nonnegative int32, sized once for at most
// max_items entries (load factor <= 0.5). Erase uses backward-shift deletion.
class FlatMap {
public:
    explicit FlatMap(std::size_t max_items) {
        std::size_t n = 8;
        while (n < 2 * max_items + 2) n <<= 1;
        mask_ = n - 1;
        keys_.assign(n, kEmpty);
        vals_.resize(n);
    }

    std::int32_t find(std::uint64_t key) const {
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            if (keys_[i] == key) return vals_[i];
            if (keys_[i] == kEmpty) return kNone;
        }
    }

    void insert(std::uint64_t key, std::int32_t value) {  // key must be absent
        std::size_t i = home(key);
        while (keys_[i] != kEmpty) i = (i + 1) & mask_;
        keys_[i] = key;
        vals_[i] = value;
        ++size_;
    }

    void erase(std::uint64_t key) {  // key must be present
        std::size_t i = home(key);
        while (keys_[i] != key) i = (i + 1) & mask_;
        for (std::size_t j = i;;) {
            j = (j + 1) & mask_;
            if (keys_[j] == kEmpty) break;
            const std::size_t k = home(keys_[j]);
            // Leave entry j in place if its home slot k lies cyclically in (i, j].
            if (i <= j ? (i < k && k <= j) : (i < k || k <= j)) continue;
            keys_[i] = keys_[j];
            vals_[i] = vals_[j];
            i = j;
        }
        keys_[i] = kEmpty;
        --size_;
    }

    std::size_t size() const { return size_; }

private:
    std::size_t home(std::uint64_t x) const {  // splitmix64 finalizer
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x) & mask_;
    }

    std::size_t mask_ = 0, size_ = 0;
    std::vector<std::uint64_t> keys_;
    std::vector<std::int32_t> vals_;
};

using IdArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using RowArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using MaskArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

class HotTier {
public:
    HotTier(std::int64_t capacity, std::int64_t dim)
        : capacity_(check_capacity(capacity)), dim_(dim),
          index_(static_cast<std::size_t>(capacity)) {
        if (dim <= 0) throw std::invalid_argument("dim must be positive");
        rows_.resize(static_cast<std::size_t>(capacity * dim));
        keys_.resize(capacity);
        prev_.resize(capacity);
        next_.resize(capacity);
    }

    // Returns (out, hits, miss_tables, miss_ids, fill_pos, fill_slot). Rows of out
    // at hot hits are filled. Every other position appears in fill_pos, and the
    // caller must set out[fill_pos[k]] to the cold row of miss fill_slot[k]; those
    // positions are uninitialized until then.
    py::tuple lookup(IdArray tables_in, IdArray ids_in) {
        if (tables_in.ndim() != 1 || ids_in.ndim() != 1 || tables_in.shape(0) != ids_in.shape(0))
            throw std::invalid_argument("tables and ids must be one-dimensional and equal length");
        const auto n = ids_in.shape(0);
        if (n > INT32_MAX) throw std::invalid_argument("batch too large");
        const std::int64_t* tables = tables_in.data();
        const std::int64_t* ids = ids_in.data();
        RowArray out({n, static_cast<py::ssize_t>(dim_)});
        float* o = out.mutable_data();

        std::int64_t hits = 0;
        std::vector<std::int64_t> miss_tables, miss_ids, fill_pos, fill_slot;
        {
            py::gil_scoped_release release;
            FlatMap miss_slot(static_cast<std::size_t>(n));
            for (py::ssize_t i = 0; i < n; ++i) {
                const std::uint64_t key = make_key(tables[i], ids[i]);
                const std::int32_t s = index_.find(key);
                if (s != kNone) {
                    touch(s);
                    std::memcpy(o + i * dim_, slot_row(s), sizeof(float) * dim_);
                    ++hits;
                    continue;
                }
                std::int32_t m = miss_slot.size() ? miss_slot.find(key) : kNone;
                if (m != kNone) {
                    ++hits;  // repeat of a key that already missed in this batch
                } else {
                    m = static_cast<std::int32_t>(miss_ids.size());
                    miss_slot.insert(key, m);
                    miss_tables.push_back(tables[i]);
                    miss_ids.push_back(ids[i]);
                }
                fill_pos.push_back(i);
                fill_slot.push_back(m);
            }
        }
        return py::make_tuple(out, hits, to_array(miss_tables), to_array(miss_ids),
                              to_array(fill_pos), to_array(fill_slot));
    }

    // Insert rows in order, skipping entries whose found flag is false.
    void put_batch(IdArray tables_in, IdArray ids_in, RowArray rows_in, MaskArray found_in) {
        const auto m = ids_in.ndim() == 1 ? ids_in.shape(0) : -1;
        if (m < 0 || tables_in.ndim() != 1 || tables_in.shape(0) != m || rows_in.ndim() != 2 ||
            rows_in.shape(0) != m || rows_in.shape(1) != dim_ || found_in.ndim() != 1 ||
            found_in.shape(0) != m)
            throw std::invalid_argument(
                "put_batch expects tables (m,), ids (m,), rows (m, dim), found (m,)");
        if (capacity_ == 0) return;
        const std::int64_t* tables = tables_in.data();
        const std::int64_t* ids = ids_in.data();
        const float* rows = rows_in.data();
        const bool* found = found_in.data();
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < m; ++i)
            if (found[i]) put(make_key(tables[i], ids[i]), rows + i * dim_);
    }

    std::int64_t size() const { return static_cast<std::int64_t>(index_.size()); }
    std::int64_t capacity() const { return capacity_; }
    std::int64_t dim() const { return dim_; }
    std::int64_t nbytes() const { return size() * dim_ * static_cast<std::int64_t>(sizeof(float)); }

private:
    static std::int64_t check_capacity(std::int64_t capacity) {
        if (capacity < 0) throw std::invalid_argument("capacity must be nonnegative");
        if (capacity > INT32_MAX) throw std::invalid_argument("capacity too large");
        return capacity;
    }

    float* slot_row(std::int32_t s) { return rows_.data() + static_cast<std::size_t>(s) * dim_; }

    void unlink(std::int32_t s) {
        if (prev_[s] != kNone) next_[prev_[s]] = next_[s]; else head_ = next_[s];
        if (next_[s] != kNone) prev_[next_[s]] = prev_[s]; else tail_ = prev_[s];
    }

    void push_back(std::int32_t s) {  // tail = most recently used
        prev_[s] = tail_;
        next_[s] = kNone;
        if (tail_ != kNone) next_[tail_] = s; else head_ = s;
        tail_ = s;
    }

    void touch(std::int32_t s) {
        if (s == tail_) return;
        unlink(s);
        push_back(s);
    }

    void put(std::uint64_t key, const float* row) {
        std::int32_t s = index_.find(key);
        if (s != kNone) {
            touch(s);
        } else {
            if (used_ < capacity_) {
                s = static_cast<std::int32_t>(used_++);
            } else {  // evict least recently used, reuse its slot
                s = head_;
                index_.erase(keys_[s]);
                unlink(s);
            }
            index_.insert(key, s);
            keys_[s] = key;
            push_back(s);
        }
        std::memcpy(slot_row(s), row, sizeof(float) * dim_);
    }

    static py::array_t<std::int64_t> to_array(const std::vector<std::int64_t>& v) {
        py::array_t<std::int64_t> a(static_cast<py::ssize_t>(v.size()));
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), sizeof(std::int64_t) * v.size());
        return a;
    }

    std::int64_t capacity_, dim_, used_ = 0;
    std::vector<float> rows_;
    std::vector<std::uint64_t> keys_;
    std::vector<std::int32_t> prev_, next_;
    std::int32_t head_ = kNone, tail_ = kNone;
    FlatMap index_;
};

}  // namespace

PYBIND11_MODULE(_hot, m) {
    m.doc() = "C++ LRU hot tier with batched lookup";
    py::class_<HotTier>(m, "HotTier")
        .def(py::init<std::int64_t, std::int64_t>(), py::arg("capacity"), py::arg("dim"))
        .def("lookup", &HotTier::lookup, py::arg("tables"), py::arg("ids"))
        .def("put_batch", &HotTier::put_batch, py::arg("tables"), py::arg("ids"),
             py::arg("rows"), py::arg("found"))
        .def("__len__", &HotTier::size)
        .def_property_readonly("capacity", &HotTier::capacity)
        .def_property_readonly("dim", &HotTier::dim)
        .def_property_readonly("nbytes", &HotTier::nbytes);
}
