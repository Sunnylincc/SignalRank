#include <algorithm>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class ANNIndex {
 public:
  ANNIndex(const std::vector<std::vector<float>>& vectors, const std::vector<std::int64_t>& item_ids)
      : vectors_(vectors), item_ids_(item_ids) {
    if (vectors_.empty()) {
      throw std::invalid_argument("vectors cannot be empty");
    }
    if (vectors_.size() != item_ids_.size()) {
      throw std::invalid_argument("vectors and item_ids size mismatch");
    }
    dim_ = vectors_[0].size();
  }

  std::vector<std::pair<std::int64_t, float>> top_k(const std::vector<float>& query, int k) const {
    if (query.size() != dim_) {
      throw std::invalid_argument("query dimension mismatch");
    }
    if (k <= 0) return {};

    using Node = std::pair<float, std::int64_t>;
    auto cmp = [](const Node& a, const Node& b) { return a.first > b.first; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);

    for (std::size_t i = 0; i < vectors_.size(); ++i) {
      const auto score = dot(vectors_[i], query);
      if (static_cast<int>(heap.size()) < k) {
        heap.emplace(score, item_ids_[i]);
      } else if (score > heap.top().first) {
        heap.pop();
        heap.emplace(score, item_ids_[i]);
      }
    }

    std::vector<std::pair<std::int64_t, float>> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
      out.emplace_back(heap.top().second, heap.top().first);
      heap.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
  }

 private:
  static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
  }

  std::vector<std::vector<float>> vectors_;
  std::vector<std::int64_t> item_ids_;
  std::size_t dim_{};
};

PYBIND11_MODULE(signalrank_cpp, m) {
  m.doc() = "C++ accelerated top-k vector retrieval for SignalRank";
  py::class_<ANNIndex>(m, "ANNIndex")
      .def(py::init<const std::vector<std::vector<float>>&, const std::vector<std::int64_t>&>())
      .def("top_k", &ANNIndex::top_k, py::arg("query"), py::arg("k"));
}
