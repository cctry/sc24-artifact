#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <omp.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <unordered_set>
#include <vector>

at::Tensor cat_unique(std::vector<at::Tensor> &list) {
  int num_threads = std::min(omp_get_max_threads(), (int)(list.size()));
  std::vector<int64_t> size(num_threads, 0);
  std::vector<int64_t> offset(num_threads + 1, 0);
  std::vector<int64_t> range(num_threads + 1, 0);
  size_t part_size = (list.size() + num_threads - 1) / num_threads;
  for (int i = 0; i < num_threads; i++) {
    range[i + 1] = std::min((i + 1) * part_size, list.size());
    auto total_size = 0;
    for (auto j = range[i]; j < range[i + 1]; j++) {
      total_size += list[j].numel();
    }
    size[i] = total_size;
  }
  std::vector<int64_t> *global_res;
#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    std::vector<int64_t> local_res;
    local_res.reserve(size[tid]);
    for (auto i = range[tid]; i < range[tid + 1]; i++) {
      auto data = list[i].data_ptr<int64_t>();
      std::copy(data, data + list[i].numel(), std::back_inserter(local_res));
    }
    std::sort(local_res.begin(), local_res.end());
    auto last = std::unique(local_res.begin(), local_res.end());
    local_res.erase(last, local_res.end());
    size[tid] = local_res.size();
#pragma omp barrier
    if (tid == 0) {
      for (int i = 0; i < num_threads; i++) {
        offset[i + 1] = offset[i] + size[i];
      }
      global_res = new std::vector<int64_t>(offset.back(), 0);
    }
#pragma omp barrier
    std::copy(local_res.begin(), local_res.end(),
              global_res->begin() + offset[tid]);
  }
  std::sort(global_res->begin(), global_res->end());
  auto last = std::unique(global_res->begin(), global_res->end());
  global_res->erase(last, global_res->end());
  auto out =
      torch::empty({static_cast<int64_t>(global_res->size())}, torch::kInt64);
  std::copy(global_res->begin(), global_res->end(), out.data_ptr<int64_t>());
  delete global_res;
  return out;
}

at::Tensor cat_searchsorted(at::Tensor sorted_tensor,
                            std::vector<at::Tensor> &list) {
  std::vector<int64_t> offset(list.size() + 1, 0);
  for (int i = 0; i < list.size(); i++) {
    offset[i + 1] = offset[i] + list[i].numel();
  }
  auto ret_tensor = torch::empty({offset.back()}, torch::kInt64);
  auto ret_ptr = ret_tensor.data_ptr<int64_t>();
  auto v = sorted_tensor.data_ptr<int64_t>();
#pragma omp parallel for schedule(static, 64 / sizeof(int64_t))
  for (int i = 0; i < offset.back(); i++) {
    auto t = std::lower_bound(offset.begin(), offset.end(), i);
    int tensor_idx = distance(offset.begin(), t) - (i != *t);
    int ind = i - offset[tensor_idx];
    auto src_ptr = list[tensor_idx].data_ptr<int64_t>();
    auto target = src_ptr[ind];
    auto it = std::lower_bound(v, v + sorted_tensor.numel(), target);
    ret_ptr[i] = std::distance(v, it);
  }

  return ret_tensor;
}

at::Tensor vstack(py::list list, py::object type) {
  auto dtype = torch::python::detail::py_object_to_dtype(type);
  std::vector<py::buffer_info> buf_infos;
  for (int i = 0; i < list.size(); i++) {
    py::buffer buf = list[i].cast<py::buffer>();
    buf_infos.push_back(buf.request());
  }
  py::gil_scoped_release release;
  auto shape = buf_infos[0].shape;
  auto size = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());
  shape.insert(shape.begin(), list.size());
  auto out = torch::empty(shape, dtype);
  auto out_stride = out.strides()[0];
  AT_DISPATCH_ALL_TYPES(out.scalar_type(), "fast_vstack_np", [&] {
    auto out_ptr = out.data_ptr<scalar_t>();
    for (int i = 0; i < list.size(); i++) {
      auto src_ptr = static_cast<scalar_t *>(buf_infos[i].ptr);
      std::copy(src_ptr, src_ptr + size, out_ptr + i * out_stride);
    }
  });
  py::gil_scoped_acquire acquire;
  return out;
}

at::Tensor sorted_count(at::Tensor content, at::Tensor target,
                        int64_t target_space) {
  const auto *const content_ptr = content.data_ptr<int64_t>();
  const auto *const target_ptr = target.data_ptr<int64_t>();
  const auto content_size = content.numel();
  const auto target_size = target.numel();
  const auto out_size = target_space > 0 ? target_space : target_size;
  const auto out = torch::zeros({out_size}, torch::kInt64);
  auto out_ptr = out.data_ptr<int64_t>();

  int num_thd = std::min(omp_get_max_threads(), (int)(content_size));

  auto tgt_start_ptr = target_ptr;
  auto tgt_end_ptr = target_ptr + target_size;
  auto content_part_size = (content_size + num_thd - 1) / num_thd;
  auto tgt_part_size = (target_size + num_thd - 1) / num_thd;
  std::vector<int64_t> global_out(target_size, 0);

#pragma omp parallel num_threads(num_thd)
  {
    int tid = omp_get_thread_num();
    auto content_start = tid * content_part_size;
    auto content_end = std::min(content_start + content_part_size,
                                static_cast<int64_t>(content_size));
    std::vector<int64_t> local_out(target_size, 0);
    for (int i = content_start; i < content_end; i++) {
      auto it = std::lower_bound(tgt_start_ptr, tgt_end_ptr, content_ptr[i]);
      if (it != tgt_end_ptr && *it == content_ptr[i]) {
        local_out[std::distance(tgt_start_ptr, it)]++;
      }
    }
    // sum up results to global out
    for (auto i = 0; i < num_thd; ++i) {
      auto part_idx = (tid + i) % num_thd;
      auto start = part_idx * tgt_part_size;
      auto end = std::min(start + tgt_part_size, target_size);
      for (auto j = start; j < end; ++j) {
        global_out[j] += local_out[j];
      }
#pragma omp barrier
    }
    auto tgt_start = tid * tgt_part_size;
    auto tgt_end = std::min(tgt_start + tgt_part_size, target_size);
    for (auto i = tgt_start; i < tgt_end; ++i) {
      if (target_space > 0) {
        out_ptr[target_ptr[i]] = global_out[i];
      } else {
        out_ptr[i] = global_out[i];
      }
    }
  }
  return out;
}

template <typename OP_t>
void indexed_acc(torch::Tensor &out, torch::Tensor &out_idx, torch::Tensor &in,
                 torch::Tensor &in_idx) {
  auto out_data = out.data_ptr<float>();
  auto in_data = in.data_ptr<float>();
  auto out_idx_data = out_idx.data_ptr<int64_t>();
  auto in_idx_data = in_idx.data_ptr<int64_t>();

  // Get the size of the index tensors (assuming both have the same size)
  auto idx_size = out_idx.size(0);
  int dim = out.size(1);

  OP_t op;

#pragma omp parallel for
  for (int64_t i = 0; i < idx_size; ++i) {
    float *src_ptr = in_data + (in_idx_data[i] * dim);
    float *tgt_ptr = out_data + (out_idx_data[i] * dim);
#pragma omp simd
    for (int64_t j = 0; j < dim; ++j) {
      tgt_ptr[j] = op(tgt_ptr[j], src_ptr[j]);
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cat_unique", &cat_unique, "Concatenate and get unique values (CPU)",
        py::call_guard<py::gil_scoped_release>());
  m.def("cat_searchsorted", &cat_searchsorted,
        "Concatenate and searchsorted (CPU)",
        py::call_guard<py::gil_scoped_release>());
  m.def("vstack", &vstack, "vstack");
  m.def("sorted_count", &sorted_count, "sorted_count",
        py::call_guard<py::gil_scoped_release>());
  m.def("indexed_add", &indexed_acc<std::plus<float>>, "indexed_add",
        py::call_guard<py::gil_scoped_release>());
}
