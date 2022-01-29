# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as np


np.import_array()


from .quad_tree cimport QuadTree


ctypedef fused sparse_index_type:
    np.int32_t
    np.int64_t


cpdef np.ndarray compute_gaussian_perplexity(
    np.ndarray distances,
    np.ndarray desired_perplexities,
    float perplexity_tol=*,
    Py_ssize_t max_iter=*,
    Py_ssize_t num_threads=*,
)

cpdef tuple estimate_positive_gradient_nn(
    sparse_index_type[:] indices,
    sparse_index_type[:] indptr,
    float[:] P_data,
    float[:, ::1] embedding,
    float[:, ::1] reference_embedding,
    float[:, ::1] gradient,
    float dof=*,
    Py_ssize_t num_threads=*,
    bint should_eval_error=*,
)

cpdef float estimate_negative_gradient_bh(
    QuadTree tree,
    float[:, ::1] embedding,
    float[:, ::1] gradient,
    float theta=*,
    float dof=*,
    Py_ssize_t num_threads=*,
    bint pairwise_normalization=*,
)

cpdef float estimate_negative_gradient_fft_1d(
    float[::1] embedding,
    float[::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    float ints_in_interval=*,
    float dof=*,
)

cpdef tuple prepare_negative_gradient_fft_interpolation_grid_1d(
    float[::1] reference_embedding,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    float ints_in_interval=*,
    float dof=*,
    float padding=*,
)

cpdef float estimate_negative_gradient_fft_1d_with_grid(
    float[::1] embedding,
    float[::1] gradient,
    float[:, ::1] y_tilde_values,
    float[::1] box_lower_bounds,
    Py_ssize_t n_interpolation_points,
    float dof,
)

cpdef float estimate_negative_gradient_fft_2d(
    float[:, ::1] embedding,
    float[:, ::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    float ints_in_interval=*,
    float dof=*,
)

cpdef tuple prepare_negative_gradient_fft_interpolation_grid_2d(
    float[:, ::1] reference_embedding,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    float ints_in_interval=*,
    float dof=*,
    float padding=*,
)

cpdef float estimate_negative_gradient_fft_2d_with_grid(
    float[:, ::1] embedding,
    float[:, ::1] gradient,
    float[:, ::1] y_tilde_values,
    float[::1] box_x_lower_bounds,
    float[::1] box_y_lower_bounds,
    Py_ssize_t n_interpolation_points,
    float dof,
)
