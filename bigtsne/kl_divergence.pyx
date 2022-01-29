# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as np
import numpy as np
from .quad_tree cimport QuadTree
from ._tsne cimport (
    estimate_negative_gradient_bh,
    estimate_negative_gradient_fft_1d,
    estimate_negative_gradient_fft_2d,
)
# This returns a tuple, and can"t be called from C
from ._tsne import estimate_positive_gradient_nn


cdef float EPSILON = np.finfo(float).eps

cdef extern from "math.h":
    float log(float x) nogil


cdef sqeuclidean(float[:] x, float[:] y):
    cdef:
        Py_ssize_t n_dims = x.shape[0]
        float result = 0
        Py_ssize_t i

    for i in range(n_dims):
        result += (x[i] - y[i]) ** 2

    return result


cpdef float kl_divergence_exact(float[:, ::1] P, float[:, ::1] embedding):
    """Compute the exact KL divergence."""
    cdef:
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t i, j

        float sum_P = 0, sum_Q = 0, p_ij, q_ij
        float kl_divergence = 0

    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                p_ij = P[i, j]
                q_ij = 1 / (1 + sqeuclidean(embedding[i], embedding[j]))
                sum_Q += q_ij
                sum_P += p_ij
                if p_ij > 0:
                    kl_divergence += p_ij * log(p_ij / (q_ij + EPSILON))

    kl_divergence += sum_P * log(sum_Q + EPSILON)

    return kl_divergence


cpdef float kl_divergence_approx_bh(
    int[:] indices,
    int[:] indptr,
    float[:] P_data,
    float[:, ::1] embedding,
    float theta=0.5,
    float dof=1,
):
    """Compute the KL divergence using the Barnes-Hut approximation."""
    cdef:
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t i, j

        QuadTree tree = QuadTree(embedding)
        # We don"t actually care about the gradient, so don"t waste time
        # initializing memory
        float[:, ::1] gradient = np.empty_like(embedding, dtype=float)

        float sum_P = 0, sum_Q = 0
        float kl_divergence = 0

    sum_Q = estimate_negative_gradient_bh(tree, embedding, gradient, theta, dof)
    sum_P, kl_divergence = estimate_positive_gradient_nn(
        indices,
        indptr,
        P_data,
        embedding,
        embedding,
        gradient,
        dof=dof,
        should_eval_error=True,
    )

    kl_divergence += sum_P * log(sum_Q + EPSILON)

    return kl_divergence



cpdef float kl_divergence_approx_fft(
    int[:] indices,
    int[:] indptr,
    float[:] P_data,
    float[:, ::1] embedding,
    float dof=1,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    float ints_in_interval=1,
):
    """Compute the KL divergence using the interpolation based approximation."""
    cdef:
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_dims = embedding.shape[1]
        Py_ssize_t i, j

        # We don"t actually care about the gradient, so don"t waste time
        # initializing memory
        float[:, ::1] gradient = np.empty_like(embedding, dtype=float)

        float sum_P = 0, sum_Q = 0
        float kl_divergence = 0


    if n_dims == 1:
        sum_Q = estimate_negative_gradient_fft_1d(
            embedding.ravel(),
            gradient.ravel(),
            n_interpolation_points,
            min_num_intervals,
            ints_in_interval,
            dof,
        )
    elif n_dims == 2:
        sum_Q = estimate_negative_gradient_fft_2d(
            embedding,
            gradient,
            n_interpolation_points,
            min_num_intervals,
            ints_in_interval,
            dof,
        )
    else:
        return -1

    sum_P, kl_divergence = estimate_positive_gradient_nn(
        indices,
        indptr,
        P_data,
        embedding,
        embedding,
        gradient,
        dof=dof,
        should_eval_error=True,
    )

    kl_divergence += sum_P * log(sum_Q + EPSILON)

    return kl_divergence
