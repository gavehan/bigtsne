# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3

cdef void matrix_multiply_fft_1d(
    float[::1] kernel_tilde,
    float[:, ::1] w_coefficients,
    float[:, ::1] out,
)

cdef void matrix_multiply_fft_2d(
    float[:, ::1] kernel_tilde,
    float[:, ::1] w_coefficients,
    float[:, ::1] out,
)
