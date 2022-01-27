# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport bigtsne._matrix_mul.matrix_mul
cimport numpy as np
import numpy as np


cdef extern from 'fftw3.h':
    int fftwf_init_threads()
    void fftwf_plan_with_nthreads(int)

    cdef unsigned FFTW_ESTIMATE
    cdef unsigned FFTW_DESTROY_INPUT
    ctypedef float fftwf_complex[2]

    ctypedef struct _fftwf_plan:
       pass

    ctypedef _fftwf_plan *fftwf_plan

    void fftwf_execute(fftwf_plan)
    void fftwf_destroy_plan(fftwf_plan)
    fftwf_plan fftwf_plan_dft_r2c_1d(int, float*, fftwf_complex*, unsigned)
    fftwf_plan fftwf_plan_dft_c2r_1d(int, fftwf_complex*, float*, unsigned)
    fftwf_plan fftwf_plan_dft_r2c_2d(int, int, float*, fftwf_complex*, unsigned)
    fftwf_plan fftwf_plan_dft_c2r_2d(int, int, fftwf_complex*, float*, unsigned)


cdef void matrix_multiply_fft_1d(
    float[::1] kernel_tilde,
    float[:, ::1] w_coefficients,
    float[:, ::1] out,
):
    """Multiply the the kernel vectr K tilde with the w coefficients.
    
    Parameters
    ----------
    kernel_tilde : memoryview
        The generating vector of the 2d Toeplitz matrix i.e. the kernel 
        evaluated all all interpolation points from the left most 
        interpolation point, embedded in a circulant matrix (doubled in size 
        from (n_interp, n_interp) to (2 * n_interp, 2 * n_interp) and 
        symmetrized. See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    
    """
    cdef:
        Py_ssize_t n_interpolation_points_1d = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]

        complex[::1] fft_kernel_tilde = np.empty(n_fft_coeffs, dtype=np.csingle)
        complex[::1] fft_w_coeffs = np.empty(n_fft_coeffs, dtype=np.csingle)
        # Note that we can't use the same buffer for the input and output since
        # we only write to the first half of the vector - we'd need to
        # manually zero out the rest of the entries that were inevitably
        # changed during the IDFT, so it's faster to use two buffers, at the
        # cost of some memory
        float[::1] fft_in_buffer = np.zeros(n_fft_coeffs, dtype=np.single)
        float[::1] fft_out_buffer = np.zeros(n_fft_coeffs, dtype=np.single)

        Py_ssize_t d, i

    # Compute the FFT of the kernel vector
    cdef fftwf_plan plan_dft, plan_idft
    plan_dft = fftwf_plan_dft_r2c_1d(
        n_fft_coeffs,
        &kernel_tilde[0], <fftwf_complex *>(&fft_kernel_tilde[0]),
        FFTW_ESTIMATE,
    )
    fftwf_execute(plan_dft)
    fftwf_destroy_plan(plan_dft)

    plan_dft = fftwf_plan_dft_r2c_1d(
        n_fft_coeffs,
        &fft_in_buffer[0], <fftwf_complex *>(&fft_w_coeffs[0]),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )
    plan_idft = fftwf_plan_dft_c2r_1d(
        n_fft_coeffs,
        <fftwf_complex *>(&fft_w_coeffs[0]), &fft_out_buffer[0],
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )

    for d in range(n_terms):
        for i in range(n_interpolation_points_1d):
            fft_in_buffer[i] = w_coefficients[i, d]

        fftwf_execute(plan_dft)

        # Take the Hadamard product of two complex vectors
        for i in range(n_fft_coeffs):
            fft_w_coeffs[i] = fft_w_coeffs[i] * fft_kernel_tilde[i]

        fftwf_execute(plan_idft)

        for i in range(n_interpolation_points_1d):
            # FFTW doesn't perform IDFT normalization, so we have to do it
            # ourselves. This is done by multiplying the result with the number
            #  of points in the input
            out[i, d] = fft_out_buffer[n_interpolation_points_1d + i].real / n_fft_coeffs

    fftwf_destroy_plan(plan_dft)
    fftwf_destroy_plan(plan_idft)


cdef void matrix_multiply_fft_2d(
    float[:, ::1] kernel_tilde,
    float[:, ::1] w_coefficients,
    float[:, ::1] out,
):
    """Multiply the the kernel matrix K tilde with the w coefficients.
    
    Parameters
    ----------
    kernel_tilde : memoryview
        The generating matrix of the 3d Toeplitz tensor i.e. the kernel 
        evaluated all all interpolation points from the top left most 
        interpolation point, embedded in a circulant matrix (doubled in size 
        from (n_interp, n_interp) to (2 * n_interp, 2 * n_interp) and 
        symmetrized. See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    
    """
    cdef:
        Py_ssize_t total_interpolation_points = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t n_interpolation_points_1d = n_fft_coeffs / 2

        fftwf_plan plan_dft, plan_idft
        complex[::1] fft_w_coefficients = np.empty(n_fft_coeffs * (n_fft_coeffs / 2 + 1), dtype=np.csingle)
        complex[::1] fft_kernel_tilde = np.empty(n_fft_coeffs * (n_fft_coeffs / 2 + 1), dtype=np.csingle)
        # Note that we can't use the same buffer for the input and output since
        # we only write to the top quadrant of the in matrix - we'd need to
        # manually zero out the rest of the entries that were inevitably
        # changed during the IDFT, so it's faster to use two buffers, at the
        # cost of some memory
        float[:, ::1] fft_in_buffer = np.zeros((n_fft_coeffs, n_fft_coeffs))
        float[:, ::1] fft_out_buffer = np.zeros((n_fft_coeffs, n_fft_coeffs))

        Py_ssize_t d, i, j, idx

    plan_dft = fftwf_plan_dft_r2c_2d(
        n_fft_coeffs, n_fft_coeffs,
        &kernel_tilde[0, 0], <fftwf_complex *>(&fft_kernel_tilde[0]),
        FFTW_ESTIMATE,
    )
    fftwf_execute(plan_dft)
    fftwf_destroy_plan(plan_dft)

    plan_dft = fftwf_plan_dft_r2c_2d(
        n_fft_coeffs, n_fft_coeffs,
        &fft_in_buffer[0, 0], <fftwf_complex *>(&fft_w_coefficients[0]),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )
    plan_idft = fftwf_plan_dft_c2r_2d(
        n_fft_coeffs, n_fft_coeffs,
        <fftwf_complex *>(&fft_w_coefficients[0]), &fft_out_buffer[0, 0],
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )

    for d in range(n_terms):
        for i in range(n_interpolation_points_1d):
            for j in range(n_interpolation_points_1d):
                fft_in_buffer[i, j] = w_coefficients[i * n_interpolation_points_1d + j, d]

        fftwf_execute(plan_dft)

        # Take the Hadamard product of two complex vectors
        for i in range(n_fft_coeffs * (n_fft_coeffs / 2 + 1)):
            fft_w_coefficients[i] = fft_w_coefficients[i] * fft_kernel_tilde[i]

        # Invert the computed values at the interpolated nodes
        fftwf_execute(plan_idft)
        # FFTW doesn't perform IDFT normalization, so we have to do it
        # ourselves. This is done by multiplying the result with the number of
        # points in the input
        for i in range(n_interpolation_points_1d):
            for j in range(n_interpolation_points_1d):
                idx = i * n_interpolation_points_1d + j
                out[idx, d] = fft_out_buffer[n_interpolation_points_1d + i,
                                                        n_interpolation_points_1d + j] / n_fft_coeffs ** 2

    fftwf_destroy_plan(plan_dft)
    fftwf_destroy_plan(plan_idft)
