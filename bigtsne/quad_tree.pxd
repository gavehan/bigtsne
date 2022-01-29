# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
import numpy as np

cdef float EPSILON = np.finfo(float).eps

ctypedef struct Node:
    Py_ssize_t n_dims
    float *center
    float length

    bint is_leaf
    Node *children

    float *center_of_mass
    Py_ssize_t num_points


cdef bint is_duplicate(Node * node, float * point, float duplicate_eps=*) nogil


cdef class QuadTree:
    cdef Node root
    cpdef void add_points(self, float[:, ::1] points)
    cpdef void add_point(self, float[::1] point)
