from libc.stdlib cimport malloc, free

cdef extern from "pangulu_r64_cpu.h":

    ctypedef struct pangulu_init_options:
        int nthread
        int nb

    ctypedef struct pangulu_gstrf_options:
        pass

    ctypedef struct pangulu_gstrs_options:
        pass

    void pangulu_init(
        int n, long long nnz, long* csr_rowptr,
        int* csr_colidx, double* csr_value,
        pangulu_init_options* opts, void** handle
    )
    void pangulu_gstrf(
            pangulu_gstrf_options *gstrf_options, 
            void **handle);
    void pangulu_gstrs(double *rhs, 
                       pangulu_gstrs_options *gstrs_options, 
                       void **handle);
    void pangulu_gssv(double *rhs, 
                      pangulu_gstrf_options *gstrf_options, 
                      pangulu_gstrs_options *gstrs_options, void **handle);
    void pangulu_finalize(void **handle);

cdef class InitOptions:
    cdef pangulu_init_options _opts
    def __init__(self, int nthread=0, int nb=0):
        self._opts.nthread = nthread
        self._opts.nb = nb
    @property
    def nthread(self):
        return self._opts.nthread
    @nthread.setter
    def nthread(self, value):
        self._opts.nthread = value
    @property
    def nb(self):
        return self._opts.nb
    @nb.setter
    def nb(self, value):
        self._opts.nb = value

cdef class GstrfOptions:
    cdef pangulu_gstrf_options _opts  # 假设后续可能需要参数

cdef class GstrsOptions:
    cdef pangulu_gstrs_options _opts

cdef class Handle:
    cdef void* _handle
    def __dealloc__(self):
        if self._handle != NULL:
            pangulu_finalize(&self._handle)

def init(int n, long long nnz, long[::1] csr_rowptr, int[::1] csr_colidx, 
         double[::1] csr_value, InitOptions opts not None):
    cdef Handle handle_obj = Handle()
    pangulu_init(n, nnz, &csr_rowptr[0], &csr_colidx[0], &csr_value[0], 
                &opts._opts, &handle_obj._handle)
    return handle_obj

def gstrf(GstrfOptions opts, Handle handle not None):
    """
    performs distribute sparse LU factorisation. Note that you should
    call pangulu_init() before calling pangulu_gstrf() to create a handle of PanguLU.
    """
    pangulu_gstrf(&opts._opts if opts is not None else NULL, &handle._handle)

def gstrs(double[::1] rhs, GstrsOptions opts, Handle handle not None):
    """
    solves linear equation with factorised L and U, and right-hand
    side vector b. Note that you should call pangulu_gstrf() before calling
    pangulu_gstrs() to ensure that L and U are available.
    """
    pangulu_gstrs(&rhs[0], &opts._opts if opts is not None else NULL, 
                 &handle._handle)

def gssv(double[::1] rhs, GstrfOptions gstrf_opts, GstrsOptions gstrs_opts, 
         Handle handle not None):
    """
    pangulu_gssv() solves the linear equation with A and right-hand size b. This
    function is equivalent to calling pangulu_gstrs() after pangulu_gstrf().
    """
    pangulu_gssv(&rhs[0], 
                &gstrf_opts._opts if gstrf_opts is not None else NULL,
                &gstrs_opts._opts if gstrs_opts is not None else NULL,
                &handle._handle)
