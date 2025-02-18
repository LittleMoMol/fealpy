import numpy as np
from . import _pangulu_r64_cpu


solver_classes = {
        'r64_cpu': _pangulu_r64_cpu.r64_cpu_solver,
        }

class PanguLUSolver:
    def __new__(cls, data_type='r', precision='64', platform='cpu', **kwargs):
        version = f"{data_type[0]}{precision}_{platform}"
        return solver_classes[version](**kwargs)


class _PanguLUR64CPUSolver(PanguLUSolver):

    def __init__(self, nb, nthread):
        self._init_options = _pangulu_r64_cpu.InitOptions(nb, nthread)

    def solve(self, A, b):
        """
        """
        assert A.shape[0] == A.shape[1]
        assert len(A.data.shape) == 1
        csr_rowptr = np.astype(A.indptr, dtype=np.int64)
        csr_colidx = np.astype(A.indices, dtype=np.int32)
        csr_value = A.data

        x = b.copy()
        # Initialize the handle
        self._handle = _pangulu_r64_cpu.init(
                A.shape[0], 
                A.nnz,
                csr_rowptr,
                csr_colidx,
                csr_value,
                self._init_options)
        # factorize and solve
        _pangulu_r64_cpu.gssv(
                x, 
                _pangulu_r64_cpu.GstrfOptions(), 
                _pangulu_r64_cpu.GstrsOptions(), self._handle)
        return x


