import numpy as np
from ...backend import backend_manager as bm
from . import _pangulu_r64_cpu
#from . import _pangulu_r64_gpu

class PanguLUSolver:
    def __new__(cls, data_type='r', precision='64', platform='cpu', **kwargs):
        version = f"{data_type[0]}{precision}_{platform}"
        if version == 'r64_cpu':
            return _PanguLUR64CPUSolver(**kwargs)
        elif version == 'r64_gpu':
            return _PanguLUR64GPUSolver(**kwargs)
        else:
            raise ValueError(f"Unsupported version: {version}")


class _PanguLUR64CPUSolver:

    def __init__(self, nthread:int=4, nb:int=100):
        """
        Parameters:
            nb (int) : the size of the small matrix block, default is 100 
            nthread () : the number of threads per block, default is 4
        """
        self._init_options = _pangulu_r64_cpu.InitOptions(nthread=nthread, nb=nb)

    def solve(self, A, b):
        """Solve the linear system Ax = b on CPU
        """
        assert A.shape[0] == A.shape[1]
        assert len(A.data.shape) == 1
        csr_rowptr = bm.to_numpy(A.indptr).astype(np.int64)
        csr_colidx = bm.to_numpy(A.indices).astype(np.int32)
        csr_value = bm.to_numpy(A.data).astype(np.float64)

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

class _PanguLUR64GPUSolver(PanguLUSolver):
    def __init__(self, nb:int=100, nthread:int=4):
        """
        Parameters:
            nb (int) : the size of the small matrix block, default is 100 
            nthread () : the number of threads per block, default is 4
        """
        self._init_options = _pangulu_r64_gpu.InitOptions(nb, nthread)

    def solve(self, A, b):
        """Solve the linear system Ax = b on GPU
        """
        pass

