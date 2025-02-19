
import numpy as np
import pytest
import scipy.sparse as sp

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve
from fealpy.sparse import COOTensor, CSRTensor

class TestPanguLUSolver:
    @pytest.mark.parametrize('backend', ['numpy'])
    def test_cpu(self, backend, solver_type):
        bm.set_backend(backend)
        solver = lambda A, b: spsolve(A, b, solver_type)
        A, x, b = self._get_cpu_data()
        x0 = solver(A, b) 
        assert self._check_solution(x0, x), "f{backend} Test failed!!!!!!!!!!!!!!!!!!!!!!!!"
