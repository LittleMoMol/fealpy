#!/usr/bin/python3
import ipdb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import TriangleMesh
from fealpy.fem import PoissonLFEMSolver
from fealpy.solver.mumps import spsolve_triangular
from fealpy.solver.pangulu import PanguLUSolver

from mpi4py import MPI # 提前导入




tmr = timer()
next(tmr)

p = 1
n = 3 
pde = CosCosData() 
domain = pde.domain()
mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
mesh.uniform_refine(n=7)
s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)

ds = PanguLUSolver(nb=200, nthread=12)
x = ds.solve(s0.A, s0.b)
residual = bm.max(bm.abs(s0.A@x - s0.b))
print(residual)


node = mesh.entity('node')
cell = mesh.entity('cell')
fig = plt.figure()
axes = fig.add_subplot(121)
mesh.add_plot(axes)
axes = fig.add_subplot(122, projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], x, triangles=cell, cmap='rainbow')
tmr.send('画图时间')
plt.show()
