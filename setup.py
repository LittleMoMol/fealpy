import os
import pathlib
from setuptools import setup, find_packages, Extension

__version__ = "3.0.4"

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def load_requirements(path_dir=here, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [line.strip() for line in file.readlines()]
    requirements = []
    for line in lines:
        # filer all comments
        if comment_char in line:
            line = line[: line.index(comment_char)]
        if line:  # if requirement is not empty
            requirements.append(line)
    return requirements

ext_modules_dict={
        "mumps": Extension(
            'fealpy.solver.mumps._dmumps',
            sources=['fealpy/solver/mumps/_dmumps.pyx'],
            libraries=['dmumps', 'mumps_common'],
            ),
        "pangulu_cpu": Extension(
            'fealpy.solver.pangulu._pangulu_r64_cpu',
            sources=['fealpy/solver/pangulu/_pangulu_r64.pyx'],
            libraries=[f'pangulu_r64_cpu', 'metis', 'openblas'],
            library_dirs=['/usr/local/pangulu/lib'],
            include_dirs=['/usr/local/pangulu/include'],  # 指向生成的头文件
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
            ),
        "pangulu_gpu": Extension(
            'fealpy.solver.pangulu._pangulu_r64_gpu',
            sources=['fealpy/solver/pangulu/_pangulu_r64.pyx'],
            libraries=[f'pangulu_r64_gpu', 'metis', 'openblas', 'cudart', 'cusparse'],
            library_dirs=['/usr/local/pangulu/lib', '/usr/local/cuda/lib64'],
            include_dirs=['/usr/local/pangulu/include'], 
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
            )
        }

def get_ext_modules():
    ext_modules = []
    if os.getenv("WITH_MUMPS"):
        ext_modules.append(ext_modules_dict['mumps'])
    if os.getenv("WITH_PANGULU"):
        ext_modules.append(ext_modules_dict['pangulu_cpu'])
        ext_modules.append(ext_modules_dict['pangulu_gpu'])
    return ext_modules


ext_modules = get_ext_modules() 


setup(
    name="fealpy",
    version=__version__,
    description="FEALPy: Finite Element Analysis Library in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/weihuayi/fealpy",
    author="Huayi Wei",
    author_email="weihuayi@xtu.edu.cn",
    license="GNU",
    packages=find_packages(),
    install_requires=load_requirements(),
    zip_safe=False,
    extras_require={
        "doc": ["sphinx", "recommonmark", "sphinx-rtd-theme"],
        "dev": ["pytest", "pytest-cov", "bump2version"],
        "optional": ["pypardiso", "pyamg", "mpi4py", "meshpy"],
    },
    ext_modules=ext_modules,
    include_package_data=True,
    python_requires=">=3.10",
)

def build_pangulu():
    """
    """
    # install system dependencies
    os.system("sudo apt install -y libopenblas-dev libmetis-dev libopenmpi-dev")
    remote_url = 'git@github.com:SuperScientificSoftwareLaboratory/PanguLU.git'
    local_path = '/tmp/PanguLU'
    if not os.path.exists(local_path):
        print(f"Cloning repository to {local_path}...")
        os.system(f"git clone {remote_url} {local_path}")
    else:
        print(f"Repository already exists at {local_path}")
    os.chdir(local_path)
    generalFlags = """
        COMPILE_LEVEL = -O3
        CC = gcc $(COMPILE_LEVEL) #-fsanitize=address
        MPICC = mpicc $(COMPILE_LEVEL) #-fsanitize=address
        OPENBLAS_INC = -I/usr/include/x86_64-linux-gnu/openblas-openmp/
        OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu/openblas-openmp/ -lopenblas
        MPICCFLAGS = $(OPENBLAS_INC) $(CUDA_INC) $(OPENBLAS_LIB) -fopenmp -lpthread -lm
        MPICCLINK = $(OPENBLAS_LIB)
        METISFLAGS =  -I/usr/include
    """

    cudaFlags = """
        #0201000,GPU_CUDA
        CUDA_PATH = /usr/local/cuda
        CUDA_INC = -I/usr/local/cuda/include
        CUDA_LIB = -L/usr/local/cuda/lib64 -lcudart -lcusparse
        NVCC = nvcc $(COMPILE_LEVEL)
        NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 $(CUDA_INC) $(CUDA_LIB)
    """

    datatype = ['R64', 'R32']
    paltform = ['cpu', 'gpu']
    for dt in datatype:
        for pf in paltform:
            panguluFlags = f"-DPANGULU_LOG_INFO -DCALCULATE_TYPE_{dt} -DMETIS -DPANGULU_MC64 -DHT_IS_OPEN"
            os.system(f"make -f Makefile.{dt}_{pf}") 
            os.system("echo {compile_flags} > make.inc")
