from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

gencodes = [
                '-gencode', 'arch=compute_52,code=sm_52',
                '-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61',
                '-gencode', 'arch=compute_70,code=sm_70',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_75,code=compute_75',]
extra_compile_flags = {'cxx': [], 'nvcc': []}
extra_compile_flags['nvcc'] += gencodes


ext_modules=[
    CUDAExtension('soft_renderer.cuda.load_textures', [
        'soft_renderer/cuda/load_textures_cuda.cpp',
        'soft_renderer/cuda/load_textures_cuda_kernel.cu',
        ],extra_compile_args=extra_compile_flags,),
    CUDAExtension('soft_renderer.cuda.create_texture_image', [
        'soft_renderer/cuda/create_texture_image_cuda.cpp',
        'soft_renderer/cuda/create_texture_image_cuda_kernel.cu',
        ],extra_compile_args=extra_compile_flags,),
    CUDAExtension('soft_renderer.cuda.soft_rasterize', [
        'soft_renderer/cuda/soft_rasterize_cuda.cpp',
        'soft_renderer/cuda/soft_rasterize_cuda_kernel.cu',
        ],extra_compile_args=extra_compile_flags,),
    CUDAExtension('soft_renderer.cuda.voxelization', [
        'soft_renderer/cuda/voxelization_cuda.cpp',
        'soft_renderer/cuda/voxelization_cuda_kernel.cu',
        ],extra_compile_args=extra_compile_flags,),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "Soft Rasterizer"',
    author='Shichen Liu',
    author_email='liushichen95@gmail.com',
    license='MIT License',
    version='1.0.0',
    name='soft_renderer',
    packages=['soft_renderer', 'soft_renderer.cuda', 'soft_renderer.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
