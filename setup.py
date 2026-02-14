from setuptools import setup, find_packages

setup(
    name='mujoco-torch',
    version='0.1.0',
    packages=find_packages(),
    package_data={'mujoco_torch': ['test_data/**/*']},
    url='https://github.com/vmoens/mujoco-torch',
    license='Apache-2.0',
    author='vmoens',
    author_email='',
    description='MuJoCo MJX ported to PyTorch',
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.1',
        'mujoco>=3.0',
        'numpy',
        'tensordict>=0.11',
        'absl-py',
        'etils',
        'scipy',
        'trimesh',
    ],
    extras_require={
        'test': ['pytest'],
    },
)
