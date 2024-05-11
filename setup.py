from setuptools import setup

setup(
    name='mujoco-torch',
    version='0.1.0',
    packages=[
        'mujoco_torch',
        'mujoco_torch._src',
        # 'mujoco_torch.benchmark'
    ],
    url='',
    license='',
    author='vmoens',
    author_email='',
    description='',
    dependencies=["trimesh"],
)
