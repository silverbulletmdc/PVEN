from setuptools import setup, find_packages
import glob

setup(
    name="vehicle_reid_pytorch",
    version="1.0",
    # keywords=("pytorch", "vehicle", "ReID"),
    # description="Vechile ReID utils implemented with pytorch",
    # long_description="",
    packages=find_packages(exclude=('examples', 'examples.*')),
    scripts=glob.glob('scripts/*')
)