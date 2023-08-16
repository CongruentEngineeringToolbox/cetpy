from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='cetpy',
    version='0.1.0b2',
    description='Congruent Engineering Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CongruentEngineeringToolbox/cetpy",
    author='CET developers',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="engineering, system engineering, congruent engineering",
    packages=find_packages(include=['cetpy', 'cetpy.*']),
    include_package_data=True,
    python_requires=">=3.10, <4",
    install_requires=[
        'numpy',
        'pandas',
        'CoolProp',
        'tomli',
        'smt',
    ],
)
