from setuptools import setup

setup(
    name="VisInspec",
    version="0.1.0",
    url="https://github.com/brain-facens/VisInspection",
    author="Vitor Ozaki",
    author_email="vitor.ozaki@facens.br",
    license="Apache License",
    packages=["VisInspec"],
    install_requires=[  "matplotlib>=3.6.3",
                        "numpy>=1.23.5",
                        "opencv-python>=4.7.0.66",
                        "pandas>=1.5.2",
                        "scikit-image>=0.19.3"]
)