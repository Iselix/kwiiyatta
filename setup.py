import setuptools

setuptools.setup(
    name="kwiiyatta",
    version="0.0.1",
    author="Iselix",
    description="Voice conversion tool",
    packages=setuptools.find_packages(),
    install_requires=[
        'nnmnkwii',
        'numpy',
        'scipy',
        'sklearn',
        'pyworld',
        'pysptk',
    ],
)
