from setuptools import setup, find_packages

setup(
    name='modelforge',
    version='0.1.0',
    author='Michael Malin',
    author_email='mmalin@modelforge.ai',
    description='A machine learning package for EENN and GraphWeld models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://modelforge.ai',
    license='Apache 2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 'pandas', 'scikit-learn'
    ],
    extras_require={
        'tensorflow': ['tensorflow'],
        'pytorch': ['torch'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Apache 2.0',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
)