"""
Setup script for EARCP library.

Copyright (c) 2025 Mike Amega. All rights reserved.
"""

from setuptools import setup, find_packages
import os


# Read README for long description
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


# Read version from __init__.py
def get_version():
    init_path = os.path.join('earcp', '__init__.py')
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '1.0.0'


setup(
    name='earcp',
    version=get_version(),
    author='Mike Amega',
    author_email='info@amewebstudio.com',
    description='EARCP: Ensemble Auto-Régulé par Cohérence et Performance',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/Volgat/earcp',
    project_urls={
        'Documentation': 'https://github.com/Volgat/earcp/blob/main/docs/USAGE.md',
        'Source': 'https://github.com/Volgat/earcp',
        'Bug Reports': 'https://github.com/Volgat/earcp/issues',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords=[
        'machine-learning',
        'ensemble-learning',
        'online-learning',
        'adaptive-algorithms',
        'model-combination',
        'expert-weighting',
        'coherence',
        'performance',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'torch': [
            'torch>=1.9.0',
        ],
        'sklearn': [
            'scikit-learn>=0.24.0',
        ],
        'full': [
            'torch>=1.9.0',
            'scikit-learn>=0.24.0',
            'pandas>=1.3.0',
            'seaborn>=0.11.0',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license='MIT',
)
