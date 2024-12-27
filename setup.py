from setuptools import setup, find_packages

setup(
    name="virtual-power-plant",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pulp>=2.7.0",  # For LP/MILP optimization
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=21.5b2',
            'mypy>=0.900',
        ],
        'research': [
            'matplotlib>=3.4.0',  # For visualization
            'pandas>=1.3.0',      # For data analysis
            'scipy>=1.7.0',       # For advanced computations
            'torch>=1.9.0',       # For future RL implementation
        ],
    },
    author="Moudather Chelbi",
    author_email="moudather.chelbi@gmail.com",
    description="A comprehensive Python library for virtual power plant management, simulation, and research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vinerya/virtual-power-plant",
    project_urls={
        "Documentation": "https://github.com/vinerya/virtual-power-plant/docs",
        "Source": "https://github.com/vinerya/virtual-power-plant",
        "Issues": "https://github.com/vinerya/virtual-power-plant/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Energy",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    package_data={
        "vpp": ["py.typed"],
    },
    zip_safe=False,
)
