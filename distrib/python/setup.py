from setuptools import setup

setup(
    name="librla",
    version="1.0.0",
    description="Randomized linear-algebra routines for low-rank matrix approximations",
    author="Adrianna Gillman, Zydrunas Gimbutas",
    license="NIST-PD",
    py_modules=["librla"],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
    ],
)
