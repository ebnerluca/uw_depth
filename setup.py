from setuptools import setup

setup(
    name="depth_estimation_evaluation",
    version="0.1",
    description="The depth estimation evaluation package",
    url="",
    author="Luca Ebner",
    author_email="luca.ebner@sydney.edu.au",
    license="MIT",
    packages=["depth_estimation_evaluation"],  # , "depth_estimation"],
    package_dir={"depth_estimation_evaluation": "depth_estimation_evaluation"},
    zip_safe=True,
)
