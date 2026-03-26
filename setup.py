from setuptools import setup, find_packages

setup(
    name="franka_perception_thiago",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "open3d==0.19.0",
    ],
    extras_require={
        "sam": [
            "torch",
            "segment-anything",
            "transformers>=4.45.0",
            "pillow",
        ],
    },
)
