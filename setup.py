from setuptools import setup, find_packages

setup(
    name='Deal',
    version='0.1.0',  # Update the version number for new releases
    # author='Your Name',
    # author_email='your.email@example.com',
    # description='A short description of the project',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',  # This is important for a markdown README
    # url='https://github.com/yourusername/deal',  # Optional project URL
    packages=find_packages(),  # Automatically discover all packages and subpackages
    install_requires=[
        'numpy', 
        'pandas',
        'pyarrow',
        'torch',
        'dgl',
        'ogb'
    ],
    # classifiers=[
    #     'Development Status :: 3 - Alpha',  # Change as appropriate
    #     'Intended Audience :: Developers',
    #     'License :: OSI Approved :: MIT License',  # Change the license as needed
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.7',
    #     'Programming Language :: Python :: 3.8',
    #     'Programming Language :: Python :: 3.9',
    # ],
    python_requires='>=3.9', 
)
