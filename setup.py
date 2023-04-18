from setuptools import setup

setup(name='dyna',
      version='0.1',
      description='Method for dynamic knowledge graph embeddings',
      author='Peter Bloem',
      author_email='dyna@peterbloem.nl',
      license='MIT',
      packages=['dyna'],
      install_requires=[
            'torch',
            'tqdm',
            'fire',
            'embed' # Install from https://github.com/pbloem/embed
      ],
      zip_safe=False)