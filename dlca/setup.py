from setuptools import setup


setup(name='dlca',
      version='0.1',
      description='Tools for post processing of data acquired of DLC session',
      url='https://github.com/caniko2/DeepLabCutAnalysis',
      author='CINPLA',
      author_email='canhtart@gmail.com',
      license='LGPL-3.0',
      packages=['dlca'],
      zip_safe=False,
      install_requires=['matplotlib', 'numpy',
                        'scipy', 'opencv-python',
                        'pandas'])
