from setuptools import setup

setup(name='almiky',
      version='0.1',
      description='Orthogonal moments library',
      url='https://gitlab.udg.co.cu/Investigacion/almiky',
      author='Yenner Diaz, Anier Soria',
      author_email='yennerdiaz@gmail.com',
      license='LGPL3',
      packages=['almiky'],
      install_requires=[
          'numpy',
          'scipy', 
          'mpmath'
      ],
      zip_safe=False)