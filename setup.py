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
<<<<<<< HEAD
          'mpmath',
          'opencv-python',
=======
          'mpmath'
>>>>>>> 8554db8802f30548a6f6c15194aafa068e6e2783
      ],
      zip_safe=False)