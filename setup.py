from setuptools import setup

setup(
    name='almiky',
    version='0.1',
    description='Python library for data hiding in images',
      url='https://gitlab.udg.co.cu/Investigacion/almiky',
      author='Yenner J. Diaz-Nuñez, Anier Soria-Lorente, Ernesto Avila-Domenech',
      author_email='yennerdiaz@gmail.com',
      license='LGPL3',
      packages=['almiky'],
      install_requires=[
          'numpy==1.19.4',
          'scipy==1.5.4', 
          'mpmath==1.1.0',
          'opencv-python==4.4.0.46'
      ],
      zip_safe=False)