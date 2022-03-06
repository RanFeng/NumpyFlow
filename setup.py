from setuptools import setup, find_packages

setup(
    name='nf',
    version='0.0.2',
    keywords='DL',
    description='a toy deep learning library write by pure numpy named NumpyFlow',
    license='',
    url='https://github.com/RanFeng/NumpyFlow',
    author='Xun Ai',
    author_email='kidformyself@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=["numpy"],
)