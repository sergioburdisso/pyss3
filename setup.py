"""Setup script."""
import re
from os import path
from codecs import open
from setuptools import setup, find_packages

__encode__ = 'utf8'

DISTNAME = 'pyss3'
DESCRIPTION = ("Python package that implements the SS3 text classifier (with "
               "visualizations tools for XAI)")
AUTHOR = 'Sergio Burdisso'
AUTHOR_EMAIL = 'sergio.burdisso@gmail.com, sburdisso@unsl.edu.ar'
URL = "https://github.com/sergioburdisso/pyss3"
LICENSE = "MIT License"

CLASSIFIERS = ['Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Visualization',
               'Operating System :: OS Independent']

with open('README.md', encoding=__encode__) as readme:
    LONG_DESCRIPTION = readme.read()

with open('requirements.txt') as requirements:
    INSTALL_REQS = requirements.read().splitlines()


_version_re__ = r"__version__\s*=\s*['\"]([^'\"]+)['\"]"
__cwd__ = path.abspath(path.dirname(__file__))
__init_file__ = path.join(__cwd__, '%s/__init__.py' % DISTNAME)
with open(__init_file__, encoding=__encode__) as __init__py:
    VERSION = re.search(_version_re__, __init__py.read()).group(1)

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          maintainer=AUTHOR,
          maintainer_email=AUTHOR_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          long_description_content_type='text/markdown',
          packages=find_packages(),
          package_data={DISTNAME: ['resources/**/*', 'resources/**/**/*']},
          include_package_data=True,
          classifiers=CLASSIFIERS,
          python_requires='>=2.7',
          install_requires=INSTALL_REQS,
          tests_require=['pytest'] + INSTALL_REQS,
          entry_points={'console_scripts': ['pyss3=pyss3.cmd:main']})
