import re
import setuptools

SECTION_RE = re.compile(r'^\[(.*?)\]$')
FIELD_RE = re.compile(r'^(.*?) = \[?"(.*?)"\]?$')
pyproject = {}
section = None
with open('pyproject.toml') as fin:
    for line in fin:
        line = line.rstrip('\n')
        m = SECTION_RE.match(line)
        if m is not None:
            section = m.group(1)
            pyproject[section] = {}
        else:
            m = FIELD_RE.match(line)
            if m is not None:
                key, value = m.groups()
                pyproject[section][key] = value

VERSION_RE = re.compile(r'^\^(.*)$')
def get_version(name):
    version_str = pyproject['tool.poetry.dependencies'][name]
    return VERSION_RE.match(version_str).group(1)

info = pyproject['tool.poetry']
with open('README.md') as fin:
    long_description = fin.read()

setuptools.setup(
    name=info['name'],
    version=info['version'],
    author=info['authors'],
    description=info['description'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=info['repository'],
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=' + get_version('python'),
    install_requires=[
        'torch>=' + get_version('torch')
    ]
)
