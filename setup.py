import re
import setuptools

pyproject = {}
FIELD_RE = re.compile(r'^(.*?) = \[?"(.*?)"\]?\n')
with open('pyproject.toml') as fin:
    for line in fin:
        if line == '[tool.poetry]\n':
            continue
        elif line == '\n':
            break
        key, value = FIELD_RE.match(line).groups()
        pyproject[key] = value

with open('README.md') as fin:
    long_description = fin.read()

setuptools.setup(
    name=pyproject['name'],
    version=pyproject['version'],
    author=pyproject['authors'],
    description=pyproject['description'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=pyproject['repository'],
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
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.0.0'
    ]
)
