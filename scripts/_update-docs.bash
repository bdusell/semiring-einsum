set -e
set -u
set -o pipefail

bash scripts/exec.bash bash scripts/build-docs.bash
git checkout gh-pages
rm -rf *.html _images/ *.inv *.js _sources/ _static/
cp -r docs/_build/html/. .
rm .buildinfo
echo 'You can now commit the updated files to this branch and push them to GitHub.'
