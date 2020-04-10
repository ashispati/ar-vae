#! /bin/sh

cd .. || exit
git clone https://github.com/deepmind/dsprites-dataset.git
cd dsprites-dataset ||exit
rm -rf .git* *.,d LICENSE *.ipynb *.gif *.hdf5
cd .. || exit
mv dsprites-dataset dsprites