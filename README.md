# Dynamics Demos

Part of a presentation at PyData NYC 2018. [Slides here](https://tinyurl.com/pydata-hh).

### To run in the cloud

- https://mybinder.org/v2/gh/hsharrison/dynamics-demos/master?urlpath=/proxy/5006/hkb
- https://mybinder.org/v2/gh/hsharrison/dynamics-demos/master?urlpath=/proxy/5006/vdp
- https://mybinder.org/v2/gh/hsharrison/dynamics-demos/master?urlpath=/proxy/5006/sdm

### To run locally

    git clone https://github.com/hsharrison/dynamics-demos
    cd dynamics-demos

    # If necessary, apt-get install or equivalent gfortran
    # sudo apt-get install gfortran

    conda create --name=dynamics --file=environment.yml
    source activate dynamics

    bokeh serve --show apps/???.py
 