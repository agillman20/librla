brew install meson

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip --version

pip3 install numpy

---

Fails due to use of external routines

python3 -m numpy.f2py -m libid ../src/*.f
python3 -m numpy.f2py -c -m libid ../src/*.f

---


python3 -m numpy.f2py -c -m libid ../src/dfft.f
python3
(.venv) $ python3
Python 3.13.2 (main, Feb  4 2025, 14:51:09) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> import libid

>>> wsave = numpy.zeros(1000,'double')
>>> libid.dffti(12,wsave)

>>> x = numpy.random.rand(12)
>>> x
array([0.45189547, 0.63720631, 0.58370215, 0.11816658, 0.03516651,
       0.47889908, 0.79604129, 0.02401566, 0.21064929, 0.89429693,
       0.2444664 , 0.51293853])
>>> libid.dfftf(12,x,wsave)
>>> x
array([ 4.98744418,  0.50754827,  0.34474052,  0.52501087, -0.15943777,
       -0.92649857, -1.35528154,  0.8968783 ,  0.73208086, -0.61348716,
        0.62836899, -0.34360196])

>>> quit()

deactivate
