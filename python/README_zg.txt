    
If you wish to install a Python library that isn't in Homebrew,
use a virtual environment:
    
python3 -m venv path/to/venv
source path/to/venv/bin/activate
python3 -m pip install xyz
    
If you wish to install a Python application that isn't in Homebrew,
it may be easiest to use 'pipx install xyz', which will manage a
virtual environment for you. You can install pipx with
    
brew install pipx
    
You may restore the old behavior of pip by passing
the '--break-system-packages' flag to pip, or by adding
'break-system-packages = true' to your pip.conf file. The latter
will permanently disable this error.
    
If you disable this error, we STRONGLY recommend that you additionally
pass the '--user' flag to pip, or set 'user = true' in your pip.conf
file. Failure to do this can result in a broken Homebrew installation.
    
Read more about this behavior here: <https://peps.python.org/pep-0668/>

-------------------------------------------------------------------------------

Installation:

python3 -m venv local
source local/bin/activate

###python3 -m pip install xyz
pip3 install jax

-------------------------------------------------------------------------------

activate/deactivate:

$ source local/bin/activate
(local) $ deactivate 
$ 

-------------------------------------------------------------------------------

machdep.cpu.brand_string: Apple M4 Max

$ source local/bin/activate
(local) $ python rrqr_rand1_dr.py 
a shape (4000, 2000)
Elapsed time: 0.0914 seconds
Elapsed time: 0.0074 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0076 seconds
Elapsed time: 0.0074 seconds
Elapsed time: 0.0072 seconds
r.shape =  (31, 2000)
k = 31
relerr =  2.271840641809376e-15
(local) $ python rrqr_rand2_dr.py 
a shape (4000, 2000)
Elapsed time: 0.0100 seconds
Elapsed time: 0.0079 seconds
Elapsed time: 0.0067 seconds
Elapsed time: 0.0077 seconds
Elapsed time: 0.0070 seconds
Elapsed time: 0.0073 seconds
Elapsed time: 0.0067 seconds
Elapsed time: 0.0065 seconds
Elapsed time: 0.0064 seconds
r.shape =  (31, 2000)
k = 31
relerr =  3.864282280129504e-15
(local) $ deactivate 

