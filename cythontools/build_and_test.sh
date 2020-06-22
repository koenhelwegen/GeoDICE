echo
echo

CC=gcc python cythontools/setup.py build_ext --inplace

echo 
echo 

nosetests -v test/test_suite_lrgeodice.py
