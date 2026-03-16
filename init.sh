python -m pip install -r requirements.txt
python setup_glip.py build_ext --inplace
python -m pip install -e . --no-build-isolation
