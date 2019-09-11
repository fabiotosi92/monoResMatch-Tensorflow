: "${TF_VERSION:?}"

pip install -q -U numpy
pip install -q "tensorflow==$TF_VERSION"
pip install matplotlib
pip install opencv-python
