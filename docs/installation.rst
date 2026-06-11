Installation
============

torchcor requires **Python 3.10+** and a CUDA-capable GPU (it also runs on CPU).

From source (editable install, recommended for development):

.. code-block:: bash

   git clone <repo-url> torchcor
   cd torchcor
   pip install -e .

This pulls in the runtime dependencies (PyTorch, NumPy, SciPy, PyVista, ...).

Verify the install:

.. code-block:: bash

   python -c "import torchcor as tc; print(tc.get_device())"

Select the device at the start of a script:

.. code-block:: python

   import torchcor as tc
   tc.set_device("cuda:0")   # or "cpu"
