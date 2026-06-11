torchcor
========

**GPU-accelerated cardiac electrophysiology simulation in PyTorch.**

torchcor solves large-scale cardiac electrophysiology models on the GPU:

* **Monodomain** reaction-diffusion (the full reference model),
* the **Reaction-Eikonal** model (fast activation times + recovered voltage),
* body-surface **ECGs / electrograms** via a lead-field solver,

with a library of biophysical **ionic cell models** and a finite-element core
that runs on CUDA or CPU.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
