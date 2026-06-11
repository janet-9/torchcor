Ionic models
============

Biophysical cell models live in :mod:`torchcor.ionic`.  Each exposes the same
minimal interface the simulators use:

* ``initialize(n_nodes)`` -- allocate per-node state, return the resting Vm;
* ``differentiate(Vm)``   -- one ionic-current update step.

Pass an instance (or a list, one per region) to
:class:`~torchcor.simulator.monodomain.Monodomain` or ``ReactionEikonal``.

Available models
----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Description
   * - ``ModifiedMS2v``
     - Modified two-variable Mitchell-Schaeffer model (fast, phenomenological).
   * - ``MitchellSchaeffer``
     - Mitchell-Schaeffer phenomenological model.
   * - ``CourtemancheRamirezNattel``
     - Human **atrial** myocyte model.
   * - ``TenTusscherPanfilov``
     - Human **ventricular** myocyte model
       (``cell_type`` = ``ENDO`` / ``MID`` / ``EPI``).

Example
-------

.. code-block:: python

   from torchcor.ionic import ModifiedMS2v, TenTusscherPanfilov

   atrial      = ModifiedMS2v(dt=0.01)                       # phenomenological
   ventricular = TenTusscherPanfilov(cell_type="ENDO", dt=0.01)

.. note::
   The cell models are compiled with ``@torch.jit.script`` for speed, so they
   are not introspected by autodoc -- the table above is the reference.
