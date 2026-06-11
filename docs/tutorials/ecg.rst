ECGs and electrograms (lead field)
==================================

:class:`~torchcor.ecg.leadfield.LeadField` turns a cardiac transmembrane voltage
``Vm`` into body-surface potentials using the **reciprocal lead-field method** on
a heart-torso finite-element mesh.  It solves the elliptic problem

.. math::

   -\nabla \cdot (G \nabla u) = \nabla \cdot (\sigma_i \nabla V_m)

once per electrode (a Right-Leg-grounded adjoint solve), so each subsequent ECG
is just a dense matrix-vector product ``Vm(t) . q_electrode`` — cheap for the
whole time trace.

Mesh requirement
----------------

The heart mesh must be the **same sub-domain of the torso mesh** (shared node
ordering), so that the ``Vm`` you computed on the heart maps onto the torso
nodes.  Extract the heart with ``torchcor.ecg.torso_heart.TorsoHeartMesh``;
``LeadField.build()`` verifies the node correspondence.

Step 1 -- get the source ``Vm``
-------------------------------

Run a heart EP simulation (monodomain for full fidelity, or reaction-eikonal for
a cheap source) and keep ``Vm`` on the heart mesh:

.. code-block:: python

   # monodomain (most accurate) -- see the Monodomain tutorial
   Vm = sim.solve(..., result_path="./out")          # (T, N_heart)
   # ... or the cheap reaction-eikonal source:
   #   Vm = reaction_eikonal_sim.solve(...)

Step 2 -- assemble the lead field
---------------------------------

.. code-block:: python

   import torch
   from torchcor.ecg import LeadField

   lf = LeadField(torso_mesh_dir="/path/to/torso",
                  heart_mesh_dir="/path/to/heart",
                  device=torch.device("cuda:0"), dtype=torch.float64)

   # torso conductivities (S/m, isotropic) for every non-heart region tag
   lf.add_torso_conductivity([10, 11, 12], g=0.6667)
   lf.add_torso_conductivity([3], g=0.05)
   # ... one call per torso tag ...

   # heart conductivities (S/m, anisotropic bidomain) -- same as the EP run
   lf.add_heart_conductivity([24, 25],     il=0.5272, it=0.2076, el=1.0732, et=0.4227)
   lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

   lf.build()

The solve runs in **float64** (lead-field accuracy degrades badly in float32).

Step 3 -- electrodes and lead-field precompute
----------------------------------------------

.. code-block:: python

   lf.load_electrodes("/path/to/electrodes/lf_src.vtx")   # V1..V6, RA, LA, RL, LL
   lf.precompute_all(a_tol=1e-8, r_tol=1e-8, max_iter=20000)

This is one CG solve per electrode (nine, with RL as the ground reference).

Step 4 -- compute and plot the 12-lead ECG
------------------------------------------

.. code-block:: python

   ecg = lf.compute_12lead(Vm.cpu())     # dict: I, II, III, aVR, aVL, aVF, V1..V6
   lf.plot_ecg(ecg, "ecg_12lead.png", dt=1.0)

``compute_12lead`` forms the Einthoven / Goldberger limb leads and the Wilson
central-terminal precordial leads, and prints an Einthoven consistency check
(``II == I + III``).

.. note::
   To sanity-check the whole pipeline independently of the reciprocity
   assumption, ``lf.validate_direct(Vm, frames=[...])`` solves the full forward
   problem at a few frames and compares it to the lead-field result — they must
   agree to solver tolerance.  If the ECG still looks wrong after that passes,
   the cause is the ``Vm`` source, not the lead field.
