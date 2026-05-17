.. _ch06:

###############################
Multi-Input Einsum Contraction
###############################

In diesem Assignment werden zwei intermediäre Tensoren einer
Light-Field-Tensor-Ring-Zerlegung kontrahiert. Zunächst dient PyTorchs
``torch.einsum`` als Referenz; anschließend wird ein cuTile-Kernel
gebaut, der das ``Config``/``Optimizer``-Interface aus Assignment 05
nutzt und gegen die Referenz verglichen wird.

.. toctree::
   :maxdepth: 1

   aufgabe
   loesung
