The package ``chain_ufunc`` allows one to create chains of ufuncs,
which are executed in order.  The idea is to do this at the C level, executing
the inner loops in order, so that one avoids allocating arrays for the
intermediate steps.  There is also a python version for comparison.

Example:

  >>> import numpy as np
  >>> from chain_ufunc import create_chained_ufunc
  >>> muladd = create_chained_ufunc([(np.multiply, [1, 2, 3]), (np.add, [1, 3, 3])], 3, 1, 0, "muladd")
  >>> muladd([0., 2., 1.], [4., 1., 6.], 0.1)
  array([4.4, 1.1, 6.6])
