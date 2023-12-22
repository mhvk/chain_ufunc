project('chain_ufunc', 'c',
  version : '0.1',
)

cc = meson.get_compiler('c')
py = import('python').find_installation(pure: false, modules: ['numpy'])

incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
print(os.path.relpath(np.get_include()))
  '''
    ],
    check: true
  ).stdout().strip()
message(incdir_numpy)
inc_np = include_directories(incdir_numpy)

c_sources = ['chain_ufunc/ufunc_chain.c']

py.extension_module('ufunc_chain',
  c_sources,
  install: true,
  include_directories: inc_np,
  subdir: 'chain_ufunc',
)

install_subdir('chain_ufunc',
  install_dir: py.get_install_dir(),
  exclude_files: c_sources
)
