project('chain_ufunc', 'c',
  version : '0.1',
)

cc = meson.get_compiler('c')
py = import('python').find_installation(pure: false)

c_sources = ['chain_ufunc/ufunc_chain.c']

py.extension_module('ufunc_chain',
  c_sources,
  install: true,
  subdir: 'chain_ufunc',
)

install_subdir('chain_ufunc',
  install_dir: py.get_install_dir(),
  exclude_files: c_sources
)
