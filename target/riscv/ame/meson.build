riscv_ss.add(files(
  'backend.c',
  'cap-set.c',
  'datatype.c',
  'layout.c',
  'tuple-set.c',
))

riscv_ss.add(when: 'CONFIG_TCG', if_true: files(
  'exec.c',
  'fp.c',
  'int.c',
))

riscv_system_ss.add(files(
  'cap-wire.c',
  'migration.c',
  'pool.c',
))
