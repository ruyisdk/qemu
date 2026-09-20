riscv_ss.add(files(
  'cap-set.c',
  'datatype.c',
  'layout.c',
  'tuple-set.c',
))

riscv_ss.add(when: 'CONFIG_TCG', if_true: files(
  'fp.c',
  'int.c',
))

riscv_system_ss.add(files(
))
