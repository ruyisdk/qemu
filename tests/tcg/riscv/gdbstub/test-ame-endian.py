# RISC-V AME GDB register endianness tests
#
# SPDX-License-Identifier: GPL-2.0-or-later

import gdb

from test_gdbstub import main, report


def run_test():
    for name, value in (("amefflags", 0x1f), ("amexsat", 1),
                        ("amefflags", 0), ("amexsat", 0)):
        gdb.execute("set $%s = %d" % (name, value))
        gdb.execute("maintenance flush register-cache")
        report(int(gdb.parse_and_eval("$" + name)) == value,
               "%s reads back %d on a big-endian hart" % (name, value))


main(run_test)
