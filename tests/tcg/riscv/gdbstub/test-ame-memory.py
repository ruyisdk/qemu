# SPDX-License-Identifier: GPL-2.0-or-later
"""AME load/store watchpoints must stop before any register or RAM commit."""

import gdb

from test_gdbstub import main, report


def address(symbol):
    return int(gdb.parse_and_eval("&" + symbol))


def packet(command):
    return gdb.execute("maintenance packet " + command, to_string=True)


def stop_before(symbol):
    stop = gdb.Breakpoint("*%d" % address(symbol))
    gdb.execute("continue")
    stop.delete()


def run_test():
    stop_before("memory_load")
    gdb.execute("set $m0.b[0] = 0xa5")
    gdb.execute("set $m0.b[15] = 0xa5")
    # A normal GDB watchpoint automatically steps the stopped instruction.
    # Raw packets let us inspect QEMU's pre-access state before that step.
    watch = address("memory_input") + 15
    report('"OK"' in packet(f"Z3,{watch:x},1"), "read watchpoint installed")
    reply = packet("c")
    gdb.execute("maintenance flush register-cache")
    report(f"rwatch:{watch:x};" in reply and
           int(gdb.parse_and_eval("$pc")) == address("memory_load"),
           "AME read watchpoint reports the load instruction")
    report(int(gdb.parse_and_eval("$m0.b[0]")) == 0xa5 and
           int(gdb.parse_and_eval("$m0.b[15]")) == 0xa5,
           "read watchpoint precedes the whole M register commit")
    report('"OK"' in packet(f"z3,{watch:x},1"), "read watchpoint removed")
    gdb.execute("stepi")
    report(all(int(gdb.parse_and_eval(f"$m0.b[{i}]")) == i
               for i in range(16)),
           "load retries and completes after watchpoint")

    stop_before("memory_store")
    watch = address("memory_output") + 15
    report('"OK"' in packet(f"Z2,{watch:x},1"), "write watchpoint installed")
    reply = packet("c")
    gdb.execute("maintenance flush register-cache")
    report(f"watch:{watch:x};" in reply and
           int(gdb.parse_and_eval("$pc")) == address("memory_store"),
           "AME write watchpoint reports the store instruction")
    data = bytes(gdb.selected_inferior().read_memory(
        address("memory_output"), 16))
    report(data == bytes([0xa5] * 16),
           "write watchpoint precedes every byte of the RAM commit")
    report('"OK"' in packet(f"z2,{watch:x},1"), "write watchpoint removed")
    gdb.execute("stepi")
    data = bytes(gdb.selected_inferior().read_memory(
        address("memory_output"), 16))
    report(data == bytes(range(16)),
           "store retries and completes after watchpoint")


main(run_test)
