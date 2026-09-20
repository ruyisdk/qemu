# RISC-V AME GDB register tests
#
# SPDX-License-Identifier: GPL-2.0-or-later

import gdb
import xml.etree.ElementTree as ET

from test_gdbstub import main, report

GDB_CHUNK_BYTES = 64 * 1024


def reg_u32(name):
    return int(gdb.parse_and_eval("$" + name))


def reg_byte(name, index):
    return int(gdb.parse_and_eval("$%s.b[%d]" % (name, index)))


def set_reg_byte(name, index, value):
    gdb.execute("set $%s.b[%d] = %d" % (name, index, value))


def xml_registers():
    tree = ET.fromstring(gdb.execute("maint print xml-tdesc", False, True))
    all_names = set()
    ame_regs = {}
    duplicate = False

    for feature in tree.findall("feature"):
        for reg in feature.findall("reg"):
            name = reg.attrib["name"]
            duplicate |= name in all_names
            all_names.add(name)
            if feature.attrib["name"] == "org.gnu.gdb.riscv.ame":
                ame_regs[name] = reg
    report(not duplicate, "target description has no duplicate registers")
    return ame_regs


def run_test():
    n = reg_u32("amenlen")
    u = reg_u32("ameudsz")
    m_bytes = n ** 2 * u // 8
    # The current capability maximum is Int128.
    acc_bytes = n ** 2 * 128 // 8

    report(reg_u32("ame_max_int_dtype") == 0x40000080,
           "AME_MAX_INT_DTYPE is exposed")
    m_chunks = (m_bytes + GDB_CHUNK_BYTES - 1) // GDB_CHUNK_BYTES
    acc_chunks = (acc_bytes + GDB_CHUNK_BYTES - 1) // GDB_CHUNK_BYTES
    m_first = "m0" if m_chunks == 1 else "m0_0"
    acc_first = "acc0" if acc_chunks == 1 else "acc0_0"
    mreg16 = "m16" if m_chunks == 1 else "m16_0"
    acc1 = "acc1" if acc_chunks == 1 else "acc1_0"
    is_max = n == 128 and u == 64
    num_mregs = 32 if is_max else 16
    num_accregs = 16 if is_max else 1

    ame_regs = xml_registers()
    expected_regs = num_mregs * (m_chunks + 1) + \
                    num_accregs * (acc_chunks + 1) + 8
    report(len(ame_regs) == expected_regs,
           "AME XML register count follows N, U, NM, and NA")
    report(int(ame_regs[m_first].attrib["bitsize"]) ==
           min(m_bytes, GDB_CHUNK_BYTES) * 8 and
           int(ame_regs[acc_first].attrib["bitsize"]) ==
           min(acc_bytes, GDB_CHUNK_BYTES) * 8,
           "AME XML backing register bitsizes follow the configuration")

    report((mreg16 in ame_regs) == is_max and
           (acc1 in ame_regs) == is_max,
           "GDB feature follows configured NM and NA")
    last_mreg = 31 if is_max else 15
    last_acc = 15 if is_max else 0
    m_last = "m%d" % last_mreg if m_chunks == 1 else \
             "m%d_%d" % (last_mreg, m_chunks - 1)
    acc_last = "acc%d" % last_acc if acc_chunks == 1 else \
               "acc%d_%d" % (last_acc, acc_chunks - 1)
    m_last_index = m_bytes - 1 if m_chunks == 1 else GDB_CHUNK_BYTES - 1
    acc_last_index = (acc_bytes - 1 if acc_chunks == 1 else
                      GDB_CHUNK_BYTES - 1)

    report(reg_byte(m_first, 0) == 0 and
           reg_byte(m_last, m_last_index) == 0,
           "M register endpoints reset to zero")
    report(reg_byte(acc_first, 0) == 0 and
           reg_byte(acc_last, acc_last_index) == 0,
           "Acc register endpoints reset to zero")

    report(reg_u32("ameown") == 0,
           "a hart without a grant reports no ownership")
    set_reg_byte(m_first, 0, 0x5a)
    set_reg_byte(acc_first, 0, 0x3c)
    gdb.execute("set $md0 = 0x20300120")
    gdb.execute("set $amestype = 0x20300120")
    report(reg_byte(m_first, 0) == 0 and
           reg_byte(acc_first, 0) == 0 and reg_u32("md0") == 0 and
           reg_u32("amestype") == 0,
           "a non-owner reads zero and cannot modify backend state")

    owner_thread = gdb.selected_thread()
    acquire_break = gdb.Breakpoint("gdb_ame_after_acquire")
    gdb.execute("set $pc = gdb_ame_acquire")
    gdb.execute("continue")
    acquire_break.delete()
    report(reg_u32("ameown") == 1, "ownership is visible while stopped")

    set_reg_byte(m_first, 0, 0x5a)
    set_reg_byte(m_last, m_last_index, 0xa5)
    report(reg_byte(m_first, 0) == 0x5a and
           reg_byte(m_last, m_last_index) == 0xa5,
           "M register full-packet read/write round trip")

    set_reg_byte(acc_first, 0, 0x3c)
    set_reg_byte(acc_last, acc_last_index, 0xc3)
    report(reg_byte(acc_first, 0) == 0x3c and
           reg_byte(acc_last, acc_last_index) == 0xc3,
           "Acc register full-packet read/write round trip")

    gdb.execute("set $md0 = 0x20300120")
    report(reg_u32("md0") == 0x20300120,
           "valid raw Md descriptor is accepted")
    gdb.execute("set $md0 = 0xffffffff")
    report(reg_u32("md0") == 0x20300120,
           "invalid raw Md descriptor is rejected atomically")

    gdb.execute("set $md1 = 0x40000008")
    wide_dtype = {
        8: 0x40000010,
        16: 0x40000020,
        32: 0x40000040,
        64: 0x40000080,
    }[u]
    gdb.execute("set $md1 = 0x%08x" % wide_dtype)
    report(reg_u32("md1") == 0x40000008,
           "misaligned wide Md descriptor is rejected atomically")

    gdb.execute("set $ad0 = 0x20300120")
    report(reg_u32("ad0") == 0x20300120,
           "valid raw Ad descriptor is accepted")
    gdb.execute("set $amestype = 0x20300120")
    report(reg_u32("amestype") == 0x20300120,
           "scalar descriptor cache input is writable")
    gdb.execute("set $amestype = 0xffffffff")
    report(reg_u32("amestype") == 0xffffffff,
           "unsupported scalar descriptor raw bits are retained")

    gdb.execute("set $md0 = 0x20300120")
    set_reg_byte(m_first, 0, 0xff)
    execute_break = gdb.Breakpoint("gdb_ame_after_execute")
    gdb.execute("set $pc = gdb_ame_execute")
    gdb.execute("continue")
    execute_break.delete()
    report(reg_byte(m_first, 0) == 0,
           "Md write rebuilds the dtype cache used by execution")

    gdb.execute("set $amefflags = 0xffffffff")
    gdb.execute("set $amexsat = 1")
    report(reg_u32("amefflags") == 0x1f and reg_u32("amexsat") == 1,
           "writable AME CSR masks are enforced")

    other_threads = [thread for thread in gdb.selected_inferior().threads()
                     if thread != owner_thread]
    report(len(other_threads) == 1,
           "the shared-backend fixture exposes two harts")
    if other_threads:
        other_threads[0].switch()
        gdb.execute("maintenance flush register-cache")
        report(reg_u32("ameown") == 0 and
               reg_byte(m_first, 0) == 0 and
               reg_byte(acc_first, 0) == 0 and reg_u32("md0") == 0 and
               reg_u32("amestype") == 0,
               "a non-owner GDB thread cannot observe the owner slot")
        set_reg_byte(m_first, 0, 0x66)
        set_reg_byte(acc_first, 0, 0x77)
        gdb.execute("set $md0 = 0x40000008")
        gdb.execute("set $amestype = 0x40000008")
        report(reg_byte(m_first, 0) == 0 and
               reg_byte(acc_first, 0) == 0 and reg_u32("md0") == 0 and
               reg_u32("amestype") == 0,
               "a non-owner GDB thread cannot modify the owner slot")
        gdb.execute("set $amefflags = 3")
        gdb.execute("set $amexsat = 1")
        report(reg_u32("amefflags") == 3 and reg_u32("amexsat") == 1,
               "GDB can read and write a non-owner hart's local flags")

        owner_thread.switch()
        gdb.execute("maintenance flush register-cache")
        report(reg_u32("ameown") == 1 and
               reg_byte(acc_first, 0) == 0x3c and
               reg_u32("md0") == 0x20300120 and
               reg_u32("amestype") == 0xffffffff and
               reg_u32("amefflags") == 0x1f and reg_u32("amexsat") == 1,
               "non-owner access does not corrupt owner or hart-local state")

        # A debugger stop is a management event, not an architectural
        # interrupt.  It must leave a pending bounded acquire intact.
        gdb.execute("set {int}&gdb_ame_wait_ready = 0")
        gdb.execute("set {int}&gdb_ame_wait_done = 0")
        gdb.execute("set {int}&gdb_ame_wait_result = 0")
        gdb.execute("set $pc = gdb_ame_owner_hold")
        other_threads[0].switch()
        gdb.execute("set $pc = gdb_ame_waiter")
        wait_stop = gdb.Breakpoint("gdb_ame_owner_wait_stopped")
        gdb.execute("continue")
        wait_stop.delete()

        other_threads[0].switch()
        gdb.execute("maintenance flush register-cache")
        wait_pc = int(gdb.parse_and_eval("&gdb_ame_wait_acquire"))
        report(int(gdb.parse_and_eval("$pc")) == wait_pc and
               reg_u32("a0") == 0x55 and reg_u32("ameown") == 0,
               "GDB stop preserves a pending bounded acquire")

        owner_thread.switch()
        gdb.execute("set $pc = gdb_ame_owner_release")
        wait_done = gdb.Breakpoint("gdb_ame_after_wait")
        gdb.execute("continue")
        wait_done.delete()

        other_threads[0].switch()
        gdb.execute("maintenance flush register-cache")
        report(reg_u32("a0") == 1 and reg_u32("ameown") == 1 and
               int(gdb.parse_and_eval(
                   "*(int *)&gdb_ame_wait_result")) == 1,
               "release grants a waiter after a debugger stop")

        # Release the waiter-owned slot, reacquire it on the original owner,
        # and create a second pending wait.  Redirecting the waiter PC is an
        # architectural state edit, so it must cancel the hidden wait state.
        gdb.execute("set $pc = gdb_ame_waiter_release")
        waiter_release = gdb.Breakpoint("gdb_ame_after_waiter_release")
        gdb.execute("continue")
        waiter_release.delete()

        owner_thread.switch()
        reacquire = gdb.Breakpoint("gdb_ame_after_acquire")
        gdb.execute("set $pc = gdb_ame_acquire")
        gdb.execute("continue")
        reacquire.delete()
        report(reg_u32("ameown") == 1,
               "the original owner reacquires the released backend")

        gdb.execute("set {int}&gdb_ame_wait_ready = 0")
        gdb.execute("set {int}&gdb_ame_wait_done = 0")
        gdb.execute("set {int}&gdb_ame_wait_result = 0")
        gdb.execute("set $pc = gdb_ame_owner_hold")
        other_threads[0].switch()
        gdb.execute("set $pc = gdb_ame_waiter")
        edit_stop = gdb.Breakpoint("gdb_ame_owner_wait_stopped")
        gdb.execute("continue")
        edit_stop.delete()

        other_threads[0].switch()
        gdb.execute("set $pc = gdb_ame_wait_cancelled")
        cancelled = gdb.Breakpoint("gdb_ame_waiter_done")
        gdb.execute("continue")
        cancelled.delete()
        report(reg_u32("ameown") == 0 and
               int(gdb.parse_and_eval(
                   "*(int *)&gdb_ame_wait_result")) == 0x77,
               "editing a waiter PC cancels the pending acquire")

        owner_thread.switch()
        gdb.execute("set $pc = gdb_ame_owner_release")
        owner_release = gdb.Breakpoint("gdb_ame_after_wait")
        gdb.execute("continue")
        owner_release.delete()
        gdb.execute("maintenance flush register-cache")

    gdb.execute("monitor system_reset")
    gdb.execute("maintenance flush register-cache")
    report(reg_byte(m_first, 0) == 0 and reg_u32("md0") == 0 and
           reg_u32("amestype") == 0 and
           reg_u32("amefflags") == 0 and reg_u32("amexsat") == 0 and
           reg_u32("ameown") == 0,
           "system reset clears exposed AME state")


main(run_test)
