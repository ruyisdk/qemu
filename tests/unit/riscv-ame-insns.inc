/*
 * Ztt v0.6 encoding and disassembly fixtures.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Values from the specification's INSTRUCTION_ENCODINGS.md, independent
 * of target/riscv/ztt.decode and the production disassembler.
 * The disassembly operands use rd=3, rs1=4, rs2=5 in physical field order.
 */

AME_TEST_INSN(ame_acquire, R2, 0xa490002b,
              "ame.acquire", "gp,tp")
AME_TEST_INSN(ame_release, FIXED, 0xa4a0002b,
              "ame.release", "")
AME_TEST_INSN(agettyp, R2, 0xa200002b,
              "agettyp", "gp,acc4")
AME_TEST_INSN(asettyp, R2, 0xa210002b,
              "asettyp", "acc3,tp")
AME_TEST_INSN(mabs_ew, R2, 0xa240002b,
              "mabs.ew", "m3,m4")
AME_TEST_INSN(mabsdiff_ew, R3, 0x0000002b,
              "mabsdiff.ew", "m3,m4,m5")
AME_TEST_INSN(mabsdiff_ew_x, R3, 0x0200002b,
              "mabsdiff.ew.x", "m3,tp,m5")
AME_TEST_INSN(madd_ew, R3, 0x0400002b,
              "madd.ew", "m3,m4,m5")
AME_TEST_INSN(madd_ew_x, R3, 0x0600002b,
              "madd.ew.x", "m3,tp,m5")
AME_TEST_INSN(mand_ew, R3, 0x3400002b,
              "mand.ew", "m3,m4,m5")
AME_TEST_INSN(mand_ew_x, R3, 0x3600002b,
              "mand.ew.x", "m3,tp,m5")
AME_TEST_INSN(mandnot_ew, R3, 0x3800002b,
              "mandnot.ew", "m3,m4,m5")
AME_TEST_INSN(mandnot_ew_x, R3, 0x3a00002b,
              "mandnot.ew.x", "m3,tp,m5")
AME_TEST_INSN(mcmovge_ew, R3, 0x5400002b,
              "mcmovge.ew", "m3,m4,m5")
AME_TEST_INSN(mcmovlt_ew, R3, 0x5600002b,
              "mcmovlt.ew", "m3,m4,m5")
AME_TEST_INSN(mcmpge_ew, R3, 0x5800002b,
              "mcmpge.ew", "m3,m4,m5")
AME_TEST_INSN(mcmpge_ew_x, R3, 0x5a00002b,
              "mcmpge.ew.x", "m3,tp,m5")
AME_TEST_INSN(mcmplt_ew, R3, 0x5c00002b,
              "mcmplt.ew", "m3,m4,m5")
AME_TEST_INSN(mcmplt_ew_x, R3, 0x5e00002b,
              "mcmplt.ew.x", "m3,tp,m5")
AME_TEST_INSN(mcolbcast_ew_x, R3, 0x6400002b,
              "mcolbcast.ew.x", "m3,tp,m5")
AME_TEST_INSN(mcolgather_ew, R3, 0x6600002b,
              "mcolgather.ew", "m3,m4,m5")
AME_TEST_INSN(mcolid_ew, R1, 0xa601002b,
              "mcolid.ew", "m3")
AME_TEST_INSN(mcolshift_ew_x, R3, 0x6800002b,
              "mcolshift.ew.x", "m3,tp,m5")
AME_TEST_INSN(mcolunzip_ew, ZIP, 0xc200002b,
              "mcolunzip.ew", "m4,m5")
AME_TEST_INSN(mcolzip_ew, ZIP, 0xc400002b,
              "mcolzip.ew", "m4,m5")
AME_TEST_INSN(mconv_ew, R2, 0xa2d0002b,
              "mconv.ew", "m3,m4")
AME_TEST_INSN(mbcast_m_x, R3, 0xb000002b,
              "mbcast.m.x", "m3,tp,t0")
AME_TEST_INSN(mcos_ew, R2, 0xa2e0002b,
              "mcos.ew", "m3,m4")
AME_TEST_INSN(mexp2_ew, R2, 0xa2f0002b,
              "mexp2.ew", "m3,m4")
AME_TEST_INSN(mfrintm_ew, R2, 0xa250002b,
              "mfrintm.ew", "m3,m4")
AME_TEST_INSN(mfrintn_ew, R2, 0xa260002b,
              "mfrintn.ew", "m3,m4")
AME_TEST_INSN(mfrintp_ew, R2, 0xa270002b,
              "mfrintp.ew", "m3,m4")
AME_TEST_INSN(mfrintz_ew, R2, 0xa280002b,
              "mfrintz.ew", "m3,m4")
AME_TEST_INSN(mgettyp, R2, 0xa220002b,
              "mgettyp", "gp,m4")
AME_TEST_INSN(mhdiff_ew, R3, 0x0800002b,
              "mhdiff.ew", "m3,m4,m5")
AME_TEST_INSN(mhdiff_ew_x, R3, 0x0a00002b,
              "mhdiff.ew.x", "m3,tp,m5")
AME_TEST_INSN(mldexp_ew, R3, 0x7e00002b,
              "mldexp.ew", "m3,m4,m5")
AME_TEST_INSN(mldexp_ew_x, R3, 0x8000002b,
              "mldexp.ew.x", "m3,tp,m5")
AME_TEST_INSN(mldexpacc_ew, R3, 0x8200002b,
              "mldexpacc.ew", "m3,m4,m5")
AME_TEST_INSN(mldexpacc_ew_x, R3, 0x8400002b,
              "mldexpacc.ew.x", "m3,tp,m5")
AME_TEST_INSN(mlog2_ew, R2, 0xa300002b,
              "mlog2.ew", "m3,m4")
AME_TEST_INSN(mlog2sub_ew, R3, 0x8600002b,
              "mlog2sub.ew", "m3,m4,m5")
AME_TEST_INSN(mlog2sub_ew_x, R3, 0x8800002b,
              "mlog2sub.ew.x", "m3,tp,m5")
AME_TEST_INSN(mls_1r, R2, 0xa360002b,
              "mls.1r", "m3,tp")
AME_TEST_INSN(mls_cm, R2, 0xa370002b,
              "mls.cm", "m3,tp")
AME_TEST_INSN(mls_rm, R2, 0xa380002b,
              "mls.rm", "m3,tp")
AME_TEST_INSN(mls_st, R3, 0xa800002b,
              "mls.st", "m3,(tp),t0")
AME_TEST_INSN(mls_tst, R3, 0xaa00002b,
              "mls.tst", "m3,(tp),t0")
AME_TEST_INSN(mmax_ew, R3, 0x0c00002b,
              "mmax.ew", "m3,m4,m5")
AME_TEST_INSN(mmax_ew_x, R3, 0x0e00002b,
              "mmax.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmean_ew, R3, 0x1000002b,
              "mmean.ew", "m3,m4,m5")
AME_TEST_INSN(mmean_ew_x, R3, 0x1200002b,
              "mmean.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmin_ew, R3, 0x1400002b,
              "mmin.ew", "m3,m4,m5")
AME_TEST_INSN(mmin_ew_x, R3, 0x1600002b,
              "mmin.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmov_m_a, R2, 0xa3c0002b,
              "mmov.m.a", "m3,acc4")
AME_TEST_INSN(mmov_a_m, R2, 0xa3d0002b,
              "mmov.a.m", "acc3,m4")
AME_TEST_INSN(mmov_m_m, R2, 0xa3e0002b,
              "mmov.m.m", "m3,m4")
AME_TEST_INSN(mmove8_m_x, R3, 0xb200002b,
              "mmove8.m.x", "m3,tp,t0")
AME_TEST_INSN(mmove16_m_x, R3, 0xb400002b,
              "mmove16.m.x", "m3,tp,t0")
AME_TEST_INSN(mmove32_m_x, R3, 0xb600002b,
              "mmove32.m.x", "m3,tp,t0")
AME_TEST_INSN(mmove64_m_x, R3, 0xb800002b,
              "mmove64.m.x", "m3,tp,t0")
AME_TEST_INSN(mmove8_x_m, R3, 0xba00002b,
              "mmove8.x.m", "gp,m4,t0")
AME_TEST_INSN(mmove16_x_m, R3, 0xbc00002b,
              "mmove16.x.m", "gp,m4,t0")
AME_TEST_INSN(mmove32_x_m, R3, 0xbe00002b,
              "mmove32.x.m", "gp,m4,t0")
AME_TEST_INSN(mmove64_x_m, R3, 0xc000002b,
              "mmove64.x.m", "gp,m4,t0")
AME_TEST_INSN(mmul_ew, R3, 0x1800002b,
              "mmul.ew", "m3,m4,m5")
AME_TEST_INSN(mmul_ew_x, R3, 0x1a00002b,
              "mmul.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmulacc_2d, R3, 0x9400002b,
              "mmulacc.2d", "acc3,m4,m5")
AME_TEST_INSN(mmulacc_ew, R3, 0x1c00002b,
              "mmulacc.ew", "m3,m4,m5")
AME_TEST_INSN(mmulacc_ew_x, R3, 0x1e00002b,
              "mmulacc.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmulaccneg_2d, R3, 0x9600002b,
              "mmulaccneg.2d", "acc3,m4,m5")
AME_TEST_INSN(mmulaccneg_ew, R3, 0x2000002b,
              "mmulaccneg.ew", "m3,m4,m5")
AME_TEST_INSN(mmulaccneg_ew_x, R3, 0x2200002b,
              "mmulaccneg.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmuladd_ew, R3, 0x2400002b,
              "mmuladd.ew", "m3,m4,m5")
AME_TEST_INSN(mmuladd_ew_x, R3, 0x2600002b,
              "mmuladd.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmulatacc_2d, R3, 0x9a00002b,
              "mmulatacc.2d", "acc3,m4,m5")
AME_TEST_INSN(mmulataccneg_2d, R3, 0x9c00002b,
              "mmulataccneg.2d", "acc3,m4,m5")
AME_TEST_INSN(mmulbtacc_2d, R3, 0x9e00002b,
              "mmulbtacc.2d", "acc3,m4,m5")
AME_TEST_INSN(mmulbtaccneg_2d, R3, 0xa000002b,
              "mmulbtaccneg.2d", "acc3,m4,m5")
AME_TEST_INSN(mmulneg_ew, R3, 0x2800002b,
              "mmulneg.ew", "m3,m4,m5")
AME_TEST_INSN(mmulneg_ew_x, R3, 0x2a00002b,
              "mmulneg.ew.x", "m3,tp,m5")
AME_TEST_INSN(mmulsub_ew, R3, 0x2c00002b,
              "mmulsub.ew", "m3,m4,m5")
AME_TEST_INSN(mmulsub_ew_x, R3, 0x2e00002b,
              "mmulsub.ew.x", "m3,tp,m5")
AME_TEST_INSN(mor_ew, R3, 0x3c00002b,
              "mor.ew", "m3,m4,m5")
AME_TEST_INSN(mor_ew_x, R3, 0x3e00002b,
              "mor.ew.x", "m3,tp,m5")
AME_TEST_INSN(mornot_ew, R3, 0x4000002b,
              "mornot.ew", "m3,m4,m5")
AME_TEST_INSN(mornot_ew_x, R3, 0x4200002b,
              "mornot.ew.x", "m3,tp,m5")
AME_TEST_INSN(mpack_ew_x, R3, 0x7a00002b,
              "mpack.ew.x", "m3,tp,m5")
AME_TEST_INSN(mprefixadd_col, R2, 0xa3f0002b,
              "mprefixadd.col", "m3,m4")
AME_TEST_INSN(mprefixadd_row, R2, 0xa400002b,
              "mprefixadd.row", "m3,m4")
AME_TEST_INSN(mprefixmax_col, R2, 0xa410002b,
              "mprefixmax.col", "m3,m4")
AME_TEST_INSN(mprefixmax_row, R2, 0xa420002b,
              "mprefixmax.row", "m3,m4")
AME_TEST_INSN(mrdexp_ew, R3, 0x8a00002b,
              "mrdexp.ew", "m3,m4,m5")
AME_TEST_INSN(mrdexpacc_ew, R3, 0x8c00002b,
              "mrdexpacc.ew", "m3,m4,m5")
AME_TEST_INSN(mrec_ew, R2, 0xa310002b,
              "mrec.ew", "m3,m4")
AME_TEST_INSN(mreduceadd_col, R2, 0xa430002b,
              "mreduceadd.col", "m3,m4")
AME_TEST_INSN(mreduceadd_row, R2, 0xa440002b,
              "mreduceadd.row", "m3,m4")
AME_TEST_INSN(mreducemax_col, R2, 0xa450002b,
              "mreducemax.col", "m3,m4")
AME_TEST_INSN(mreducemax_row, R2, 0xa460002b,
              "mreducemax.row", "m3,m4")
AME_TEST_INSN(mreducemin_col, R2, 0xa470002b,
              "mreducemin.col", "m3,m4")
AME_TEST_INSN(mreducemin_row, R2, 0xa480002b,
              "mreducemin.row", "m3,m4")
AME_TEST_INSN(mrowbcast_ew_x, R3, 0x6a00002b,
              "mrowbcast.ew.x", "m3,tp,m5")
AME_TEST_INSN(mrowgather_ew, R3, 0x6c00002b,
              "mrowgather.ew", "m3,m4,m5")
AME_TEST_INSN(mrowid_ew, R1, 0xa601802b,
              "mrowid.ew", "m3")
AME_TEST_INSN(mrowshift_ew_x, R3, 0x6e00002b,
              "mrowshift.ew.x", "m3,tp,m5")
AME_TEST_INSN(mrowunzip_ew, ZIP, 0xc600002b,
              "mrowunzip.ew", "m4,m5")
AME_TEST_INSN(mrowzip_ew, ZIP, 0x7000002b,
              "mrowzip.ew", "m4,m5")
AME_TEST_INSN(mrsqrt_ew, R2, 0xa320002b,
              "mrsqrt.ew", "m3,m4")
AME_TEST_INSN(mrowscatadd_ew, R3, 0x7200002b,
              "mrowscatadd.ew", "m3,m4,m5")
AME_TEST_INSN(mcolscatadd_ew, R3, 0x7400002b,
              "mcolscatadd.ew", "m3,m4,m5")
AME_TEST_INSN(mrowscatmax_ew, R3, 0x7600002b,
              "mrowscatmax.ew", "m3,m4,m5")
AME_TEST_INSN(mcolscatmax_ew, R3, 0x7800002b,
              "mcolscatmax.ew", "m3,m4,m5")
AME_TEST_INSN(mselge_ew, R3, 0x6000002b,
              "mselge.ew", "m3,m4,m5")
AME_TEST_INSN(msellt_ew, R3, 0x6200002b,
              "msellt.ew", "m3,m4,m5")
AME_TEST_INSN(msettyp, R2, 0xa230002b,
              "msettyp", "m3,tp")
AME_TEST_INSN(msin_ew, R2, 0xa330002b,
              "msin.ew", "m3,m4")
AME_TEST_INSN(msll_ew, R3, 0x4400002b,
              "msll.ew", "m3,m4,m5")
AME_TEST_INSN(msll_ew_x, R3, 0x4600002b,
              "msll.ew.x", "m3,tp,m5")
AME_TEST_INSN(msqrt_ew, R2, 0xa340002b,
              "msqrt.ew", "m3,m4")
AME_TEST_INSN(msra_ew, R3, 0x4800002b,
              "msra.ew", "m3,m4,m5")
AME_TEST_INSN(msra_ew_x, R3, 0x4a00002b,
              "msra.ew.x", "m3,tp,m5")
AME_TEST_INSN(msrl_ew, R3, 0x4c00002b,
              "msrl.ew", "m3,m4,m5")
AME_TEST_INSN(msrl_ew_x, R3, 0x4e00002b,
              "msrl.ew.x", "m3,tp,m5")
AME_TEST_INSN(mss_1r, R2, 0xa390002b,
              "mss.1r", "m3,tp")
AME_TEST_INSN(mss_cm, R2, 0xa3a0002b,
              "mss.cm", "m3,tp")
AME_TEST_INSN(mss_rm, R2, 0xa3b0002b,
              "mss.rm", "m3,tp")
AME_TEST_INSN(mss_st, R3, 0xac00002b,
              "mss.st", "m3,(tp),t0")
AME_TEST_INSN(mss_tst, R3, 0xae00002b,
              "mss.tst", "m3,(tp),t0")
AME_TEST_INSN(msub_ew, R3, 0x3000002b,
              "msub.ew", "m3,m4,m5")
AME_TEST_INSN(msub_ew_x, R3, 0x3200002b,
              "msub.ew.x", "m3,tp,m5")
AME_TEST_INSN(msublog2_ew, R3, 0x8e00002b,
              "msublog2.ew", "m3,m4,m5")
AME_TEST_INSN(msublog2_ew_x, R3, 0x9000002b,
              "msublog2.ew.x", "m3,tp,m5")
AME_TEST_INSN(mtanh_ew, R2, 0xa350002b,
              "mtanh.ew", "m3,m4")
AME_TEST_INSN(munpack_ew_x, R3, 0x7c00002b,
              "munpack.ew.x", "m3,tp,m5")
AME_TEST_INSN(mxor_ew, R3, 0x5000002b,
              "mxor.ew", "m3,m4,m5")
AME_TEST_INSN(mxor_ew_x, R3, 0x5200002b,
              "mxor.ew.x", "m3,tp,m5")
AME_TEST_INSN(mzero_2d_acc, R1, 0xa600002b,
              "mzero.2d.acc", "acc3")
AME_TEST_INSN(mzero_2d_m, R1, 0xa600802b,
              "mzero.2d.m", "m3")
