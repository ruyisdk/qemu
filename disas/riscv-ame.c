/*
 * QEMU RISC-V Attached Matrix Extension disassembler
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "disas/riscv-ame.h"

typedef enum {
    /* 0 is reserved for rv_op_illegal. */
    ame_op_ame_acquire = 1,
    ame_op_ame_release,
    ame_op_agettyp,
    ame_op_asettyp,
    ame_op_mabs_ew,
    ame_op_mabsdiff_ew,
    ame_op_mabsdiff_ew_x,
    ame_op_madd_ew,
    ame_op_madd_ew_x,
    ame_op_mand_ew,
    ame_op_mand_ew_x,
    ame_op_mandnot_ew,
    ame_op_mandnot_ew_x,
    ame_op_mcmovge_ew,
    ame_op_mcmovlt_ew,
    ame_op_mcmpge_ew,
    ame_op_mcmpge_ew_x,
    ame_op_mcmplt_ew,
    ame_op_mcmplt_ew_x,
    ame_op_mcolbcast_ew_x,
    ame_op_mcolgather_ew,
    ame_op_mcolid_ew,
    ame_op_mcolshift_ew_x,
    ame_op_mcolunzip_ew,
    ame_op_mcolzip_ew,
    ame_op_mconv_ew,
    ame_op_mbcast_m_x,
    ame_op_mcos_ew,
    ame_op_mexp2_ew,
    ame_op_mfrintm_ew,
    ame_op_mfrintn_ew,
    ame_op_mfrintp_ew,
    ame_op_mfrintz_ew,
    ame_op_mgettyp,
    ame_op_mhdiff_ew,
    ame_op_mhdiff_ew_x,
    ame_op_mldexp_ew,
    ame_op_mldexp_ew_x,
    ame_op_mldexpacc_ew,
    ame_op_mldexpacc_ew_x,
    ame_op_mlog2_ew,
    ame_op_mlog2sub_ew,
    ame_op_mlog2sub_ew_x,
    ame_op_mls_1r,
    ame_op_mls_cm,
    ame_op_mls_rm,
    ame_op_mls_st,
    ame_op_mls_tst,
    ame_op_mmax_ew,
    ame_op_mmax_ew_x,
    ame_op_mmean_ew,
    ame_op_mmean_ew_x,
    ame_op_mmin_ew,
    ame_op_mmin_ew_x,
    ame_op_mmov_m_a,
    ame_op_mmov_a_m,
    ame_op_mmov_m_m,
    ame_op_mmove8_m_x,
    ame_op_mmove16_m_x,
    ame_op_mmove32_m_x,
    ame_op_mmove64_m_x,
    ame_op_mmove8_x_m,
    ame_op_mmove16_x_m,
    ame_op_mmove32_x_m,
    ame_op_mmove64_x_m,
    ame_op_mmul_ew,
    ame_op_mmul_ew_x,
    ame_op_mmulacc_2d,
    ame_op_mmulacc_ew,
    ame_op_mmulacc_ew_x,
    ame_op_mmulaccneg_2d,
    ame_op_mmulaccneg_ew,
    ame_op_mmulaccneg_ew_x,
    ame_op_mmuladd_ew,
    ame_op_mmuladd_ew_x,
    ame_op_mmulatacc_2d,
    ame_op_mmulataccneg_2d,
    ame_op_mmulbtacc_2d,
    ame_op_mmulbtaccneg_2d,
    ame_op_mmulneg_ew,
    ame_op_mmulneg_ew_x,
    ame_op_mmulsub_ew,
    ame_op_mmulsub_ew_x,
    ame_op_mor_ew,
    ame_op_mor_ew_x,
    ame_op_mornot_ew,
    ame_op_mornot_ew_x,
    ame_op_mpack_ew_x,
    ame_op_mprefixadd_col,
    ame_op_mprefixadd_row,
    ame_op_mprefixmax_col,
    ame_op_mprefixmax_row,
    ame_op_mrdexp_ew,
    ame_op_mrdexpacc_ew,
    ame_op_mrec_ew,
    ame_op_mreduceadd_col,
    ame_op_mreduceadd_row,
    ame_op_mreducemax_col,
    ame_op_mreducemax_row,
    ame_op_mreducemin_col,
    ame_op_mreducemin_row,
    ame_op_mrowbcast_ew_x,
    ame_op_mrowgather_ew,
    ame_op_mrowid_ew,
    ame_op_mrowshift_ew_x,
    ame_op_mrowunzip_ew,
    ame_op_mrowzip_ew,
    ame_op_mrsqrt_ew,
    ame_op_mrowscatadd_ew,
    ame_op_mcolscatadd_ew,
    ame_op_mrowscatmax_ew,
    ame_op_mcolscatmax_ew,
    ame_op_mselge_ew,
    ame_op_msellt_ew,
    ame_op_msettyp,
    ame_op_msin_ew,
    ame_op_msll_ew,
    ame_op_msll_ew_x,
    ame_op_msqrt_ew,
    ame_op_msra_ew,
    ame_op_msra_ew_x,
    ame_op_msrl_ew,
    ame_op_msrl_ew_x,
    ame_op_mss_1r,
    ame_op_mss_cm,
    ame_op_mss_rm,
    ame_op_mss_st,
    ame_op_mss_tst,
    ame_op_msub_ew,
    ame_op_msub_ew_x,
    ame_op_msublog2_ew,
    ame_op_msublog2_ew_x,
    ame_op_mtanh_ew,
    ame_op_munpack_ew_x,
    ame_op_mxor_ew,
    ame_op_mxor_ew_x,
    ame_op_mzero_2d_acc,
    ame_op_mzero_2d_m,
} rv_ame_op;

const rv_opcode_data ame_opcode_data[] = {
    [rv_op_illegal] = { "illegal", rv_codec_illegal, rv_fmt_none },
    [ame_op_ame_acquire] = { "ame.acquire", rv_codec_ame, "O\t0,1" },
    [ame_op_ame_release] = { "ame.release", rv_codec_ame, "O\t" },
    [ame_op_agettyp] = { "agettyp", rv_codec_ame, "O\t0,C" },
    [ame_op_asettyp] = { "asettyp", rv_codec_ame, "O\tB,1" },
    [ame_op_mabs_ew] = { "mabs.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mabsdiff_ew] = { "mabsdiff.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mabsdiff_ew_x] = { "mabsdiff.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_madd_ew] = { "madd.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_madd_ew_x] = { "madd.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mand_ew] = { "mand.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mand_ew_x] = { "mand.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mandnot_ew] = { "mandnot.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mandnot_ew_x] = { "mandnot.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mcmovge_ew] = { "mcmovge.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcmovlt_ew] = { "mcmovlt.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcmpge_ew] = { "mcmpge.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcmpge_ew_x] = { "mcmpge.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mcmplt_ew] = { "mcmplt.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcmplt_ew_x] = { "mcmplt.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mcolbcast_ew_x] = { "mcolbcast.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mcolgather_ew] = { "mcolgather.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcolid_ew] = { "mcolid.ew", rv_codec_ame, "O\tM" },
    [ame_op_mcolshift_ew_x] = { "mcolshift.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mcolunzip_ew] = { "mcolunzip.ew", rv_codec_ame, "O\tN,P" },
    [ame_op_mcolzip_ew] = { "mcolzip.ew", rv_codec_ame, "O\tN,P" },
    [ame_op_mconv_ew] = { "mconv.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mbcast_m_x] = { "mbcast.m.x", rv_codec_ame, "O\tM,1,2" },
    [ame_op_mcos_ew] = { "mcos.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mexp2_ew] = { "mexp2.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mfrintm_ew] = { "mfrintm.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mfrintn_ew] = { "mfrintn.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mfrintp_ew] = { "mfrintp.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mfrintz_ew] = { "mfrintz.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mgettyp] = { "mgettyp", rv_codec_ame, "O\t0,N" },
    [ame_op_mhdiff_ew] = { "mhdiff.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mhdiff_ew_x] = { "mhdiff.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mldexp_ew] = { "mldexp.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mldexp_ew_x] = { "mldexp.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mldexpacc_ew] = { "mldexpacc.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mldexpacc_ew_x] = { "mldexpacc.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mlog2_ew] = { "mlog2.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mlog2sub_ew] = { "mlog2sub.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mlog2sub_ew_x] = { "mlog2sub.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mls_1r] = { "mls.1r", rv_codec_ame, "O\tM,1" },
    [ame_op_mls_cm] = { "mls.cm", rv_codec_ame, "O\tM,1" },
    [ame_op_mls_rm] = { "mls.rm", rv_codec_ame, "O\tM,1" },
    [ame_op_mls_st] = { "mls.st", rv_codec_ame, "O\tM,(1),2" },
    [ame_op_mls_tst] = { "mls.tst", rv_codec_ame, "O\tM,(1),2" },
    [ame_op_mmax_ew] = { "mmax.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmax_ew_x] = { "mmax.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmean_ew] = { "mmean.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmean_ew_x] = { "mmean.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmin_ew] = { "mmin.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmin_ew_x] = { "mmin.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmov_m_a] = { "mmov.m.a", rv_codec_ame, "O\tM,C" },
    [ame_op_mmov_a_m] = { "mmov.a.m", rv_codec_ame, "O\tB,N" },
    [ame_op_mmov_m_m] = { "mmov.m.m", rv_codec_ame, "O\tM,N" },
    [ame_op_mmove8_m_x] = { "mmove8.m.x", rv_codec_ame, "O\tM,1,2" },
    [ame_op_mmove16_m_x] = { "mmove16.m.x", rv_codec_ame, "O\tM,1,2" },
    [ame_op_mmove32_m_x] = { "mmove32.m.x", rv_codec_ame, "O\tM,1,2" },
    [ame_op_mmove64_m_x] = { "mmove64.m.x", rv_codec_ame, "O\tM,1,2" },
    [ame_op_mmove8_x_m] = { "mmove8.x.m", rv_codec_ame, "O\t0,N,2" },
    [ame_op_mmove16_x_m] = { "mmove16.x.m", rv_codec_ame, "O\t0,N,2" },
    [ame_op_mmove32_x_m] = { "mmove32.x.m", rv_codec_ame, "O\t0,N,2" },
    [ame_op_mmove64_x_m] = { "mmove64.x.m", rv_codec_ame, "O\t0,N,2" },
    [ame_op_mmul_ew] = { "mmul.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmul_ew_x] = { "mmul.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmulacc_2d] = { "mmulacc.2d", rv_codec_ame, "O\tB,N,P" },
    [ame_op_mmulacc_ew] = { "mmulacc.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmulacc_ew_x] = { "mmulacc.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmulaccneg_2d] = { "mmulaccneg.2d", rv_codec_ame, "O\tB,N,P" },
    [ame_op_mmulaccneg_ew] = { "mmulaccneg.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmulaccneg_ew_x] = { "mmulaccneg.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmuladd_ew] = { "mmuladd.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmuladd_ew_x] = { "mmuladd.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmulatacc_2d] = { "mmulatacc.2d", rv_codec_ame, "O\tB,N,P" },
    [ame_op_mmulataccneg_2d] = { "mmulataccneg.2d", rv_codec_ame, "O\tB,N,P" },
    [ame_op_mmulbtacc_2d] = { "mmulbtacc.2d", rv_codec_ame, "O\tB,N,P" },
    [ame_op_mmulbtaccneg_2d] = { "mmulbtaccneg.2d", rv_codec_ame, "O\tB,N,P" },
    [ame_op_mmulneg_ew] = { "mmulneg.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmulneg_ew_x] = { "mmulneg.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mmulsub_ew] = { "mmulsub.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mmulsub_ew_x] = { "mmulsub.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mor_ew] = { "mor.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mor_ew_x] = { "mor.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mornot_ew] = { "mornot.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mornot_ew_x] = { "mornot.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mpack_ew_x] = { "mpack.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mprefixadd_col] = { "mprefixadd.col", rv_codec_ame, "O\tM,N" },
    [ame_op_mprefixadd_row] = { "mprefixadd.row", rv_codec_ame, "O\tM,N" },
    [ame_op_mprefixmax_col] = { "mprefixmax.col", rv_codec_ame, "O\tM,N" },
    [ame_op_mprefixmax_row] = { "mprefixmax.row", rv_codec_ame, "O\tM,N" },
    [ame_op_mrdexp_ew] = { "mrdexp.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mrdexpacc_ew] = { "mrdexpacc.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mrec_ew] = { "mrec.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mreduceadd_col] = { "mreduceadd.col", rv_codec_ame, "O\tM,N" },
    [ame_op_mreduceadd_row] = { "mreduceadd.row", rv_codec_ame, "O\tM,N" },
    [ame_op_mreducemax_col] = { "mreducemax.col", rv_codec_ame, "O\tM,N" },
    [ame_op_mreducemax_row] = { "mreducemax.row", rv_codec_ame, "O\tM,N" },
    [ame_op_mreducemin_col] = { "mreducemin.col", rv_codec_ame, "O\tM,N" },
    [ame_op_mreducemin_row] = { "mreducemin.row", rv_codec_ame, "O\tM,N" },
    [ame_op_mrowbcast_ew_x] = { "mrowbcast.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mrowgather_ew] = { "mrowgather.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mrowid_ew] = { "mrowid.ew", rv_codec_ame, "O\tM" },
    [ame_op_mrowshift_ew_x] = { "mrowshift.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mrowunzip_ew] = { "mrowunzip.ew", rv_codec_ame, "O\tN,P" },
    [ame_op_mrowzip_ew] = { "mrowzip.ew", rv_codec_ame, "O\tN,P" },
    [ame_op_mrsqrt_ew] = { "mrsqrt.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_mrowscatadd_ew] = { "mrowscatadd.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcolscatadd_ew] = { "mcolscatadd.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mrowscatmax_ew] = { "mrowscatmax.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mcolscatmax_ew] = { "mcolscatmax.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mselge_ew] = { "mselge.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msellt_ew] = { "msellt.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msettyp] = { "msettyp", rv_codec_ame, "O\tM,1" },
    [ame_op_msin_ew] = { "msin.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_msll_ew] = { "msll.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msll_ew_x] = { "msll.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_msqrt_ew] = { "msqrt.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_msra_ew] = { "msra.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msra_ew_x] = { "msra.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_msrl_ew] = { "msrl.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msrl_ew_x] = { "msrl.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mss_1r] = { "mss.1r", rv_codec_ame, "O\tM,1" },
    [ame_op_mss_cm] = { "mss.cm", rv_codec_ame, "O\tM,1" },
    [ame_op_mss_rm] = { "mss.rm", rv_codec_ame, "O\tM,1" },
    [ame_op_mss_st] = { "mss.st", rv_codec_ame, "O\tM,(1),2" },
    [ame_op_mss_tst] = { "mss.tst", rv_codec_ame, "O\tM,(1),2" },
    [ame_op_msub_ew] = { "msub.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msub_ew_x] = { "msub.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_msublog2_ew] = { "msublog2.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_msublog2_ew_x] = { "msublog2.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mtanh_ew] = { "mtanh.ew", rv_codec_ame, "O\tM,N" },
    [ame_op_munpack_ew_x] = { "munpack.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mxor_ew] = { "mxor.ew", rv_codec_ame, "O\tM,N,P" },
    [ame_op_mxor_ew_x] = { "mxor.ew.x", rv_codec_ame, "O\tM,1,P" },
    [ame_op_mzero_2d_acc] = { "mzero.2d.acc", rv_codec_ame, "O\tB" },
    [ame_op_mzero_2d_m] = { "mzero.2d.m", rv_codec_ame, "O\tM" },
};

void decode_ame(rv_decode *dec, rv_isa isa)
{
    rv_inst inst = dec->inst;
    rv_opcode op = rv_op_illegal;

    /* Ztt uses custom-1 with funct3 = 0. */
    switch (inst & 0x707f) {
    case 0x2b:
        switch ((inst >> 25) & 0x7f) {
        case 0x00:
            op = ame_op_mabsdiff_ew;
            break;
        case 0x01:
            op = ame_op_mabsdiff_ew_x;
            break;
        case 0x02:
            op = ame_op_madd_ew;
            break;
        case 0x03:
            op = ame_op_madd_ew_x;
            break;
        case 0x04:
            op = ame_op_mhdiff_ew;
            break;
        case 0x05:
            op = ame_op_mhdiff_ew_x;
            break;
        case 0x06:
            op = ame_op_mmax_ew;
            break;
        case 0x07:
            op = ame_op_mmax_ew_x;
            break;
        case 0x08:
            op = ame_op_mmean_ew;
            break;
        case 0x09:
            op = ame_op_mmean_ew_x;
            break;
        case 0x0a:
            op = ame_op_mmin_ew;
            break;
        case 0x0b:
            op = ame_op_mmin_ew_x;
            break;
        case 0x0c:
            op = ame_op_mmul_ew;
            break;
        case 0x0d:
            op = ame_op_mmul_ew_x;
            break;
        case 0x0e:
            op = ame_op_mmulacc_ew;
            break;
        case 0x0f:
            op = ame_op_mmulacc_ew_x;
            break;
        case 0x10:
            op = ame_op_mmulaccneg_ew;
            break;
        case 0x11:
            op = ame_op_mmulaccneg_ew_x;
            break;
        case 0x12:
            op = ame_op_mmuladd_ew;
            break;
        case 0x13:
            op = ame_op_mmuladd_ew_x;
            break;
        case 0x14:
            op = ame_op_mmulneg_ew;
            break;
        case 0x15:
            op = ame_op_mmulneg_ew_x;
            break;
        case 0x16:
            op = ame_op_mmulsub_ew;
            break;
        case 0x17:
            op = ame_op_mmulsub_ew_x;
            break;
        case 0x18:
            op = ame_op_msub_ew;
            break;
        case 0x19:
            op = ame_op_msub_ew_x;
            break;
        case 0x1a:
            op = ame_op_mand_ew;
            break;
        case 0x1b:
            op = ame_op_mand_ew_x;
            break;
        case 0x1c:
            op = ame_op_mandnot_ew;
            break;
        case 0x1d:
            op = ame_op_mandnot_ew_x;
            break;
        case 0x1e:
            op = ame_op_mor_ew;
            break;
        case 0x1f:
            op = ame_op_mor_ew_x;
            break;
        case 0x20:
            op = ame_op_mornot_ew;
            break;
        case 0x21:
            op = ame_op_mornot_ew_x;
            break;
        case 0x22:
            op = ame_op_msll_ew;
            break;
        case 0x23:
            op = ame_op_msll_ew_x;
            break;
        case 0x24:
            op = ame_op_msra_ew;
            break;
        case 0x25:
            op = ame_op_msra_ew_x;
            break;
        case 0x26:
            op = ame_op_msrl_ew;
            break;
        case 0x27:
            op = ame_op_msrl_ew_x;
            break;
        case 0x28:
            op = ame_op_mxor_ew;
            break;
        case 0x29:
            op = ame_op_mxor_ew_x;
            break;
        case 0x2a:
            op = ame_op_mcmovge_ew;
            break;
        case 0x2b:
            op = ame_op_mcmovlt_ew;
            break;
        case 0x2c:
            op = ame_op_mcmpge_ew;
            break;
        case 0x2d:
            op = ame_op_mcmpge_ew_x;
            break;
        case 0x2e:
            op = ame_op_mcmplt_ew;
            break;
        case 0x2f:
            op = ame_op_mcmplt_ew_x;
            break;
        case 0x30:
            op = ame_op_mselge_ew;
            break;
        case 0x31:
            op = ame_op_msellt_ew;
            break;
        case 0x32:
            op = ame_op_mcolbcast_ew_x;
            break;
        case 0x33:
            op = ame_op_mcolgather_ew;
            break;
        case 0x34:
            op = ame_op_mcolshift_ew_x;
            break;
        case 0x35:
            op = ame_op_mrowbcast_ew_x;
            break;
        case 0x36:
            op = ame_op_mrowgather_ew;
            break;
        case 0x37:
            op = ame_op_mrowshift_ew_x;
            break;
        case 0x38:
            if (((inst >> 7) & 0x1f) == 0) {
                op = ame_op_mrowzip_ew;
            }
            break;
        case 0x39:
            op = ame_op_mrowscatadd_ew;
            break;
        case 0x3a:
            op = ame_op_mcolscatadd_ew;
            break;
        case 0x3b:
            op = ame_op_mrowscatmax_ew;
            break;
        case 0x3c:
            op = ame_op_mcolscatmax_ew;
            break;
        case 0x3d:
            op = ame_op_mpack_ew_x;
            break;
        case 0x3e:
            op = ame_op_munpack_ew_x;
            break;
        case 0x3f:
            op = ame_op_mldexp_ew;
            break;
        case 0x40:
            op = ame_op_mldexp_ew_x;
            break;
        case 0x41:
            op = ame_op_mldexpacc_ew;
            break;
        case 0x42:
            op = ame_op_mldexpacc_ew_x;
            break;
        case 0x43:
            op = ame_op_mlog2sub_ew;
            break;
        case 0x44:
            op = ame_op_mlog2sub_ew_x;
            break;
        case 0x45:
            op = ame_op_mrdexp_ew;
            break;
        case 0x46:
            op = ame_op_mrdexpacc_ew;
            break;
        case 0x47:
            op = ame_op_msublog2_ew;
            break;
        case 0x48:
            op = ame_op_msublog2_ew_x;
            break;
        case 0x4a:
            op = ame_op_mmulacc_2d;
            break;
        case 0x4b:
            op = ame_op_mmulaccneg_2d;
            break;
        case 0x4d:
            op = ame_op_mmulatacc_2d;
            break;
        case 0x4e:
            op = ame_op_mmulataccneg_2d;
            break;
        case 0x4f:
            op = ame_op_mmulbtacc_2d;
            break;
        case 0x50:
            op = ame_op_mmulbtaccneg_2d;
            break;
        case 0x51:
            switch ((inst >> 20) & 0x1f) {
            case 0x00:
                op = ame_op_agettyp;
                break;
            case 0x01:
                op = ame_op_asettyp;
                break;
            case 0x02:
                op = ame_op_mgettyp;
                break;
            case 0x03:
                op = ame_op_msettyp;
                break;
            case 0x04:
                op = ame_op_mabs_ew;
                break;
            case 0x05:
                op = ame_op_mfrintm_ew;
                break;
            case 0x06:
                op = ame_op_mfrintn_ew;
                break;
            case 0x07:
                op = ame_op_mfrintp_ew;
                break;
            case 0x08:
                op = ame_op_mfrintz_ew;
                break;
            case 0x0d:
                op = ame_op_mconv_ew;
                break;
            case 0x0e:
                op = ame_op_mcos_ew;
                break;
            case 0x0f:
                op = ame_op_mexp2_ew;
                break;
            case 0x10:
                op = ame_op_mlog2_ew;
                break;
            case 0x11:
                op = ame_op_mrec_ew;
                break;
            case 0x12:
                op = ame_op_mrsqrt_ew;
                break;
            case 0x13:
                op = ame_op_msin_ew;
                break;
            case 0x14:
                op = ame_op_msqrt_ew;
                break;
            case 0x15:
                op = ame_op_mtanh_ew;
                break;
            case 0x16:
                op = ame_op_mls_1r;
                break;
            case 0x17:
                op = ame_op_mls_cm;
                break;
            case 0x18:
                op = ame_op_mls_rm;
                break;
            case 0x19:
                op = ame_op_mss_1r;
                break;
            case 0x1a:
                op = ame_op_mss_cm;
                break;
            case 0x1b:
                op = ame_op_mss_rm;
                break;
            case 0x1c:
                op = ame_op_mmov_m_a;
                break;
            case 0x1d:
                op = ame_op_mmov_a_m;
                break;
            case 0x1e:
                op = ame_op_mmov_m_m;
                break;
            case 0x1f:
                op = ame_op_mprefixadd_col;
                break;
            }
            break;
        case 0x52:
            switch ((inst >> 20) & 0x1f) {
            case 0x00:
                op = ame_op_mprefixadd_row;
                break;
            case 0x01:
                op = ame_op_mprefixmax_col;
                break;
            case 0x02:
                op = ame_op_mprefixmax_row;
                break;
            case 0x03:
                op = ame_op_mreduceadd_col;
                break;
            case 0x04:
                op = ame_op_mreduceadd_row;
                break;
            case 0x05:
                op = ame_op_mreducemax_col;
                break;
            case 0x06:
                op = ame_op_mreducemax_row;
                break;
            case 0x07:
                op = ame_op_mreducemin_col;
                break;
            case 0x08:
                op = ame_op_mreducemin_row;
                break;
            case 0x09:
                op = ame_op_ame_acquire;
                break;
            case 0x0a:
                /* ame.release also fixes rd and rs1 to zero. */
                if ((inst & 0xf8f80) == 0) {
                    op = ame_op_ame_release;
                }
                break;
            }
            break;
        case 0x53:
            /* R1 fixes all source bits (rs2 and rs1). */
            switch ((inst >> 15) & 0x3ff) {
            case 0:
                op = ame_op_mzero_2d_acc;
                break;
            case 1:
                op = ame_op_mzero_2d_m;
                break;
            case 2:
                op = ame_op_mcolid_ew;
                break;
            case 3:
                op = ame_op_mrowid_ew;
                break;
            }
            break;
        case 0x54:
            op = ame_op_mls_st;
            break;
        case 0x55:
            op = ame_op_mls_tst;
            break;
        case 0x56:
            op = ame_op_mss_st;
            break;
        case 0x57:
            op = ame_op_mss_tst;
            break;
        case 0x58:
            op = ame_op_mbcast_m_x;
            break;
        case 0x59:
            op = ame_op_mmove8_m_x;
            break;
        case 0x5a:
            op = ame_op_mmove16_m_x;
            break;
        case 0x5b:
            op = ame_op_mmove32_m_x;
            break;
        case 0x5c:
            op = ame_op_mmove64_m_x;
            break;
        case 0x5d:
            op = ame_op_mmove8_x_m;
            break;
        case 0x5e:
            op = ame_op_mmove16_x_m;
            break;
        case 0x5f:
            op = ame_op_mmove32_x_m;
            break;
        case 0x60:
            op = ame_op_mmove64_x_m;
            break;
        case 0x61:
            if (((inst >> 7) & 0x1f) == 0) {
                op = ame_op_mcolunzip_ew;
            }
            break;
        case 0x62:
            if (((inst >> 7) & 0x1f) == 0) {
                op = ame_op_mcolzip_ew;
            }
            break;
        case 0x63:
            if (((inst >> 7) & 0x1f) == 0) {
                op = ame_op_mrowunzip_ew;
            }
            break;
        }
        break;
    }

    dec->op = op;
}
