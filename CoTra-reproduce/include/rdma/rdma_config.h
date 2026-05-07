#pragma once

/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Enable CUDA feature */
/* #undef CUDA_PATH */

// #define MACHINE_NUM (4)
// #define MACHINE_NUM (8)
#ifndef MACHINE_NUM
#define MACHINE_NUM (16)
#endif
#define MAX_THREAD_NUM (16)
#define GROUP_SIZE (16)
// NOTE: MAX_READ_NUM is the max read num one thread cand send to one machine
#define MAX_READ_NUM (2 * GROUP_SIZE)
#define MAX_VEC_LEN (2560)
// #define MAX_QUERYBUFFER_SIZE (32768)
// #define MAX_QUERYBUFFER_SIZE (8192)
#define MAX_QUERYBUFFER_SIZE (24576)
// #define MAX_QUERYBUFFER_SIZE (2000)
#define MAX_SEND_SGE (1)
#define MAX_RECV_SGE (1)
#define MAX_SGE_NUM (16)
// WARN: this may not enough on large datasets.
#define MAX_CACHE_VEC (2000)
// #define MAX_RECV_NUM (512)  // for task push
#define MAX_WRITE_NUM (512)
#define MAX_RECV_NUM (64)
#define RELEASE_BLOCK (MAX_WRITE_NUM >> 2)
#define READ_MARK_OFF (25)
#define MAX_WC_NUM (64)
#define LOW_16BIT_MASK (0x0000ffff)

#define OFF (0)
#define ON (1)
#define DEF_PORT (18516)
#define DEF_IB_PORT (1)
#define LINK_UNSPEC (-2)
#define DEF_ITERS (1000)
#define DEF_GID_INDEX (-1)
#define DEF_INLINE (0)
#define MIN_RNR_TIMER (12)
#define DEF_QP_TIME (14)
#define DEF_CQ_MOD (100)
#define MSG_SIZE_CQ_MOD_LIMIT (8192)
#define DISABLED_CQ_MOD_VALUE (1)
#define MAX_SIZE (8388608)
#define DEF_INLINE_WRITE (220)
#define LAT_MEASURE_TAIL (2)
#define DEF_CACHE_LINE_SIZE (64)
#define SUCCESS (0)
#define FAILURE (1)

/* Have AES XTS support */
#define HAVE_AES_XTS 1

/* Enable CUDA feature */
/* #undef HAVE_CUDA */

/* Enable CUDA DMABUF feature */
/* #undef HAVE_CUDA_DMABUF */

/* Have DCS support */
#define HAVE_DCS 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Enable endian conversion */
#define HAVE_ENDIAN 1

/* Have EX support */
#define HAVE_EX 1

/* Have Extended ODP support */
#define HAVE_EX_ODP 1

/* Have a way to check gid type */
#define HAVE_GID_TYPE 1

/* API GID compatibility */
#define HAVE_GID_TYPE_DECLARED 1

/* Define to 1 if you have the <hip/hip_runtime_api.h> header file. */
/* #undef HAVE_HIP_HIP_RUNTIME_API_H */

/* Define to 1 if you have the <hip/hip_version.h> header file. */
/* #undef HAVE_HIP_HIP_VERSION_H */

/* Enable Habana Labs benchmarks */
/* #undef HAVE_HL */

/* Define to 1 if you have the <hlthunk.h> header file. */
/* #undef HAVE_HLTHUNK_H */

/* Have new post send API support */
#define HAVE_IBV_WR_API 1

/* Define to 1 if you have the <infiniband/verbs.h> header file. */
#define HAVE_INFINIBAND_VERBS_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Enable IPv4 Extended Flow Specification */
#define HAVE_IPV4_EXT 1

/* Enable IPv6 Flow Specification */
#define HAVE_IPV6 1

/* Define to 1 if you have the `ibverbs' library (-libverbs). */
#define HAVE_LIBIBVERBS 1

/* Define to 1 if you have the `rdmacm' library (-lrdmacm). */
#define HAVE_LIBRDMACM 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <misc/habanalabs.h> header file. */
/* #undef HAVE_MISC_HABANALABS_H */

/* Have Direct Verbs support */
#define HAVE_MLX5DV 1

/* Have MLX5 DEVX support */
#define HAVE_MLX5_DEVX 1

/* Enable Neuron benchmarks */
/* #undef HAVE_NEURON */

/* Enable Neuron DMA buffers */
/* #undef HAVE_NEURON_DMABUF */

/* Define to 1 if you have the <nrt/nrt.h> header file. */
/* #undef HAVE_NRT_NRT_H */

/* Have Out of order data placement support */
/* #undef HAVE_OOO_ATTR */

/* Have PACKET_PACING support */
#define HAVE_PACKET_PACING 1

/* Define to 1 if you have the <pci/pci.h> header file. */
#define HAVE_PCI_PCI_H 1

/* Enable RAW_ETH_TEST */
#define HAVE_RAW_ETH 1

/* Enable RAW_ETH_TEST_REG */
#define HAVE_RAW_ETH_REG 1

/* Enable HAVE_REG_DMABUF_MR */
#define HAVE_REG_DMABUF_MR 1

/* Enable Relaxed Ordering */
#define HAVE_RO 1

/* Enable ROCm */
/* #undef HAVE_ROCM */

/* Enable SCIF link Layer */
/* #undef HAVE_SCIF */

/* Enable Sniffer Flow Specification */
#define HAVE_SNIFFER 1

/* Have SRD support */
/* #undef HAVE_SRD */

/* Have SRD with RDMA read support */
/* #undef HAVE_SRD_WITH_RDMA_READ */

/* Have SRD with RDMA write support */
/* #undef HAVE_SRD_WITH_RDMA_WRITE */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <synapse_api.h> header file. */
/* #undef HAVE_SYNAPSE_API_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Enable XRCD feature */
#define HAVE_XRCD 1

/* OS is FreeBSD */
/* #undef IS_FREEBSD */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "perftest"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "linux-rdma@vger.kernel.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "perftest"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "perftest 6.22"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "perftest"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "6.22"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "6.22"

/* Enable ROCm */
/* #undef __HIP_PLATFORM_AMD__ */
