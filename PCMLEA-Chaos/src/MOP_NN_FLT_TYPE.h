#ifndef __MOP_FLT_TYPE_
#define __MOP_FLT_TYPE_

#define MY_FLT_TYPE double
//#define CHECK_FINITE !_finite
#define CHECK_INVALID(x) (isinf(x)||isnan(x))

//#define UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY

#endif
