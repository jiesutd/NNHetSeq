#ifndef NEWUTILTENSOR
#define NEWUTILTENSOR

#include "tensor.h"
#include "MyLib.h"

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace nr;



template<typename xpu>
inline void assign_with_extend(Tensor<xpu, 2, dtype> w, const NRMat<dtype>& wnr, const int& extend_size) {
	int dim1 = wnr.nrows();
	int dim2 = wnr.ncols();
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			w[idx][idy] = wnr[idx][idy];
		}
	}
	for (int idx = dim1; idx < dim1+extend_size; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			w[idx][idy] = wnr[0][idy];
		}
	}
}

#endif
