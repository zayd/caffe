// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/math_backends.hpp"

namespace caffe {
template<typename Dtype>
void GPUMathBackend<Dtype>::gemm(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		Dtype* C) {
	caffe_gpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const Dtype alpha, const Dtype* A, const Dtype* x,
		const Dtype beta, Dtype* y) {
	caffe_gpu_gemv<Dtype>(TransA, M, N, alpha, A, x, beta, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::axpy(const int N, const Dtype alpha, const Dtype* X,
		Dtype* Y) {
	caffe_gpu_axpy<Dtype>(N, alpha, X, Y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::axpby(const int N, const Dtype alpha, const Dtype* X,
		const Dtype beta, Dtype* Y) {
	caffe_gpu_axpby<Dtype>(N, alpha, X, beta, Y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::copy(const int N, const Dtype *X, Dtype *Y) {
	caffe_gpu_copy<Dtype>(N, X, Y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
	caffe_gpu_set<Dtype>(N, alpha, X);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::add_scalar(const int N, const Dtype alpha, Dtype *X) {
	caffe_gpu_add_scalar<Dtype>(N, alpha, X);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::scal(const int N, const Dtype alpha, Dtype *X) {
	caffe_gpu_scal<Dtype>(N, alpha, X);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::sqr(const int N, const Dtype* a, Dtype* y) {
	NOT_IMPLEMENTED;
//	caffe_gpu_sqr<Dtype>(N, a, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::add(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	NOT_IMPLEMENTED;
//	caffe_gpu_add<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::sub(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	NOT_IMPLEMENTED;
//	caffe_gpu_sub<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::mul(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	caffe_gpu_mul<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::div(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	caffe_gpu_div<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
		Dtype* y) {
	caffe_gpu_powx<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::rng_uniform(const int N, const Dtype a, const Dtype b,
		Dtype* r) {
	caffe_gpu_rng_uniform<Dtype>(N, a, b, r);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::rng_gaussian(const int N, const Dtype mu,
		const Dtype sigma, Dtype* r) {
	caffe_gpu_rng_gaussian<Dtype>(N, mu, sigma, r);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
	caffe_gpu_rng_bernoulli<Dtype>(N, p, r);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::exp(const int N, const Dtype* a, Dtype* y) {
	NOT_IMPLEMENTED;
//	caffe_gpu_exp<Dtype>(N, a, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::dot(const int N, const Dtype* x, const Dtype* y,
		Dtype* out) {
	caffe_gpu_dot<Dtype>(N, x, y, out);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::hamming_distance(const int N, const Dtype* x,
		const Dtype* y, uint32_t* out) {
	*out = caffe_gpu_hamming_distance<Dtype>(N, x, y);
}

template<typename Dtype>
// Returns the sum of the absolute values of the elements of vector x
void GPUMathBackend<Dtype>::asum(const int N, const Dtype* x, Dtype* y) {
	caffe_gpu_asum<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::sign(const int N, const Dtype* x, Dtype* y) {
	caffe_gpu_sign<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::sgnbit(const int N, const Dtype* x, Dtype* y) {
	caffe_gpu_sgnbit<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::fabs(const int N, const Dtype* x, Dtype* y) {
	caffe_gpu_fabs<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUMathBackend<Dtype>::scale(const int N, const Dtype alpha, const Dtype *x,
		Dtype* y) {
	caffe_gpu_scale<Dtype>(N, alpha, x, y);
}

INSTANTIATE_CLASS(GPUMathBackend);

}  // namespace caffe
