#include "dynet/nodes-linalg.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* Transpose *************

#ifndef __CUDACC__

string Transpose::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "transpose("<< arg_names[0] << ", ";
  for(size_t i = 0; i < dims.size(); ++i)
    s << (i == 0?'{':',') << dims[i];
  s << "})";
  return s.str();
}

Dim Transpose::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Bad arguments to Transpose: " << xs);
  DYNET_ARG_CHECK(xs[0].nd == dims.size() || xs[0].num_nonone_dims() == 1, "Dimensions passed to transpose (" << dims.size() << ") must be equal to dimensions in input tensor (" << xs[0].nd << ')');
  Dim ret(xs[0]);
  ret.nd = dims.size();
  for(size_t i = 0; i < dims.size(); ++i)
    ret.d[i] = xs[0][dims[i]];
  return ret;
}

#endif

template<class MyDevice>
void Transpose::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (dim.num_nonone_dims() <= 1) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  } else {
    Eigen::array<ptrdiff_t, 5> order;
    for(size_t i = 0; i < 5; ++i)
      order[i] = (i >= dims.size() ? i : dims[i]);
    fx.tb<4>().device(*dev.edevice) = xs[0]->tb<4>().shuffle(order);
  }
}

template<class MyDevice>
void Transpose::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::array<ptrdiff_t, 5> order;
  for(size_t i = 0; i < 5; ++i)
    order[(i >= dims.size() ? i : dims[i])] = i;
  dEdxi.tb<4>().device(*dev.edevice) += dEdf.tb<4>().shuffle(order);
}
DYNET_NODE_INST_DEV_IMPL(Transpose)

// ************* MatrixInverse *************

#ifndef __CUDACC__

string MatrixInverse::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inverse(" << arg_names[0] << ")";
  return s.str();
}

Dim MatrixInverse::dim_forward(const vector<Dim>& xs) const {
  return xs[0];
}

#endif

template<class MyDevice>
void MatrixInverse::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MatrixInverse::forward");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("MatrixInverse not yet implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = *fx;
  y = x.inverse();
#endif
  // TODO: Change into tensors after resolving test errors
  // fx.t<2>().device(*dev.edevice) = xs[0]->t<2>().inverse();
}

template<class MyDevice>
void MatrixInverse::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MatrixInverse::backward");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("MatrixInverse not yet implemented for CUDA");
#else
  auto d = *dEdf;
  auto y = *fx;
  (*dEdxi) -= y * d * y;
#endif
}
DYNET_NODE_INST_DEV_IMPL(MatrixInverse)

// ************* LogDet *************

#ifndef __CUDACC__

string LogDet::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "logdet(" << arg_names[0] << ")";
  return s.str();
}

Dim LogDet::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs[0].ndims() <= 2 && (xs[0].rows() == xs[0].cols()), "Bad arguments in LogDet: " << xs);
  return Dim({1});
}

// set use_cholesky if M is symmetric - it's faster and more stable
// for dep parsing it won't be
template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;
  if (use_cholesky) {
    LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
    auto& U = chol.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
  } else {
    PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
    auto& LU = lu.matrixLU();
    Scalar c = lu.permutationP().determinant(); // -1 or 1
    for (unsigned i = 0; i < LU.rows(); ++i) {
      const auto& lii = LU(i,i);
      if (lii < Scalar(0)) c *= -1;
      ld += log(abs(lii));
    }
    ld += log(c);
  }
  return ld;
}

#endif

template<class MyDevice>
void LogDet::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("LogDet not implemented for CUDA");
#else
  fx.v[0] = logdet(**xs[0], false);
#endif
}

template<class MyDevice>
void LogDet::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("KMHNGram not implemented for CUDA");
#else
  auto trans = (**xs[0]).transpose();
  (*dEdxi) += (dEdf.v[0]) * trans.inverse();
#endif
}
DYNET_NODE_INST_DEV_IMPL(LogDet)

// ************* TraceOfProduct *************

#ifndef __CUDACC__

string TraceOfProduct::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "Tr(" << arg_names[0] << " * " << arg_names[1] << "^T)";
  return s.str();
}

Dim TraceOfProduct::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 && xs[0] == xs[1], "Bad arguments in TraceOfProduct: " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

#endif

template<class MyDevice>
void TraceOfProduct::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("TraceOfProduct not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 * x2.transpose()).trace();
#endif
}

template<class MyDevice>
void TraceOfProduct::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i < 2, "Failed dimension check in TraceOfProduce::backward");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("TraceOfProduct not yet implemented for CUDA");
#else
  const float d = dEdf.v[0];
  auto xother = **xs[1 - i];
  *dEdxi += d * xother;
#endif
}
DYNET_NODE_INST_DEV_IMPL(TraceOfProduct)

}
