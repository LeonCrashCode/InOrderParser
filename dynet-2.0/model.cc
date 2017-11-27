#include "dynet/model.h"
#include "dynet/tensor.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/dynet.h"
#include "dynet/param-init.h"
#include "dynet/io.h"

#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#define LOAD_INIT_FUNC() initialize_lookups()

#ifdef __CUDACC__
#include "dynet/gpu-ops.h"
#endif

// Macros for defining functions over parameters
// NOTE: This only works on the default device, as parameters are currently defined over default devices
#ifdef __CUDACC__
#define DYNET_PARAMNORM_INST_DEV_IMPL(MyParam, regular_func, dev_func) \
  template void MyParam::dev_func<Device_GPU>(Device_GPU & dev, float *sqnorm) const;
#elif defined(HAVE_CUDA)
#define DYNET_PARAMNORM_INST_DEV_IMPL(MyParam, regular_func, dev_func) \
  extern template void MyParam::dev_func<Device_GPU>(Device_GPU & dev, float *sqnorm) const; \
  template void MyParam::dev_func<Device_CPU>(Device_CPU & dev, float *sqnorm) const; \
  void MyParam::regular_func(float *sqnorm) const { \
    if(default_device->type == DeviceType::CPU) { dev_func(*(Device_CPU*)default_device,sqnorm); } \
    else if(default_device->type == DeviceType::GPU) { dev_func(*(Device_GPU*)default_device,sqnorm); } \
    else { throw std::runtime_error("Invalid device type in MyParam::dev_func"); } \
  }
#else
#define DYNET_PARAMNORM_INST_DEV_IMPL(MyParam, regular_func, dev_func) \
  template void MyParam::dev_func<Device_CPU>(Device_CPU & dev, float *sqnorm) const; \
  void MyParam::regular_func(float *sqnorm) const { \
    if(default_device->type == DeviceType::CPU) { dev_func(*(Device_CPU*)default_device,sqnorm); } \
    else { throw std::runtime_error("Invalid device type in MyParam::dev_func"); } \
  }
#endif

using namespace std;

namespace dynet {

// CPU only functions
#ifndef __CUDACC__

ParameterStorageBase::~ParameterStorageBase() {}

ParameterStorage::ParameterStorage(const Dim& d, float scale, const std::string & name) : name(name), dim(d), updated(true), nonzero_grad(false), owner(nullptr) {
  values.d = g.d = d;
  values.device = g.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, values);
  default_device->allocate_tensor(DeviceMempool::PS, g);
  TensorTools::zero(g);
  if (scale == 0.0f) {
    ParameterInitGlorot init;
    init.initialize_params(values);
  } else {
    ParameterInitUniform init(scale);
    init.initialize_params(values);
  }
}

ParameterStorage::ParameterStorage(const Dim& d, const ParameterInit & init, const std::string & name) : name(name), dim(d), updated(true), nonzero_grad(false), owner(nullptr) {
  values.d = g.d = d;
  values.device = g.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, values);
  default_device->allocate_tensor(DeviceMempool::PS, g);
  TensorTools::zero(g);
  init.initialize_params(values);
}

size_t ParameterStorage::size() const { return dim.size(); }

void ParameterStorage::zero() {
  TensorTools::zero(values);
  clear();
}

void ParameterStorage::copy(const ParameterStorage & param) {
  DYNET_ARG_CHECK(dim == param.dim,
                  "Attempt to copy between parameters with mismatched dimensions: " << dim << " != " << param.dim);
  TensorTools::copy_elements(values, param.values);
}

void ParameterStorage::clear() {
  nonzero_grad = false;
  if (g.v != nullptr)
    TensorTools::zero(g);
}

void ParameterStorage::clip(float left, float right) {
  TensorTools::clip(values, left, right);
}

bool valid_parameter(const std::string & s) {
  auto it = std::find_if(s.begin(), s.end(), [] (char ch) { return ch == '/' || ch == '_'; });
  return it == s.end();
}

LookupParameterStorage::LookupParameterStorage(unsigned n, const Dim& d, const ParameterInit & init, const std::string & name) : name(name), dim(d), updated(true), all_updated(false), nonzero_grad(false), owner(nullptr) {
  all_dim = dim; all_dim.d[all_dim.nd++] = n;
  all_grads.d = all_values.d = all_dim;
  all_grads.device = all_values.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, all_values);
  default_device->allocate_tensor(DeviceMempool::PS, all_grads);
  init.initialize_params(all_values);
  initialize_lookups();
}

void LookupParameterStorage::initialize_lookups() {
  int num = all_dim[all_dim.nd - 1];
  dim = all_dim; dim.nd--;
  int dim_size = dim.size();
  if (values.size() == 0) {
    values.resize(num);
    for (int i = 0; i < num; ++i)
      values[i] = Tensor(dim, all_values.v + i * dim_size, all_values.device, all_values.mem_pool);
  }
  if (grads.size() == 0 && all_grads.v != nullptr) {
    grads.resize(num);
    for (int i = 0; i < num; ++i)
      grads[i] = Tensor(dim, all_grads.v + i * dim_size, all_grads.device, all_grads.mem_pool);
  }
}

void LookupParameterStorage::zero() {
  TensorTools::zero(all_values);
}

size_t LookupParameterStorage::size() const {
  return all_dim.size();
}

void LookupParameterStorage::copy(const LookupParameterStorage& param) {
  if (all_dim != param.all_dim)
    DYNET_INVALID_ARG("Attempt to copy between lookup parameters with mismatched dimensions: " << all_dim << " != " << param.all_dim);
  TensorTools::copy_elements(all_values, param.all_values);
}

void LookupParameterStorage::clear() {
  // TODO: the GPU part is hacky, probably need a better heuristic
  if (all_grads.device->type == DeviceType::GPU || all_updated) {
    TensorTools::zero(all_grads);
  } else {
    for (auto i : non_zero_grads)
      TensorTools::zero(grads[i]);
  }
  non_zero_grads.clear();
  all_updated = false;
  nonzero_grad = false;
}

Parameter::Parameter() : p(nullptr) {}

Parameter::Parameter(ParameterStorage* p) : p(p) {}

ParameterStorage& Parameter::get_storage() const {
  DYNET_ASSERT(p != nullptr, "Attempt to get pointer for null parameter");
  return *p;
}

void Parameter::zero() {
  get_storage().zero();
}

string Parameter::get_fullname() const {
  DYNET_ASSERT(p != nullptr, "Attempt to get pointer for null parameter");
  return p->name;
}

void Parameter::clip_inplace(float left, float right){
  float my_scale = 1./ current_weight_decay();
  get_storage().clip(left * my_scale, right * my_scale);
}

void Parameter::set_updated(bool b) {
  get_storage().updated = b;
}

bool Parameter::is_updated() {
  return get_storage().updated;
}

float Parameter::current_weight_decay() const {
  return get_storage().owner->get_weight_decay().current_weight_decay();
}

LookupParameter::LookupParameter() : p(nullptr) { }

LookupParameter::LookupParameter(LookupParameterStorage* p) : p(p) {}

LookupParameterStorage& LookupParameter::get_storage() const {
  DYNET_ASSERT(p != nullptr, "Attempt to get pointer for null LookupParameter");
  return *p;
}

void LookupParameter::zero() {
  get_storage().zero();
}

void LookupParameter::initialize(unsigned index, const std::vector<float>& val) const {
  get_storage().initialize(index, val);
}

string LookupParameter::get_fullname() const {
  DYNET_ASSERT(p != nullptr, "Attempt to get pointer for null parameter");
  return p->name;
}

void LookupParameter::set_updated(bool b) {
  get_storage().updated = b;
}
bool LookupParameter::is_updated() {
  return get_storage().updated;
}

float LookupParameter::current_weight_decay() const {
  return get_storage().owner->get_weight_decay().current_weight_decay();
}

ParameterCollectionStorage::ParameterCollectionStorage() : gradient_norm_scratch(nullptr) {
  weight_decay.set_lambda(weight_decay_lambda);
}

ParameterCollectionStorage::~ParameterCollectionStorage() {
  for (auto p : all_params) delete p;
  if (gradient_norm_scratch)
    default_device->mem->free(gradient_norm_scratch);
}

void ParameterCollectionStorage::project_weights(float radius) {
  static float* project_scratch = 0;
  if (!project_scratch)
    project_scratch = (float*)default_device->mem->malloc(all_params.size() * sizeof(float));
  int pi = 0;
  for (auto p : all_params) {
    p->squared_l2norm(&project_scratch[pi]);
    ++pi;
  }
  double gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += project_scratch[i];
  cerr << "NORM: " << sqrt(gg) << endl;
}

ParameterCollection::ParameterCollection() : name("/"), storage(new ParameterCollectionStorage), parent(nullptr) { }

ParameterCollection::ParameterCollection(const string & my_name, ParameterCollection* my_parent) :
    name(my_name), storage(new ParameterCollectionStorage), parent(my_parent) { }

ParameterCollection ParameterCollection::add_subcollection(const string & sub_name) {
  if (valid_parameter(sub_name)) {
    ostringstream oss; oss << name << sub_name;
    int idx = collec_name_cntr[sub_name]++;
    if (idx > 0 || sub_name.size() == 0) oss << "_" << idx;
    oss << "/";
    return ParameterCollection(oss.str(), this);
  } else {
    throw std::runtime_error("Submodel name could not include '/' and '_'");
  }
}

ParameterCollection::~ParameterCollection() {
  if(parent == nullptr && storage != nullptr)
    delete storage;
}

void ParameterCollection::set_weight_decay_lambda(float lambda) {
  get_storage().weight_decay.set_lambda(lambda);
}

void ParameterCollection::project_weights(float radius) {
  get_storage().project_weights(radius);
}

Parameter ParameterCollection::add_parameters(const Dim & d, const std::string & p_name) {
  return add_parameters(d, ParameterInitGlorot(), p_name);
}

Parameter ParameterCollection::add_parameters(const Dim& d, float scale, const std::string & p_name) {
  if(scale == 0.0f)
    return add_parameters(d, ParameterInitGlorot(), p_name);
  else
    return add_parameters(d, ParameterInitUniform(scale), p_name);
}

Parameter ParameterCollection::add_parameters(const Dim& d, const ParameterInit & init, const std::string & p_name) {
  if (valid_parameter(p_name)) {
    ostringstream oss; oss << name << p_name;
    int idx = name_cntr[p_name]++;
    if (idx > 0 || p_name.size() == 0) oss << "_" << idx;

    ParameterStorage* p = new ParameterStorage(d, init, oss.str());
    add_parameters_to_storage(p);
    return Parameter(p);
  } else {
    throw std::runtime_error("Parameter name could not include '/' and '_'");
  }
}

void ParameterCollection::add_parameters_to_storage(ParameterStorage *p) {
  if(parent != nullptr)
    parent->add_parameters_to_storage(p);
  else
    p->owner = this;
  if(storage != nullptr) {
    storage->all_params.push_back(p);
    storage->params.push_back(p);
  }
}

std::vector<ParameterStorageBase*> ParameterCollection::get_parameter_storages_base() const {
  std::vector<ParameterStorageBase*> all_params;
  ParameterCollection *t = const_cast<ParameterCollection*>(this);
  while (t->parent != nullptr) { t = t->parent; }
  auto all_ps = t->get_storage().all_params;
  auto ps = t->get_storage().params;
  auto lps = t->get_storage().lookup_params;
  size_t i = 0, j = 0;
  for (size_t k = 0; k < all_ps.size(); ++k) {
    if (i < ps.size() && all_ps[k] == ps[i]) {
      if (ps[i]->name.find(name) == 0) {
        all_params.push_back(all_ps[k]);
      }
      ++i;
    } else {
      if (lps[j]->name.find(name) == 0) {
        all_params.push_back(all_ps[k]);
      }
      ++ j;
    }
  }
  return all_params;
}

ParameterStorage* ParameterCollection::get_parameter_storage(const std::string & pname) {
  if (pname.find(name) == 0) {
    ParameterCollection *t = this;
    while (t->parent != nullptr) { t = t->parent; }
    for (auto & param : t->get_storage().params) {
      if (param->name == pname) {
        return param;
      }
    }
  }
  std::string errMsg = "No existing parameter " + pname + " found in " + name;
  throw std::runtime_error(errMsg);
}

std::vector<ParameterStorage*> ParameterCollection::get_parameter_storages() const {
  std::vector<ParameterStorage*> params;
  ParameterCollection *t = const_cast<ParameterCollection*>(this);
  while (t->parent != nullptr) { t = t->parent; }
  for (auto & param : t->get_storage().params) {
    if (param->name.find(name) == 0) {
      params.push_back(param);
    }
  }
  return params;
}

LookupParameter ParameterCollection::add_lookup_parameters(unsigned n, const Dim& d, const std::string & p_name) {
  return add_lookup_parameters(n, d, ParameterInitGlorot(true), p_name);
}

LookupParameter ParameterCollection::add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init, const std::string & p_name) {
  if (valid_parameter(p_name)) {
    ostringstream oss; oss << name << p_name;
    int idx = name_cntr[p_name]++;
    if (idx > 0 || p_name.size() == 0) oss << "_" << idx;

    LookupParameterStorage* p = new LookupParameterStorage(n, d, init, oss.str());
    add_lookup_parameters_to_storage(p);
    return LookupParameter(p);
  } else {
    throw std::runtime_error("LookupParameter name could not include '/' and '_'");
  }
}

void ParameterCollection::add_lookup_parameters_to_storage(LookupParameterStorage *p) {
  if(parent != nullptr)
    parent->add_lookup_parameters_to_storage(p);
  else
    p->owner = this;
  if(storage != nullptr) {
    storage->all_params.push_back(p);
    storage->lookup_params.push_back(p);
  }
}

LookupParameterStorage* ParameterCollection::get_lookup_parameter_storage(const std::string & lookup_pname)
{
  if (lookup_pname.find(name) == 0) {
    ParameterCollection *t = this;
    while (t->parent != nullptr) { t = t->parent; }
    for (auto & lookup_param : t->get_storage().lookup_params) {
      if (lookup_param->name == lookup_pname) {
        return lookup_param;
      }
    }
  }
  std::string errMsg = "No existing parameter " + lookup_pname + " found in " + name;
  throw std::runtime_error(errMsg);
}

std::vector<LookupParameterStorage*>
ParameterCollection::get_lookup_parameter_storages() const {
  std::vector<LookupParameterStorage*> lookup_params;
  ParameterCollection *t = const_cast<ParameterCollection*>(this);
  while (t->parent != nullptr) { t = t->parent; }
  for (auto & lookup_param: t->get_storage().lookup_params) {
    if (lookup_param->name.find(name) == 0) {
      lookup_params.push_back(lookup_param); 
    }
  }
  return lookup_params;
}

void ParameterCollection::reset_gradient() {
  for (auto p : get_storage().params) { p->clear(); }
  for (auto p : get_storage().lookup_params) { p->clear(); }
}

size_t ParameterCollection::parameter_count() const {
  size_t r = 0;
  for (const ParameterStorageBase* param : get_storage().all_params)
    r += param->size();
  return r;
}

size_t ParameterCollection::updated_parameter_count() const {
  size_t r = 0;
  for (const ParameterStorageBase* param : get_storage().all_params)
    if(param->is_updated())
      r += param->size();
  return r;
}

ParameterCollectionStorage& ParameterCollection::get_storage() {
  if(storage == nullptr) {
    if (parent == nullptr)
      storage = new ParameterCollectionStorage;
    else
      DYNET_RUNTIME_ERR("ParameterCollection::get_storage() not implemented yet for subsets");
  }
  return *storage;
}

const ParameterCollectionStorage& ParameterCollection::get_storage() const {
  if(storage == nullptr) {
    if (parent == nullptr)
      const_cast<ParameterCollectionStorage*&>(storage) = new ParameterCollectionStorage;
    else
      DYNET_RUNTIME_ERR("ParameterCollection::get_storage() not implemented yet for subsets");
  }
  return *storage;
}

void save_dynet_model(std::string filename, ParameterCollection* model) {
  TextFileSaver saver(filename);
  saver.save(*model, "model");
};

void load_dynet_model(std::string filename, ParameterCollection* model) {
  TextFileLoader loader(filename);
  loader.populate(*model, "model");
};

Model::Model() : ParameterCollection() {
  cerr << "The name dynet::Model has been deprecated and replaced by dynet::ParameterCollection." << endl
       << "Please replace references to dynet::Model with references to dynet::ParameterCollection." << endl;
}

#endif

// CPU/GPU code
// TODO: It's a bit annoying to re-implement the CPU/GPU control code for each
//       function, but it's not clear how to handle heterogeneous functions w/
//       macros

// Note: Using DeviceMempool::NONE here because these tensors are not persistent
// and won't be saved so it doesn't matter which mempool they belong to.

// Take the squared norm
template <class MyDevice>
void ParameterStorage::squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  sqnorm_t.t<0>().device(*dev.edevice) = values.tvec().square().sum();
}
DYNET_PARAMNORM_INST_DEV_IMPL(ParameterStorage, squared_l2norm, squared_l2norm_dev)

// Take the squared norm of the gradient
template <class MyDevice>
void ParameterStorage::g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  DYNET_ASSERT(g.v != nullptr, "Cannot take norm of gradient with null parameter");
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  sqnorm_t.t<0>().device(*dev.edevice) = g.tvec().square().sum();
}
DYNET_PARAMNORM_INST_DEV_IMPL(ParameterStorage, g_squared_l2norm, g_squared_l2norm_dev)

template <class MyDevice>
void ParameterStorage::accumulate_grad_dev(MyDevice & dev, const Tensor& d) {
  g.tvec().device(*dev.edevice) += d.tvec();
}
#ifdef __CUDACC__
template void ParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
#elif defined(HAVE_CUDA)
extern template void ParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
template void ParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void ParameterStorage::accumulate_grad(const Tensor& d) {
  nonzero_grad = true;
  if (values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values.device, d); }
  else if (values.device->type == DeviceType::GPU) { accumulate_grad_dev(*(Device_GPU*)values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void ParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void ParameterStorage::accumulate_grad(const Tensor& d) {
  nonzero_grad = true;
  if (values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void ParameterStorage::scale_parameters_dev(MyDevice & dev, float a) {
  values.tvec().device(*dev.edevice) = values.tvec() * a;
}
#ifdef __CUDACC__
template void ParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void ParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
template void ParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_parameters(float a) {
  if (values.device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values.device, a); }
  else if (values.device->type == DeviceType::GPU) { scale_parameters_dev(*(Device_GPU*)values.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void ParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_parameters(float a) {
  if (values.device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void ParameterStorage::scale_gradient_dev(MyDevice & dev, float a) {
  g.tvec().device(*dev.edevice) = g.tvec() * a;
}
#ifdef __CUDACC__
template void ParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void ParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
template void ParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_gradient(float a) {
  if (g.device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)g.device, a); }
  else if (g.device->type == DeviceType::GPU) { scale_gradient_dev(*(Device_GPU*)g.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void ParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_gradient(float a) {
  if (g.device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)g.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::initialize_dev(MyDevice & dev, unsigned index, const vector<float>& val) {
  DYNET_ARG_CHECK(int(val.size()) == int(dim.size()),
                  "Attempt to initialize LookupParameters with vector of wrong size "
                  "(" << val.size() << " != " << dim.size() << ")");
#ifdef __CUDACC__
  cudaMemcpyAsync(values[index].v, &val[0], val.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
#endif
}
#ifdef __CUDACC__
template void LookupParameterStorage::initialize_dev<Device_GPU>(Device_GPU & dev, unsigned index, const vector<float>& val);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::initialize_dev<Device_GPU>(Device_GPU & dev, unsigned index, const vector<float>& val);
template void LookupParameterStorage::initialize_dev<Device_CPU>(Device_CPU & dev, unsigned index, const vector<float>& val);
void LookupParameterStorage::initialize(unsigned index, const vector<float>& val) {
  if (values[index].device->type == DeviceType::CPU) { initialize_dev(*(Device_CPU*)values[index].device, index, val); }
  else if (values[index].device->type == DeviceType::GPU) { initialize_dev(*(Device_GPU*)values[index].device, index, val); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::initialize_dev<Device_CPU>(Device_CPU & dev, unsigned index, const vector<float>& val);
void LookupParameterStorage::initialize(unsigned index, const vector<float>& val) {
  if (values[index].device->type == DeviceType::CPU) { initialize_dev(*(Device_CPU*)values[index].device, index, val); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  sqnorm_t.t<0>().device(*dev.edevice) = all_values.tvec().square().sum();
}
DYNET_PARAMNORM_INST_DEV_IMPL(LookupParameterStorage, squared_l2norm, squared_l2norm_dev)

template <class MyDevice>
void LookupParameterStorage::g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  TensorTools::zero(sqnorm_t);
  // TODO: the GPU part is hacky, probably need a better heuristic
  if (all_grads.device->type == DeviceType::GPU || all_updated) {
    sqnorm_t.t<0>().device(*dev.edevice) += all_grads.tvec().square().sum();
  } else {
    auto it = non_zero_grads.begin();
    while (it != non_zero_grads.end())
      sqnorm_t.t<0>().device(*dev.edevice) += grads[*(it++)].tvec().square().sum();
  }
}
DYNET_PARAMNORM_INST_DEV_IMPL(LookupParameterStorage, g_squared_l2norm, g_squared_l2norm_dev)

template <class MyDevice>
void LookupParameterStorage::accumulate_grad_dev(MyDevice & dev, const Tensor& d) {
  all_updated = true;
  all_grads.tvec().device(*dev.edevice) += d.tvec();
}
#ifdef __CUDACC__
template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void LookupParameterStorage::accumulate_grad(const Tensor& d) {
  nonzero_grad = true;
  if (all_values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)all_values.device, d); }
  else if (all_values.device->type == DeviceType::GPU) { accumulate_grad_dev(*(Device_GPU*)all_values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void LookupParameterStorage::accumulate_grad(const Tensor& d) {
  nonzero_grad = true;
  if (all_values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)all_values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::accumulate_grad_dev(MyDevice & dev, unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
  grads[index].tvec().device(*dev.edevice) += d.tvec();
}
#ifdef __CUDACC__
template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, unsigned index, const Tensor& d);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, unsigned index, const Tensor& d);
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, unsigned index, const Tensor& d);
void LookupParameterStorage::accumulate_grad(unsigned index, const Tensor& d) {
  nonzero_grad = true;
  if (values[index].device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values[index].device, index, d); }
  else if (values[index].device->type == DeviceType::GPU) { accumulate_grad_dev(*(Device_GPU*)values[index].device, index, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, unsigned index, const Tensor& d);
void LookupParameterStorage::accumulate_grad(unsigned index, const Tensor& d) {
  nonzero_grad = true;
  if (values[index].device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values[index].device, index, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::accumulate_grads_dev(MyDevice & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g) {
#ifdef __CUDACC__
  for (unsigned i = 0; i < n; ++i)
    non_zero_grads.insert(ids_host[i]);
  dynet::gpu::dense_to_sparse_block_add(n, ids_dev, dim.size(), g, all_grads.v);
#else
  size_t gsize = dim.size();
  Tensor gt(dim, g, all_grads.device, all_grads.mem_pool);
  for (unsigned i = 0; i < n; ++i) {
    non_zero_grads.insert(ids_host[i]);
    grads[ids_host[i]].tvec().device(*dev.edevice) += gt.tvec();
    gt.v += gsize;
  }
#endif
}
#ifdef __CUDACC__
template void LookupParameterStorage::accumulate_grads_dev<Device_GPU>(Device_GPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::accumulate_grads_dev<Device_GPU>(Device_GPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
template void LookupParameterStorage::accumulate_grads_dev<Device_CPU>(Device_CPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
void LookupParameterStorage::accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g) {
  if (all_values.device->type == DeviceType::CPU) { accumulate_grads_dev(*(Device_CPU*)all_values.device, n, ids_host, ids_dev, g); }
  else if (all_values.device->type == DeviceType::GPU) { accumulate_grads_dev(*(Device_GPU*)all_values.device, n, ids_host, ids_dev, g); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::accumulate_grads_dev<Device_CPU>(Device_CPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
void LookupParameterStorage::accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g) {
  if (all_values.device->type == DeviceType::CPU) { accumulate_grads_dev(*(Device_CPU*)all_values.device, n, ids_host, ids_dev, g); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::scale_parameters_dev(MyDevice & dev, float a) {
  all_values.tvec().device(*dev.edevice) = all_values.tvec() * a;
}
#ifdef __CUDACC__
template void LookupParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
template void LookupParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_parameters(float a) {
  if (values[0].device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values[0].device, a); }
  else if (values[0].device->type == DeviceType::GPU) { scale_parameters_dev(*(Device_GPU*)values[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_parameters(float a) {
  if (values[0].device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::scale_gradient_dev(MyDevice & dev, float a) {
  all_grads.tvec().device(*dev.edevice) = all_grads.tvec() * a;
}
#ifdef __CUDACC__
template void LookupParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
template void LookupParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_gradient(float a) {
  if (grads[0].device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)grads[0].device, a); }
  else if (grads[0].device->type == DeviceType::GPU) { scale_gradient_dev(*(Device_GPU*)grads[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_gradient(float a) {
  if (grads[0].device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)grads[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
float ParameterCollectionStorage::gradient_l2_norm_dev(MyDevice & dev) const {
  if (gradient_norm_scratch == nullptr)
    gradient_norm_scratch = (float*)default_device->mem->malloc((all_params.size() + 1) * sizeof(float));
  size_t pi;
  for (pi = 0; pi < all_params.size(); ++pi)
    all_params[pi]->g_squared_l2norm(&gradient_norm_scratch[pi]);
  Tensor scratch_t({(unsigned int)all_params.size()}, gradient_norm_scratch, &dev, DeviceMempool::NONE);
  Tensor sum_t({1}, gradient_norm_scratch + pi, &dev, DeviceMempool::NONE);
  sum_t.t<0>().device(*dev.edevice) = scratch_t.t<1>().sum().sqrt();
#ifdef __CUDACC__
  float res = 0;
  cudaMemcpy(&res, gradient_norm_scratch + pi, sizeof(float),  cudaMemcpyDeviceToHost);
  return res;
#else
  return gradient_norm_scratch[pi];
#endif
}

#ifdef __CUDACC__
template float ParameterCollectionStorage::gradient_l2_norm_dev<Device_GPU>(Device_GPU & dev) const;
#elif defined(HAVE_CUDA)
extern template float ParameterCollectionStorage::gradient_l2_norm_dev<Device_GPU>(Device_GPU & dev) const;
template float ParameterCollectionStorage::gradient_l2_norm_dev<Device_CPU>(Device_CPU & dev) const;
float ParameterCollectionStorage::gradient_l2_norm() const {
  if (default_device->type == DeviceType::CPU) { return gradient_l2_norm_dev(*(Device_CPU*)default_device); }
  else if (default_device->type == DeviceType::GPU) { return gradient_l2_norm_dev(*(Device_GPU*)default_device); }
  else { throw std::runtime_error("Bad device type"); }
}
float ParameterCollection::gradient_l2_norm() const {
  return get_storage().gradient_l2_norm();
}
#else
template float ParameterCollectionStorage::gradient_l2_norm_dev<Device_CPU>(Device_CPU & dev) const;
float ParameterCollectionStorage::gradient_l2_norm() const {
  if (default_device->type == DeviceType::CPU) { return gradient_l2_norm_dev(*(Device_CPU*)default_device); }
  else { throw std::runtime_error("Bad device type"); }
}
float ParameterCollection::gradient_l2_norm() const {
  return get_storage().gradient_l2_norm();
}
#endif

} // namespace dynet
