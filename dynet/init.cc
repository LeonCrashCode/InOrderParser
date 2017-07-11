#include "dynet/init.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/dynet.h"
#include "dynet/weight-decay.h"

#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include "dynet/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace dynet {

static void remove_args(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  assert(argc >= 0);
}

DynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters) {
  DynetParams params;
  params.shared_parameters = shared_parameters;

  int argi = 1;

#if HAVE_CUDA
  params.gpu_mask = std::vector<int>(MAX_GPUS, 0);
#endif


  while (argi < argc) {
    string arg = argv[argi];

    // Memory
    if (arg == "--dynet-mem" || arg == "--dynet_mem") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-mem expects an argument (the memory, in megabytes, to reserve)\n";
        abort();
      } else {
        params.mem_descriptor = argv[argi + 1];
        remove_args(argc, argv, argi, 2);
      }
    }

    // Weight decay
    else if (arg == "--dynet-l2" || arg == "--dynet_l2") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-l2 requires an argument (the weight decay per update)\n";
        abort();
      } else {
        string a2 = argv[argi + 1];
        istringstream d(a2); d >> params.weight_decay;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Random seed
    else if (arg == "--dynet-seed" || arg == "--dynet_seed") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-seed expects an argument (the random number seed)\n";
        abort();
      } else {
        string a2 = argv[argi + 1];
        istringstream c(a2); c >> params.random_seed;
        remove_args(argc, argv, argi, 2);
      }
    }

#if HAVE_CUDA
    // Number of GPUs
    else if (arg == "--dynet_gpus" || arg == "--dynet-gpus") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-gpus expects an argument (number of GPUs to use)\n";
        abort();
      } else {
        if (params.ngpus_requested) {
          cerr << "Multiple instances of --dynet-gpus" << endl; abort();
        }
        params.ngpus_requested = true;
        string a2 = argv[argi + 1];
        istringstream c(a2); c >> params.requested_gpus;
        remove_args(argc, argv, argi, 2);
      }
    }

    // GPU ids
    else if (arg == "--dynet_gpu_ids" || arg == "--dynet-gpu-ids") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-gpu-ids expects an argument (comma separated list of physical GPU ids to use)\n";
        abort();
      } else {
        string a2 = argv[argi + 1];
        if (params.ids_requested) {
          cerr << "Multiple instances of --dynet-gpu-ids" << endl; abort();
        }
        params.ids_requested = true;
        if (a2.size() % 2 != 1) {
          cerr << "Bad argument to --dynet-gpu-ids: " << a2 << endl; abort();
        }
        for (unsigned i = 0; i < a2.size(); ++i) {
          if ((i % 2 == 0 && (a2[i] < '0' || a2[i] > '9')) ||
              (i % 2 == 1 && a2[i] != ',')) {
            cerr << "Bad argument to --dynet-gpu-ids: " << a2 << endl; abort();
          }
          if (i % 2 == 0) {
            int gpu_id = a2[i] - '0';
            if (gpu_id >= MAX_GPUS) { cerr << "Raise MAX_GPUS\n"; abort(); }
            params.gpu_mask[gpu_id]++;
            params.requested_gpus++;
            if (params.gpu_mask[gpu_id] != 1) {
              cerr << "Bad argument to --dynet-gpu-ids: " << a2 << endl; abort();
            }
          }
        }
        remove_args(argc, argv, argi, 2);
      }
    }
#endif

    // Go to next argument
    else {
      argi++;
    }


  }

#if HAVE_CUDA
  // Check for conflict between the two ways of requesting GPUs
  if (params.ids_requested && params.ngpus_requested) {
    cerr << "Use only --dynet_gpus or --dynet_gpu_ids, not both\n";
    abort();
  }
#endif

  return params;
}

void initialize(DynetParams params) {
  if (default_device != nullptr) {
    cerr << "WARNING: Attempting to initialize dynet twice. Ignoring duplicate initialization." << endl;
    return;
  }

  // initialize CUDA
  vector<Device*> gpudevices;
#if HAVE_CUDA
  cerr << "[dynet] initializing CUDA\n";
  gpudevices = initialize_gpu(params);
#endif

  // Set random seed
  if (params.random_seed == 0) {
    random_device rd;
    params.random_seed = rd();
  }
  cerr << "[dynet] random seed: " << params.random_seed << endl;
  rndeng = new mt19937(params.random_seed);

  // Set weight decay rate
  if (params.weight_decay < 0 || params.weight_decay >= 1) {
    cerr << "[dynet] weight decay parameter must be between 0 and 1 (probably very small like 1e-6)\n";
    abort();
  }
  weight_decay_lambda = params.weight_decay;

  // Allocate memory
  cerr << "[dynet] allocating memory: " << params.mem_descriptor << "MB\n";
  devices.push_back(new Device_CPU(devices.size(), params.mem_descriptor, params.shared_parameters));
  int default_index = 0;
  if (gpudevices.size() > 0) {
    for (auto gpu : gpudevices)
      devices.push_back(gpu);
    default_index++;
  }
  default_device = devices[default_index];

  // TODO these should be accessed through the relevant device and removed here
  kSCALAR_MINUSONE = default_device->kSCALAR_MINUSONE;
  kSCALAR_ONE = default_device->kSCALAR_ONE;
  kSCALAR_ZERO = default_device->kSCALAR_ZERO;
  cerr << "[dynet] memory allocation done.\n";

}

void initialize(int& argc, char**& argv, bool shared_parameters) {
  DynetParams params = extract_dynet_params(argc, argv, shared_parameters);
  initialize(params);
}

void cleanup() {
  delete rndeng;
  // TODO: Devices cannot be deleted at the moment
  // for(Device* device : devices) delete device;
  devices.clear();
  default_device = nullptr;
}

} // namespace dynet

