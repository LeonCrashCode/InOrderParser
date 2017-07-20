/**
 * \file cl-args.h
 * \brief This is a **very** minimal command line argument parser
 */
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

/**
 * Values used to specify the task at hand, and incidentally the required command line arguments
 */
enum Task {
  TRAIN, /**< Self-supervised learning : Only requires train and dev file */
  TRAIN_SUP, /**< Supervised learning : Requires train and dev data as well as labels */
  TEST
};

using namespace std;
/**
 * \brief Structure holding any possible command line argument
 *
 */

struct Params {
  string training_data;
  string dev_data;
  string bracketing_dev_data;
  string test_data;
  string words;

  float dropout;
  unsigned samples;
  float alpha;
  string model;
  bool use_pos_tags=false;
  
  unsigned layers = 2;
  unsigned action_dim = 16;
  unsigned input_dim = 32;
  unsigned pretrained_dim = 50;
  unsigned pos_dim = 12;
  unsigned hidden_dim = 64;
  unsigned lstm_input_dim = 60;

  bool debug = false;
  bool train = false;

};

/**
 * \brief Get parameters from command line arguments
 * \details Parses parameters from `argv` and check for required fields depending on the task
 * 
 * \param argc Number of arguments
 * \param argv Arguments strings
 * \param params Params structure
 * \param task Task
 */
void get_args(int argc,
              char** argv,
              Params& params) {
  int i = 0;
  while (i < argc) {
    string arg = argv[i];
    if (arg == "--training_data") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.training_data;
      i++;
    } else if (arg == "--dev_data") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dev_data;
      i++;
    } else if (arg == "--test_data") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.test_data;
      i++;
    } else if (arg == "--bracketing_dev_data") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.bracketing_dev_data;
      i++;
    } else if (arg == "--words") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.words;
      i++;
    } else if (arg == "--model") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.model;
      i++;
    } else if (arg == "--layers") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.layers;
      i++;
    } else if (arg == "--action_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.action_dim;
      i++;
    } else if (arg == "--input_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.input_dim;
      i++;
    } else if (arg == "--pretrained_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.pretrained_dim;
      i++;
    } else if (arg == "--pos_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.pos_dim;
      i++;
    } else if (arg == "--lstm_input_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.lstm_input_dim;
      i++;
    } else if (arg == "--hidden_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.hidden_dim;
      i++;
    } else if (arg == "--use_pos_tags") {
      params.use_pos_tags = true;
    } else if (arg == "--train") {
      params.train = true;
    } else if (arg == "--debug") {
      params.debug = true;
    } else  if (arg == "--dropout") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dropout;
      i++;
    } else  if (arg == "--alpha") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.alpha;
      i++;
    } else  if (arg == "--samples") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.samples;
      i++;
    }
    i++;
  }
}
