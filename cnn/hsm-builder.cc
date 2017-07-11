#include "cnn/hsm-builder.h"

#include <fstream>
#include <iostream>
#include <cassert>
#include <sstream>

#undef assert
#define assert(x) {}

using namespace std;

namespace cnn {

using namespace expr;

Cluster::Cluster() : initialized(false) {}
void Cluster::new_graph(ComputationGraph& cg) {
  for (Cluster* child : children) {
    child->new_graph(cg);
  }
  bias.pg = NULL;
  weights.pg = NULL;
}

Cluster* Cluster::add_child(unsigned sym) {
  assert (!initialized);
  assert (terminals.size() == 0);
  auto it = word2ind.find(sym);
  unsigned i;
  if (it == word2ind.end()) {
    Cluster* c = new Cluster();
    c->path = path;
    c->path.push_back(sym);
    i = children.size();
    word2ind.insert(make_pair(sym, i));
    children.push_back(c);
    assert (c != NULL);
  }
  else {
    i = it->second;
  }
  return children[i];
}

void Cluster::add_word(unsigned word) {
  assert (!initialized);
  assert (children.size() == 0);
  word2ind[word] = terminals.size();
  terminals.push_back(word);
}

void Cluster::initialize(unsigned rep_dim, Model* model) {
  assert (!initialized);
  output_size = (children.size() > 0) ? children.size() : terminals.size();
  assert (output_size > 0);

  if (output_size == 1) {
    p_weights = NULL;
    p_bias = NULL;
  }
  else if (output_size == 2) {
    p_weights = model->add_parameters({1, rep_dim});
    p_bias = model->add_parameters({1});
  }
  else {
    p_weights = model->add_parameters({output_size, rep_dim});
    p_bias = model->add_parameters({output_size});
  }

  for (Cluster* child : children) {
    child->initialize(rep_dim, model);
  }
}

unsigned Cluster::num_children() const {
  return children.size();
}

const Cluster* Cluster::get_child(unsigned i) const {
  assert (i < children.size());
  assert (children[i] != NULL);
  return children[i];
}

const vector<unsigned>& Cluster::get_path() const { return path; }
unsigned Cluster::get_index(unsigned word) const { return word2ind.find(word)->second; }
unsigned Cluster::get_word(unsigned index) const { return terminals[index]; }

Expression Cluster::predict(Expression h, ComputationGraph& cg) const {
  if (output_size == 1) {
    return input(cg, 1.0f);
  }
  else {
    Expression b = get_bias(cg);
    Expression w = get_weights(cg);
    return affine_transform({b, w, h});
  }
}

Expression Cluster::neg_log_softmax(Expression h, unsigned r, ComputationGraph& cg) const {
  if (output_size == 1) {
    return input(cg, 0.0f);
  }
  else if (output_size == 2) {
    Expression p = logistic(predict(h, cg));
    assert (r == 0 || r == 1);
    if (r == 1) {
      p = 1 - p;
    }
    return -log(p);
  }
  else {
    Expression dist = predict(h, cg);
    return pickneglogsoftmax(dist, r);
  }
}

unsigned Cluster::sample(expr::Expression h, ComputationGraph& cg) const {
  if (output_size == 1) {
    return 0;
  }
  else if (output_size == 2) {
    logistic(predict(h, cg));
    double prob0 = as_scalar(cg.incremental_forward());
    double p = rand01();
    if (p < prob0) {
      return 0;
    }
    else {
      return 1;
    }
  }
  else {
    softmax(predict(h, cg));
    vector<float> dist = as_vector(cg.incremental_forward());
    unsigned c = 0;
    double p = rand01();
    for (; c < dist.size(); ++c) {
      p -= dist[c];
      if (p < 0.0) { break; }
    }
    if (c == dist.size()) {
      --c;
    }
    return c;
  }
}

Expression Cluster::get_weights(ComputationGraph& cg) const {
  if (weights.pg != &cg) {
    weights = parameter(cg, p_weights);
  }
  return weights;
}

Expression Cluster::get_bias(ComputationGraph& cg) const {
  if (bias.pg != &cg) {
    bias = parameter(cg, p_bias);
  }
  return bias;
}

string Cluster::toString() const {
  stringstream ss;
  for (unsigned i = 0; i < path.size(); ++i) {
    if (i != 0) {
      ss << " ";
    }
    ss << path[i];
  }
  return ss.str();
}

HierarchicalSoftmaxBuilder::HierarchicalSoftmaxBuilder(unsigned rep_dim,
                             const std::string& cluster_file,
                             Dict* word_dict,
                             Model* model) {
  root = ReadClusterFile(cluster_file, word_dict);
  root->initialize(rep_dim, model);
}

HierarchicalSoftmaxBuilder::~HierarchicalSoftmaxBuilder() {
}

void HierarchicalSoftmaxBuilder::new_graph(ComputationGraph& cg) {
  pcg = &cg;
  root->new_graph(cg);
}

Expression HierarchicalSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned wordidx) {
  assert (pcg != NULL && "You must call new_graph before calling neg_log_softmax!");
  Cluster* path = widx2path[wordidx];

  unsigned i = 0;
  const Cluster* node = root;
  assert (root != NULL);
  vector<Expression> log_probs;
  Expression lp;
  unsigned r;
  while (node->num_children() > 0) {
    r = node->get_index(path->get_path()[i]);
    lp = node->neg_log_softmax(rep, r, *pcg);
    log_probs.push_back(lp);
    node = node->get_child(r);
    assert (node != NULL);
    i += 1;
  }

  r = path->get_index(wordidx);
  lp = node->neg_log_softmax(rep, r, *pcg);
  log_probs.push_back(lp);

  return sum(log_probs);
}

unsigned HierarchicalSoftmaxBuilder::sample(const expr::Expression& rep) {
  assert (pcg != NULL && "You must call new_graph before calling sample!");

  const Cluster* node = root;
  vector<float> dist;
  unsigned c;
  while (node->num_children() > 0) {
    c = node->sample(rep, *pcg);
    node = node->get_child(c);
  }

  c = node->sample(rep, *pcg);
  return node->get_word(c);
}

inline bool is_ws(char x) { return (x == ' ' || x == '\t'); }
inline bool not_ws(char x) { return (x != ' ' && x != '\t'); }

Cluster* HierarchicalSoftmaxBuilder::ReadClusterFile(const std::string& cluster_file, Dict* word_dict) {
  cerr << "Reading clusters from " << cluster_file << " ...\n";
  ifstream in(cluster_file);
  assert(in);
  int wc = 0;
  string line;
  vector<unsigned> path;
  Cluster* root = new Cluster();
  while(getline(in, line)) {
    path.clear();
    ++wc;
    const unsigned len = line.size();
    unsigned startp = 0;
    unsigned endp = 0;
    while (startp < len) {
      while (is_ws(line[startp]) && startp < len) { ++startp; }
      endp = startp;
      while (not_ws(line[endp]) && endp < len) { ++endp; }
      string symbol = line.substr(startp, endp - startp);
      path.push_back(path_symbols.Convert(symbol));
      if (line[endp] == ' ') {
        startp = endp + 1;
        continue;
      }
      else {
        break;
      }
    }
    Cluster* node = root;
    for (unsigned symbol : path) {
      node = node->add_child(symbol);
    }

    unsigned startw = endp;
    while (is_ws(line[startw]) && startw < len) { ++startw; }
    unsigned endw = startw;
    while (not_ws(line[endw]) && endw < len) { ++endw; }
    assert(endp > startp);
    assert(startw > endp);
    assert(endw > startw);

    string word = line.substr(startw, endw - startw);
    unsigned widx = word_dict->Convert(word);
    node->add_word(widx);

    if (widx2path.size() <= widx) {
      widx2path.resize(widx + 1);
    }
    widx2path[widx] = node;
  }
  cerr << "Done reading clusters.\n";
  return root;
}

} // namespace cnn
