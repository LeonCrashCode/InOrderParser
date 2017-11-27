#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dynet/io.h"

#include "impl/oracle.h"
#include "impl/pretrained.h"

#include "impl/cl-args.h"

// dictionaries
dynet::Dict termdict, ntermdict, adict, posdict;
volatile bool requested_stop = false;

unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
unsigned POS_SIZE = 0;

std::map<int,int> action2NTindex;  // pass in index of action PJ(X), return index of X

Params params;

using namespace dynet;
using namespace std;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training

struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // pretrained word embeddings (not updated)
  LookupParameter p_nt; // nonterminal embeddings
  LookupParameter p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameter p_a; // input action embeddings
  LookupParameter p_pos; // pos embeddings (optional)
  Parameter p_p2w;  // pos2word mapping (optional)
  Parameter p_ptbias; // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameter p_ptW;    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameter p_pbias; // parser state bias
  Parameter p_A; // action lstm to parser state
  Parameter p_B; // buffer lstm to parser state
  Parameter p_S; // stack lstm to parser state
  Parameter p_w2l; // word to LSTM input
  Parameter p_t2l; // pretrained word embeddings to LSTM input
  Parameter p_ib; // LSTM input bias
  Parameter p_cbias; // composition function bias
  Parameter p_p2a;   // parser state to action
  Parameter p_action_start;  // action bias
  Parameter p_abias;  // action bias
  Parameter p_buffer_guard;  // end of buffer
  Parameter p_stack_guard;  // end of stack

  Parameter p_cW;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(params.layers, params.lstm_input_dim, params.hidden_dim, *model),
      action_lstm(params.layers, params.action_dim, params.hidden_dim, *model),
      const_lstm_fwd(params.layers, params.lstm_input_dim, params.lstm_input_dim, *model), // used to compose children of a node into a representation of the node
      const_lstm_rev(params.layers, params.lstm_input_dim, params.lstm_input_dim, *model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {params.lstm_input_dim})),
      p_ntup(model->add_lookup_parameters(NT_SIZE, {params.lstm_input_dim})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {params.action_dim})),
      p_pbias(model->add_parameters({params.hidden_dim})),
      p_A(model->add_parameters({params.hidden_dim, params.hidden_dim})),
      p_B(model->add_parameters({params.hidden_dim, params.hidden_dim})),
      p_S(model->add_parameters({params.hidden_dim, params.hidden_dim})),
      p_w2l(model->add_parameters({params.lstm_input_dim, params.input_dim})),
      p_ib(model->add_parameters({params.lstm_input_dim})),
      p_cbias(model->add_parameters({params.lstm_input_dim})),
      p_p2a(model->add_parameters({ACTION_SIZE, params.hidden_dim})),
      p_action_start(model->add_parameters({params.action_dim})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_buffer_guard(model->add_parameters({params.lstm_input_dim})),
      p_stack_guard(model->add_parameters({params.lstm_input_dim})),

      p_cW(model->add_parameters({params.lstm_input_dim, params.lstm_input_dim * 2})) {
    if (params.use_pos_tags) {
      p_pos = model->add_lookup_parameters(POS_SIZE, {params.pos_dim});
      p_p2w = model->add_parameters({params.lstm_input_dim, params.pos_dim});
    }
    buffer_lstm = new LSTMBuilder(params.layers, params.lstm_input_dim, params.hidden_dim, *model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {params.pretrained_dim});
      for (auto it : pretrained)
        p_t.initialize(it.first, it.second);
      p_t2l = model->add_parameters({params.lstm_input_dim, params.pretrained_dim});
    }
  }

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Discriminative(const string& a, char prev_a, unsigned bsize, unsigned ssize, unsigned nopen_parens, unsigned unary) {
  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = (a[0] == 'R' && a[1]=='E');
  bool is_nt = (a[0] == 'P');
  bool is_term = (a[0] == 'T');
  assert(is_shift || is_reduce || is_nt || is_term) ;
  static const unsigned MAX_OPEN_NTS = 100;
  static const unsigned MAX_UNARY = 3;
//  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
  if (is_term){
    if(ssize == 2 && bsize == 1 && prev_a == 'R') return false;
    return true;
  }

  if(ssize == 1){
     if(!is_shift) return true;
     return false;
  }

  if (is_shift){
    if(bsize == 1) return true;
    if(nopen_parens == 0) return true;
    return false;
  }

  if (is_nt) {
    if(bsize == 1 && unary >= MAX_UNARY) return true;
    if(prev_a == 'P') return true;
    return false; 
  }

  if (is_reduce){
    if(unary > MAX_UNARY) return true;
    if(nopen_parens == 0) return true;
    return false;
  }

  // TODO should we control the depth of the parse in some way? i.e., as long as there
  // are items in the buffer, we can do an NT operation, which could cause trouble
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
Expression log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
		     vector<unsigned> *results,
                     bool is_evaluation,
                     bool sample = false) {
if(params.debug) cerr << "sent size: " << sent.size()<<"\n";
    const bool build_training_graph = correct_actions.size() > 0;
    bool apply_dropout = (params.dropout && !is_evaluation);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    buffer_lstm->new_graph(*hg);
if(params.debug) cerr<<"apply dropout: "<< apply_dropout<<" "<<params.dropout<<"\n";
    if (apply_dropout) {
      stack_lstm.set_dropout(params.dropout);
      action_lstm.set_dropout(params.dropout);
      buffer_lstm->set_dropout(params.dropout);
      const_lstm_fwd.set_dropout(params.dropout);
      const_lstm_rev.set_dropout(params.dropout);
    } else {
      stack_lstm.disable_dropout();
      action_lstm.disable_dropout();
      buffer_lstm->disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }
    stack_lstm.start_new_sequence();
    buffer_lstm->start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression p2w;
    if (params.use_pos_tags) {
      p2w = parameter(*hg, p_p2w);
    }
    Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    Expression w2l = parameter(*hg, p_w2l);
    Expression t2l;
    if (pretrained.size() > 0)
      t2l = parameter(*hg, p_t2l);

    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right
if(params.debug) cerr<<"Graph Init\n";
    // in the discriminative model, here we set up the buffer contents
    for (unsigned i = 0; i < sent.size(); ++i) {
        int wordid = sent.raw[i]; // this will be equal to unk at dev/test
        if (build_training_graph && singletons.size() > wordid && singletons[wordid] && rand01() > 0.5)
          wordid = sent.unk[i];
        Expression w = lookup(*hg, p_w, wordid);
        vector<Expression> args = {ib, w2l, w}; // learn embeddings
        if (pretrained.size()>0 && pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
          Expression t = const_lookup(*hg, p_t, sent.lc[i]);
          args.push_back(t2l);
          args.push_back(t);
        }
        if (params.use_pos_tags) {
          args.push_back(p2w);
          args.push_back(lookup(*hg, p_pos, sent.pos[i]));
        }
        buffer[sent.size() - i] = rectify(affine_transform(args));
        bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    
    for (auto& b : buffer)
      buffer_lstm->add_input(b);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    unsigned unary = 0;
    int nopen_parens = 0;
    char prev_a = '0';
//    while(stack.size() > 2 || buffer.size() > 1) {
    while(true){
	if(prev_a == 'T') break;
      // get list of possible actions for the current parser state
if(params.debug) cerr<< "action_count " << action_count <<"\n";
      current_valid_actions.clear();
if(params.debug) cerr<< "unary: " << unary << "nopen_parens: "<<nopen_parens<<"\n";
      for (auto a: possible_actions) {
        if (IsActionForbidden_Discriminative(adict.convert(a), prev_a, buffer.size(), stack.size(), nopen_parens, unary))
          continue;
        current_valid_actions.push_back(a);
      }
if(params.debug){
	cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
		cerr<<adict.convert(current_valid_actions[i])<<" ";
	}
	cerr <<"\n";

	unsigned j = 999;
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
                if(current_valid_actions[i] == correct_actions[action_count]) {j = i; break;}
        }
	if(j == 999){
		cerr<<"gold out\n";
		exit(1);
	}

}
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = action_lstm.back();
      Expression buffer_summary = buffer_lstm->back();

      if (apply_dropout) {
        stack_summary = dropout(stack_summary, params.dropout);
        action_summary = dropout(action_summary, params.dropout);
        buffer_summary = dropout(buffer_summary, params.dropout);
      }
      Expression p_t = affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
      Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});
      if (sample && params.alpha != 1.0f) r_t = r_t * params.alpha;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward(adiste));
      double best_score = adist[current_valid_actions[0]];
      unsigned model_action = current_valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        model_action = current_valid_actions[w];
      } else { // max
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[current_valid_actions[i]] > best_score) {
            best_score = adist[current_valid_actions[i]];
            model_action = current_valid_actions[i];
          }
        }
      }
      unsigned action = model_action;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        if (model_action == action) { (*right)++; }
      } else {
        //cerr << "Chosen action: " << adict.convert(action) << endl;
      }
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.convert(i) << ':' << adist[i]; }
      //cerr << endl;
      ++action_count;
      log_probs.push_back(pick(adiste, action));
      if(results) results->push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // do action
      const string& actionString=adict.convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];
if(params.debug){
      
      cerr << "MODEL_ACT: " << adict.convert(model_action)<<" ";
      cerr <<"GOLD_ACT: " << actionString<<"\n";
}

if(params.debug) {
        cerr <<"stacki: ";
        for(unsigned i = 0; i < stacki.size(); i ++){
                cerr<<stacki[i]<<" ";
        }
        cerr<<"\n";

        cerr<<"is_open_paren: ";
        for(unsigned i = 0; i < is_open_paren.size(); i ++){
                cerr<<is_open_paren[i]<<" ";
        }
        cerr<<"\n";

}

      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
          stack.push_back(buffer.back());
          stack_lstm.add_input(buffer.back());
          stacki.push_back(bufferi.back());
          buffer.pop_back();
          buffer_lstm->rewind_one_step();
          bufferi.pop_back();
          is_open_paren.push_back(-1);
	unary = 0;
      } else if (ac == 'P') { // PJ
        ++nopen_parens;
        assert(stack.size() > 1);
        auto it = action2NTindex.find(action);
        assert(it != action2NTindex.end());
        int nt_index = it->second;
        nt_count++;
        Expression nt_embedding = lookup(*hg, p_nt, nt_index);
        stack.push_back(nt_embedding);
        stack_lstm.add_input(nt_embedding);
        stacki.push_back(-1);
        is_open_paren.push_back(nt_index);
      } else if (ac == 'R'){ // REDUCE
        --nopen_parens;
	if(prev_a == 'P') unary += 1;
	if(prev_a == 'R') unary = 0;
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        // find what paren we are closing
        int i = is_open_paren.size() - 1;
        while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
        Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
        int nchildren = is_open_paren.size() - i - 1;
        assert(nchildren+1 > 0);
if(params.debug) cerr << "  number of children to reduce: " << nchildren << endl;
        vector<Expression> children(nchildren);
        const_lstm_fwd.start_new_sequence();
        const_lstm_rev.start_new_sequence();

        // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
        // TO BE COMPOSED INTO A TREE EMBEDDING
        for (i = 0; i < nchildren; ++i) {
          children[i] = stack.back();
          stacki.pop_back();
          stack.pop_back();
          stack_lstm.rewind_one_step();
          is_open_paren.pop_back();
        }

        is_open_paren.pop_back(); // nt symbol
        assert (stacki.back() == -1);
        stacki.pop_back(); // nonterminal dummy
        stack.pop_back(); // nonterminal dummy
        stack_lstm.rewind_one_step(); // nt symbol 

	children.push_back(stack.back()); // leftmost
	stacki.pop_back(); //leftmost
	stack.pop_back(); //leftmost
	stack_lstm.rewind_one_step(); //leftmost
	is_open_paren.pop_back(); //leftmost

	nchildren ++;
        // BUILD TREE EMBEDDING USING BIDIR LSTM
        const_lstm_fwd.add_input(nonterminal);
        const_lstm_rev.add_input(nonterminal);
        for (i = 0; i < nchildren; ++i) {
          const_lstm_fwd.add_input(children[i]);
          const_lstm_rev.add_input(children[nchildren - i - 1]);
        }
        Expression cfwd = const_lstm_fwd.back();
        Expression crev = const_lstm_rev.back();
        if (apply_dropout) {
          cfwd = dropout(cfwd, params.dropout);
          crev = dropout(crev, params.dropout);
        }
        Expression c = concatenate({cfwd, crev});
        Expression composed = rectify(affine_transform({cbias, cW, c}));
        stack_lstm.add_input(composed);
        stack.push_back(composed);
        stacki.push_back(999); // who knows, should get rid of this
        is_open_paren.push_back(-1); // we just closed a paren at this position
      }
      else{// TERMINATE
      }
      prev_a = ac;
    }
    if (build_training_graph && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;
  }




struct ParserState {
  LSTMBuilder stack_lstm;
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  vector<Expression> buffer;
  vector<int> bufferi;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;

  vector<Expression> stack;
  vector<int> stacki;
  vector<unsigned> results;  // sequence of predicted actions
  bool complete;
  vector<Expression> log_probs;
  double score;
  int action_count;
  int nopen_parens;
  char prev_a;
};


struct ParserStateCompare {
  bool operator()(const ParserState& a, const ParserState& b) const {
    return a.score > b.score;
  }
};

static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
}


};
void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  get_args(argc, argv, params);
  
  ostringstream os;
  os << "ntparse"
     << (params.use_pos_tags ? "_pos" : "")
     << '_' << params.layers
     << '_' << params.input_dim
     << '_' << params.hidden_dim
     << '_' << params.action_dim
     << '_' << params.lstm_input_dim
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;

  Model model;

  parser::KOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::KOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::KOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(params.training_data, true);	
  corpus.load_bdata(params.bracketing_dev_data);

  if (params.words != "")
    parser::ReadEmbeddings_word2vec(params.words, &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.freeze();
  termdict.set_unk("UNK"); // we don't actually expect to use this often
     // since the Oracles are required to be "pre-UNKified", but this prevents
     // problems with UNKifying the lowercased data which needs to be loaded
  adict.freeze();
  ntermdict.freeze();
  posdict.freeze();

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  if (params.dev_data != "") {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(params.dev_data, false);
  }
  if (params.test_data != "") {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(params.test_data, false);
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.convert(i);
    if (a[0] != 'P') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  ParserBuilder parser(&model, pretrained);
  if (params.model != "") {
    TextFileLoader loader(params.model);
    loader.populate(model);
  }

  //TRAINING
  if (params.train) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(model);
    sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_err = 9e99;
    double bestf1=0.0;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
           ComputationGraph hg;
           Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right,NULL,false);
           double lp = as_scalar(hg.incremental_forward(nll));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward(nll);
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        ofstream out("dev.act");
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right, NULL, true);
              double lp = as_scalar(hg.incremental_forward(nll));
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred;
	   parser.log_prob_parser(&hg,sentence,vector<int>(),&right, &pred, true);
           unsigned ti = 0;
           for (auto a : pred) {
                out << adict.convert(a);
                if(adict.convert(a) == "SHIFT"){
                        out<<" " << posdict.convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
                        ti++;
                }
                out<<endl;
           }
           out << endl;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

	std::string command_1="python mid2tree.py dev.act > dev.eval" ;
	const char* cmd_1=command_1.c_str();
	cerr<<system(cmd_1)<<"\n";

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" dev.eval > dev.evalout";
        const char* cmd2=command2.c_str();

        system(cmd2);
        
        std::ifstream evalfile("dev.evalout");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        while (getline(evalfile, lineS) && !newfmeasure){
		if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
			//std::cout<<lineS<<"\n";
			strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;     // alias of size_t

		        newfmeasure = std::stod (strfmeasure,&sz);
			//std::cout<<strfmeasure<<"\n";
		}
        }
        
 
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << newfmeasure << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
//        if (err < best_dev_err && (tot_seen / corpus.size()) > 1.0) {
       if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;
	  ostringstream part_os;
  	  part_os << "ntparse"
     	      << (params.use_pos_tags ? "_pos" : "")
              << '_' << params.layers
              << '_' << params.input_dim
              << '_' << params.hidden_dim
              << '_' << params.action_dim
              << '_' << params.lstm_input_dim
              << "-pid" << getpid() 
	      << "-part" << (tot_seen/corpus.size()) << ".params";
 	  
	  const string part = part_os.str();
 	  TextFileSaver saver("model/"+part);
          saver.save(model);
          system("cp dev.eval dev.eval.best");
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          /*if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }*/
        }
      }
    }
  } // should do training?
  if (test_corpus.size() > 0) { // do test evaluation
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
	const vector<int> actions;
	if(params.samples > 0){
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           dwords += sentence.size();
           for (unsigned z = 0; z < params.samples; ++z) {
             ComputationGraph hg;
             vector<unsigned> pred;
	     parser.log_prob_parser(&hg,sentence,actions,&right,&pred,true,true);
             int ti = 0;
             for (auto a : pred) {
             	cout << adict.convert(a);
		if (adict.convert(a) == "SHIFT"){
			cout<<" "<<posdict.convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
			ti++;
		}
		cout << endl;
	     }
             cout << endl;
           }
         }
         }
        ofstream out("test.act");
        t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           const vector<int>& actions=test_corpus.actions[sii];
/*	   for(unsigned i = 0; i < sentence.size(); i ++){
	   	out << termdict.convert(sentence.raw[i])<<" ";
	   }
	   out<<"||| ";
           for(unsigned i = 0; i < sentence.size(); i ++){
                out << termdict.convert(sentence.lc[i])<<" ";
           }
	   out<<"||| ";
	   for(unsigned i = 0; i < sentence.size(); i ++){
                out << posdict.convert(sentence.pos[i])<<" ";
           }
	   out<<"\n";*/
           dwords += sentence.size();
           {  ComputationGraph hg;
              Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right,NULL,true);
              double lp = as_scalar(hg.incremental_forward(nll));
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred;
	   parser.log_prob_parser(&hg,sentence,vector<int>(),&right,&pred,true);
           unsigned ti = 0;
           for (auto a : pred) {
           	out << adict.convert(a);
		if(adict.convert(a) == "SHIFT"){
			out<<" " << posdict.convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
			ti++;
		}
		out<<endl;
	   }
           out << endl;
           trs += actions.size();
        }
        
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

        std::string command_1="python mid2tree.py test.act > test.eval" ;
        const char* cmd_1=command_1.c_str();
        system(cmd_1);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" test.eval > test.evalout";
        const char* cmd2=command2.c_str();

        system(cmd2);

        std::ifstream evalfile("test.evalout");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
                if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                        //std::cout<<lineS<<"\n";
                        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;
                        newfmeasure = std::stod (strfmeasure,&sz);
                        //std::cout<<strfmeasure<<"\n";
                }
        }

       cerr<<"F1score: "<<newfmeasure<<"\n";
  }
}
