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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "impl/oracle.h"
#include "impl/pretrained.h"
#include "impl/compressed-fstream.h"
#include "impl/eval.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict;
bool DEBUG;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 32;
unsigned HIDDEN_DIM = 64;
unsigned ACTION_DIM = 16;
unsigned PRETRAINED_DIM = 100;
unsigned LSTM_INPUT_DIM = 128;
unsigned POS_DIM = 12;
float ALPHA = 1.f;
unsigned N_SAMPLES = 1;

unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
unsigned POS_SIZE = 0;

std::map<int,int> action2NTindex;  // pass in index of action PJ(X), return index of X
std::map<std::string,int> train_dict;
using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("samples,s", po::value<unsigned>(), "Sample N trees for each test sentence instead of greedy max decoding")
        ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
        ("model_dir,m", po::value<string>(), "Load saved model from this file")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
	("train_dict", po::value<string>(), "training lexical dictionary")
	("lang", po::value<string>(), "language setting")
	("debug","debug")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
}

struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_nt; // nonterminal embeddings
  LookupParameters* p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_pos; // pos embeddings (optional)
  Parameters* p_p2w;  // pos2word mapping (optional)
  Parameters* p_ptbias; // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_ptW;    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack

  Parameters* p_cW;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_ntup(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),

      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})) {
      
      p_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2w = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    
      buffer_lstm = new LSTMBuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model);
      
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained) p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
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
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const vector<int> raw,
		     const vector<int> lc,
		     const vector<int> pos,
                     bool sample = false) {
if(DEBUG) cerr << "sent size: " << raw.size()<<"\n";
    vector<unsigned> results;
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    stack_lstm.start_new_sequence();
    buffer_lstm->new_graph(*hg);
    buffer_lstm->start_new_sequence();
    action_lstm.start_new_sequence();
      
    stack_lstm.disable_dropout();
    action_lstm.disable_dropout();
    buffer_lstm->disable_dropout();
    const_lstm_fwd.disable_dropout();
    const_lstm_rev.disable_dropout();

    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression p2w;  
    p2w = parameter(*hg, p_p2w);

    Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    Expression w2l = parameter(*hg, p_w2l);
    Expression t2l;
      
    t2l = parameter(*hg, p_t2l);
    
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(raw.size() + 1);  // variables representing word embeddings
    vector<int> bufferi(raw.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    for (unsigned i = 0; i < raw.size(); ++i) {
        int wordid = raw[i]; // this will be equal to unk at dev/test
        Expression w = lookup(*hg, p_w, wordid);
        vector<Expression> args = {ib, w2l, w}; // learn embeddings
        if (pretrained.count(lc[i])) {  // include fixed pretrained vectors?
          Expression t = const_lookup(*hg, p_t, lc[i]);
          args.push_back(t2l);
          args.push_back(t);
        }
        args.push_back(p2w);
        args.push_back(lookup(*hg, p_pos, pos[i]));

        buffer[raw.size() - i] = rectify(affine_transform(args));
        bufferi[raw.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    
    for (auto& b : buffer) buffer_lstm->add_input(b);

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
if(DEBUG) cerr<< "action_count " << action_count <<"\n";
      current_valid_actions.clear();
if(DEBUG) cerr<< "unary: " << unary << "nopen_parens: "<<nopen_parens<<"\n";
      for (auto a: possible_actions) {
        if (IsActionForbidden_Discriminative(adict.Convert(a), prev_a, buffer.size(), stack.size(), nopen_parens, unary))
          continue;
        current_valid_actions.push_back(a);
      }
if(DEBUG){
	cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
		cerr<<adict.Convert(current_valid_actions[i])<<" ";
	}
	cerr <<"\n";
}
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = action_lstm.back();
      Expression buffer_summary = buffer_lstm->back();
     
      Expression p_t = affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
      Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});
      if (sample && ALPHA != 1.0f) r_t = r_t * ALPHA;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
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
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
      //cerr << endl;
      ++action_count;
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // do action
      const string& actionString=adict.Convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];
if(DEBUG){
      
      cerr << "MODEL_ACT: " << adict.Convert(model_action)<<" ";
      cerr <<"GOLD_ACT: " << actionString<<"\n";
}

if(DEBUG) {
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
if(DEBUG) cerr << "  number of children to reduce: " << nchildren << endl;
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
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    return results;
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

bool islower(char c){if(c >= 'a' && c <= 'z') return true; return false;}
bool isupper(char c){if(c >= 'A' && c <= 'Z') return true; return false;}
bool isdigital(char c){if(c >= '0' && c <= '9') return true; return false;}
bool isalpha(char c){if(islower(c) || isupper(c)) return true; return false;}

std::string unkized(const std::string& word){
    int numCaps = 0;
    bool hasDigit = false;
    bool hasDash = false;
    bool hasLower = false;
    
    std::string result = "UNK";
    for(unsigned i = 0; i < word.size(); i ++){
        if(isdigital(word[i])) hasDigit = true;
	else if(word[i] == '-') hasDash = true;
        else if(islower(word[i])) hasLower = true;
	else if(isupper(word[i])) numCaps += 1;
    }
    string lower = word;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower); 
    if(isupper(word[0])){
        if(numCaps == 1) {
	    result = result + "-INITC";
	    if(train_dict.find(lower) != train_dict.end()) result = result + "-KNOWNLC";
	}
	else{
	    result = result + "-CAPS";
	}
    }
    else if(isalpha(word[0]) == false && numCaps > 0)	result = result + "-CAPS";
    else if(hasLower) result = result + "-LC";
    
    if(hasDigit) result = result + "-NUM";
    if(hasDash) result = result + "-DASH";

    
    if(lower.back() == 's' && lower.size() >= 3){
    	char ch2 = lower.size() >= 2 ? lower[lower.size()-2] : '0';
        if(ch2 != 's' && ch2 != 'i' && ch2 != 'u') result = result + "-s";
    }
    else if(lower.size() >= 5 && hasDash == false && (hasDigit == false || numCaps == 0)){
      	char ch1 = lower.size() >= 1 ? lower[lower.size()-1] : '0';
	char ch2 = lower.size() >= 2 ? lower[lower.size()-2] : '0';
        char ch3 = lower.size() >= 3 ? lower[lower.size()-3] : '0';

	if(ch2 == 'e' && ch1 == 'd')
       	    result = result + "-ed";
    	else if(ch3 == 'i' && ch2 == 'n' && ch1 == 'g')
       	    result = result + "-ing";
	else if(ch3 == 'i' && ch2 == 'o' && ch1 == 'n')
            result = result + "-ion";
	else if(ch2 == 'e' && ch1 == 'r')
            result = result + "-er";
	else if(ch3 == 'e' && ch2 == 's' && ch1 == 't')
            result = result + "-est";
	else if(ch2 == 'l' && ch1 == 'y')
            result = result + "-ly";
	else if(ch3 == 'i' && ch2 == 't' && ch1 == 'y')
            result = result + "-ity";
	else if(ch1 == 'y')
            result = result + "-y";
	else if(ch2 == 'a' && ch1 == 'l')
	    result = result + "-al";
    }
    return result;
}

std::string unkized_ch(const std::string word){
    return "UNK";
}
int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);//, 1989121011);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  DEBUG = conf.count("debug");
  if (conf.count("alpha")) {
    ALPHA = conf["alpha"].as<float>();
    if (ALPHA <= 0.f) { cerr << "--alpha must be between 0 and +infty\n"; abort(); }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<unsigned>();
    if (N_SAMPLES == 0) { cerr << "Please specify N>0 samples\n"; abort(); }
  }

  assert(conf.count("model_dir") && conf.count("words") && conf.count("train_dict") && conf.count("lang"));
  assert(conf["lang"].as<string>() == "en" || conf["lang"].as<string>() == "ch");
  std::string model_dir = conf["model_dir"].as<string>();

  Model model;

  //parser::KOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  //parser::KOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  //parser::KOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  //corpus.load_oracle(conf["training_data"].as<string>(), true);	
  {
  std::string word;
  ifstream ifs;
  ifs.open(model_dir+"/terminal.dict");
  while(ifs>>word) termdict.Convert(word);
  ifs.close();

  ifs.open(model_dir+"/non-terminal.dict");
  while(ifs>>word) ntermdict.Convert(word);
  ifs.close();
  
  ifs.open(model_dir+"/action.dict");
  while(ifs>>word) adict.Convert(word);
  ifs.close();

  ifs.open(model_dir+"/pos.dict");
  while(ifs>>word) posdict.Convert(word);
  ifs.close();

  PRETRAINED_DIM = parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  termdict.Freeze();
  termdict.SetUnk("UNK");
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze(); 
  }

  /*ofstream ofs;
  ofs.open("termdict");
  for(int i = 0; i < termdict.size(); i ++){
    ofs<< termdict.Convert(i)<<"\n";
  }
  ofs.close();

  ofs.open("adict");
  for(int i = 0; i < adict.size(); i ++){
    ofs<< adict.Convert(i)<<"\n";
  }
  ofs.close();

  ofs.open("ntermdict");
  for(int i = 0; i < ntermdict.size(); i ++){
    ofs<< ntermdict.Convert(i)<<"\n";
  }
  ofs.close();

  ofs.open("posdict");
  for(int i = 0; i < posdict.size(); i ++){
    ofs<< posdict.Convert(i)<<"\n";
  }
  ofs.close();
  exit(1);*/
  {
  std::string word;
  ifstream ifs(conf["train_dict"].as<string>().c_str());
  while(ifs>>word) train_dict[word] = 1;
  ifs.close();
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'P') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
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
    
    cerr<<"Reading model from " <<model_dir<<"/model"<<" .....\n";
    ifstream in((model_dir+"/model").c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;

  cerr<<"Please Input:\n";
  {
  std::string line;
  std::string word;
  vector<int> raw;
  vector<int> lc;
  vector<int> pos;
  vector<std::string> words;
  while(std::getline(std::cin, line)){
    if(line == "") continue;
    istrstream istr(line.c_str());
    raw.clear();
    lc.clear();
    pos.clear();
    words.clear();
    while(istr>>word){
      words.push_back(word);

      if(train_dict.find(word) != train_dict.end()) raw.push_back(termdict.Convert(word));
      else{
	 if(conf["lang"].as<string>() == "en") raw.push_back(termdict.Convert(unkized(word)));
	 else if(conf["lang"].as<string>() == "ch") raw.push_back(termdict.Convert(unkized_ch(word)));
	 else {std::cerr<<"lang error, it should be either en or ch.\n"; abort();}
      }

      std::transform(word.begin(), word.end(), word.begin(), ::tolower);
      lc.push_back(termdict.Convert(word));

      istr>>word;
      pos.push_back(posdict.Convert(word));
    }
  /*  for(unsigned i = 0; i < raw.size(); i ++){
       cout << termdict.Convert(raw[i])<<" ";
    }
    cout << "||| ";
    for(unsigned i = 0; i < raw.size(); i ++){
       cout << termdict.Convert(lc[i])<<" ";
    }
    cout << "||| ";

    for(unsigned i = 0; i < raw.size(); i ++){
       cout << posdict.Convert(pos[i])<<" ";
    }
    cout <<"\n";

    cout.flush();*/

    if(conf.count("samples")>0){
	for (unsigned z = 0; z < N_SAMPLES; ++z) {
	    ComputationGraph hg;
	    vector<unsigned> pred = parser.log_prob_parser(&hg,raw, lc, pos, true);
	    int ti = 0;
	    std::vector<std::string> btree;
            std::vector<int> openidx;
	    for (auto a : pred) {
		std::string act = adict.Convert(a);
		if (act[0] == 'S'){
		    btree.push_back("("+posdict.Convert(pos[ti])+" "+words[ti]+")");
		    ti++;
		}
		else if(act[0] == 'P'){
		    std::string tmp = btree.back();
                    btree.pop_back();
                    btree.push_back("("+ntermdict.Convert(action2NTindex[a]));
                    btree.push_back(tmp);
                    openidx.push_back(btree.size()-2);
		}
		else if(act[0] == 'R'){
		    std::string tmp = btree.back();
                    btree.pop_back();
                    int i = btree.size()-1;
                    while(i >= openidx.back()){
                        i -= 1;
                        tmp = btree.back() + " " + tmp;
                        btree.pop_back();
                    }
                    tmp += ")";
                    btree.push_back(tmp);
                    openidx.pop_back();
	    	}
		else{break;}
            }
	    cout<<btree[0]<<"\n";
       }
    }
    else{
        ComputationGraph hg;
        vector<unsigned> pred = parser.log_prob_parser(&hg,raw, lc, pos, false);
        std::vector<std::string> btree;
        std::vector<int> openidx;
	int ti = 0;
        for (auto a : pred) {
	    std::string act = adict.Convert(a);
            if (act[0] == 'S'){
//		cout<<"action: "<<act<<"\n";
		btree.push_back("("+posdict.Convert(pos[ti])+" "+words[ti]+")");
            	ti++;
/*		cout<<"btree: ";
		for(unsigned i = 0; i < btree.size(); i ++){ cout<<btree[i]<<"|||";}
		cout<<"\n";
		cout<<"openidx: ";
		for(unsigned i = 0; i < openidx.size(); i ++){ cout<<openidx[i]<<"|||";}
                cout<<"\n";
*/
            }
	    else if(act[0] == 'P'){
//		cout<<"action: "<<act<<"\n";
	        std::string tmp = btree.back();
		btree.pop_back();
		btree.push_back("("+ntermdict.Convert(action2NTindex[a]));
		btree.push_back(tmp);
		openidx.push_back(btree.size()-2);	
		
/*                cout<<"btree: ";
                for(unsigned i = 0; i < btree.size(); i ++){ cout<<btree[i]<<"|||";}
                cout<<"\n";
                cout<<"openidx: ";
                for(unsigned i = 0; i < openidx.size(); i ++){ cout<<openidx[i]<<"|||";}
                cout<<"\n";
*/
	    }
	    else if(act[0] == 'R'){
//		cout<<"action: "<<act<<"\n";
	    	std::string tmp = btree.back();
		btree.pop_back();
		int i = btree.size()-1;
		while(i >= openidx.back()){
			i -= 1;
			tmp = btree.back() + " " + tmp;
			btree.pop_back();
		}
		tmp += ")";
		btree.push_back(tmp);
		openidx.pop_back();
	
/*                cout<<"btree: ";
                for(unsigned i = 0; i < btree.size(); i ++){ cout<<btree[i]<<"|||";}
                cout<<"\n";
                cout<<"openidx: ";
                for(unsigned i = 0; i < openidx.size(); i ++){ cout<<openidx[i]<<"|||";}
                cout<<"\n";
*/
            }
	    else {break;}
        }
	cout << btree[0] << "\n";
    }

    /*if(conf.count("samples")>0){
	for (unsigned z = 0; z < N_SAMPLES; ++z) {
	    ComputationGraph hg;
	    vector<unsigned> pred = parser.log_prob_parser(&hg,raw, lc, pos, true);
	    int ti = 0;
	    for (auto a : pred) {
                cout << adict.Convert(a);
                if (adict.Convert(a) == "SHIFT"){
                        cout<<" "<<posdict.Convert(pos[ti])<< " " <<words[ti];
                        ti++;
                }
                cout << endl;
            }
            cout << endl;
       }
    }
    else{
        ComputationGraph hg;
        vector<unsigned> pred = parser.log_prob_parser(&hg,raw, lc, pos, false);
        int ti = 0;
        for (auto a : pred) {
            cout << adict.Convert(a);
            if (adict.Convert(a) == "SHIFT"){
            	cout<<" "<<posdict.Convert(pos[ti])<< " " <<words[ti];
            	ti++;
            }
            cout << endl;
        }
        cout << endl;
    }*/
  }
  }
}
