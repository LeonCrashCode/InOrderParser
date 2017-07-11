#ifndef PARSER_ORACLE_H_
#define PARSER_ORACLE_H_

#include <iostream>
#include <vector>
#include <string>

namespace dynet { class Dict; }

namespace parser {

// a sentence can be viewed in 4 different ways:
//   raw tokens, UNKed, lowercased, and POS tags
struct Sentence {
  bool SizesMatch() const { return raw.size() == unk.size() && raw.size() == lc.size() && raw.size() == pos.size(); }
  size_t size() const { return raw.size(); }
  std::vector<int> raw, unk, lc, pos;
};

// base class for transition based parse oracles
struct Oracle {
  virtual ~Oracle();
  Oracle(dynet::Dict* dict, dynet::Dict* adict, dynet::Dict* pdict) : d(dict), ad(adict), pd(pdict), sents() {}
  unsigned size() const { return sents.size(); }
  dynet::Dict* d;  // dictionary of terminal symbols
  dynet::Dict* ad; // dictionary of action types
  dynet::Dict* pd; // dictionary of POS tags (preterminal symbols)
  std::string devdata;
  std::vector<Sentence> sents;
  std::vector<std::vector<int>> actions;
 protected:
  static void ReadSentenceView(const std::string& line, dynet::Dict* dict, std::vector<int>* sent);
};

// oracle that predicts nonterminal symbols with a NT(X) action
// the action NT(X) effectively introduces an "(X" on the stack
// # (S (NP ...
// raw tokens
// tokens with OOVs replaced
class TopDownOracle : public Oracle {
 public:
  TopDownOracle(dynet::Dict* termdict, dynet::Dict* adict, dynet::Dict* pdict, dynet::Dict* nontermdict) :
      Oracle(termdict, adict, pdict), nd(nontermdict) {}
  // if is_training is true, then both the "raw" tokens and the mapped tokens
  // will be read, and both will be available. if false, then only the mapped
  // tokens will be available
  void load_bdata(const std::string& file);
  void load_oracle(const std::string& file, bool is_training);
  dynet::Dict* nd; // dictionary of nonterminal types
};

// oracle that predicts nonterminal symbols with a NT(X) action
// the action NT(X) effectively introduces an "(X" on the stack
// # (S (NP ...
// raw tokens
// tokens with OOVs replaced
class TopDownOracleGen : public Oracle {
 public:
  TopDownOracleGen(dynet::Dict* termdict, dynet::Dict* adict, dynet::Dict* pdict, dynet::Dict* nontermdict) :
      Oracle(termdict, adict, pdict), nd(nontermdict) {}
  void load_oracle(const std::string& file);
  dynet::Dict* nd; // dictionary of nonterminal types
};

class TopDownOracleGen2 : public Oracle {
 public:
  TopDownOracleGen2(dynet::Dict* termdict, dynet::Dict* adict, dynet::Dict* pdict, dynet::Dict* nontermdict) :
      Oracle(termdict, adict, pdict), nd(nontermdict) {}
  void load_oracle(const std::string& file);
  dynet::Dict* nd; // dictionary of nonterminal types
};

} // namespace parser

#endif
