#include "impl/oracle.h"

#include <cassert>
#include <fstream>

#include "cnn/dict.h"
#include "impl/compressed-fstream.h"

using namespace std;

namespace parser {


Oracle::~Oracle() {}

inline bool is_ws(char x) { //check whether the character is a space or tab delimiter
  return (x == ' ' || x == '\t');
}

inline bool is_not_ws(char x) {
  return (x != ' ' && x != '\t');
}

void Oracle::ReadSentenceView(const std::string& line, cnn::Dict* dict, vector<int>* sent) {
  unsigned cur = 0;
  while(cur < line.size()) {
    while(cur < line.size() && is_ws(line[cur])) { ++cur; }
    unsigned start = cur;
    while(cur < line.size() && is_not_ws(line[cur])) { ++cur; }
    unsigned end = cur;
    if (end > start) {
      unsigned x = dict->Convert(line.substr(start, end - start));
      sent->push_back(x);
    }
  }
  assert(sent->size() > 0); // empty sentences not allowed
}

void TopDownOracle::load_bdata(const string& file) {
   devdata=file;
}

void TopDownOracle::load_oracle(const string& file, bool is_training) {
  cerr << "Loading top-down oracle from " << file << " [" << (is_training ? "training" : "non-training") << "] ...\n";
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const string kREDUCE = "REDUCE";
  const string kSHIFT = "SHIFT";
  const string kTERM = "TERM";
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  const int kTERM_INT = ad->Convert("TERM");
  int lc = 0;
  string line;
  vector<int> cur_acts;
  while(getline(in, line)) {
    ++lc;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    if (line.size() == 0 || (line[0] == '!' && line[1] == '#')) continue;
    sents.resize(sents.size() + 1);
    auto& cur_sent = sents.back();
    if (is_training) {  // at training time, we load both "UNKified" versions of the data, and raw versions
      ReadSentenceView(line, pd, &cur_sent.pos);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.raw);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.lc);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.unk);
    } else { // at test time, we ignore the raw strings and just use the "UNKified" versions
      ReadSentenceView(line, pd, &cur_sent.pos);
      getline(in, line);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.lc);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.unk);
      cur_sent.raw = cur_sent.unk;
    }
    lc += 3;
    if (!cur_sent.SizesMatch()) {
      cerr << "Mismatched lengths of input strings in oracle before line " << lc << endl;
      abort();
    }
    int termc = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      assert(line.find(' ') == string::npos);
      if (line == kREDUCE) {
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        // Convert NT
        nd->Convert(line.substr(3, line.size() - 4));
        // NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(ad->Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
      } else if (line == kTERM){
	cur_acts.push_back(kTERM_INT);
      } else {
        cerr << "Malformed input in line " << lc << endl;
        abort();
      }
    }
    actions.push_back(cur_acts);
    if (termc != sents.back().size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      abort();
    }
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}

void TopDownOracle::load_oracle(const string& file, bool is_training) {
  cerr << "Loading top-down oracle from " << file << " [" << (is_training ? "training" : "non-training") << "] ...\n";
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const string kREDUCE = "REDUCE";
  const string kSHIFT = "SHIFT";
  const string kTERM = "TERM";
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  const int kTERM_INT = ad->Convert("TERM");
  int lc = 0;
  string line;
  vector<int> cur_acts;
  while(getline(in, line)) {
    ++lc;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    if (line.size() == 0 || (line[0] == '!' && line[1] == '#')) continue;
    sents.resize(sents.size() + 1);
    auto& cur_sent = sents.back();
    if (is_training) {  // at training time, we load both "UNKified" versions of the data, and raw versions
      ReadSentenceView(line, pd, &cur_sent.pos);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.raw);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.lc);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.unk);
    } else { // at test time, we ignore the raw strings and just use the "UNKified" versions
      ReadSentenceView(line, pd, &cur_sent.pos);
      getline(in, line);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.lc);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.unk);
      cur_sent.raw = cur_sent.unk;
    }
    lc += 3;
    if (!cur_sent.SizesMatch()) {
      cerr << "Mismatched lengths of input strings in oracle before line " << lc << endl;
      abort();
    }
    int termc = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      assert(line.find(' ') == string::npos);
      if (line == kREDUCE) {
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        // Convert NT
        nd->Convert(line.substr(3, line.size() - 4));
        // NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(ad->Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
      } else if (line == kTERM){
	cur_acts.push_back(kTERM_INT);
      } else {
        cerr << "Malformed input in line " << lc << endl;
        abort();
      }
    }
    actions.push_back(cur_acts);
    if (termc != sents.back().size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      abort();
    }
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}

/*void TopDownOracleGen::load_oracle(const string& file) {
  cerr << "Loading top-down generative oracle from " << file << endl;
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const string kREDUCE = "REDUCE";
  const string kSHIFT = "SHIFT";
  const string kTERM = "TERM";
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  const int kTERM_INT = ad->Convert("TERM");
  int lc = 0;
  string line;
  vector<int> cur_acts;
  while(getline(in, line)) {
    ++lc;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    if (line.size() == 0 || (line[0] == '!' && line[1] == '#')) continue;
    sents.resize(sents.size() + 1);
    auto& cur_sent = sents.back();
    getline(in, line);
    ReadSentenceView(line, d, &cur_sent.raw);
    cur_sent.pos = cur_sent.unk = cur_sent.lc = cur_sent.raw;
    lc += 1;
    if (!cur_sent.SizesMatch()) {
      cerr << "Mismatched lengths of input strings in oracle before line " << lc << endl;
      abort();
    }
    int termc = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      assert(line.find(' ') == string::npos);
      if (line == kREDUCE) {
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        // Convert NT
        nd->Convert(line.substr(3, line.size() - 4));
        // NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(ad->Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
      } else if (line == kTERM) {
        cur_acts.push_back(kTERM_INT);
      }else {
        cerr << "Malformed input in line " << lc << endl;
        abort();
      }
    }
    actions.push_back(cur_acts);
    if (termc != sents.back().size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      abort();
    }
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}

void TopDownOracleGen2::load_oracle(const string& file) {
  cerr << "Loading top-down generative oracle from " << file << endl;
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const string kREDUCE = "REDUCE";
  const string kSHIFT = "SHIFT";
  const string kTERM = "TERM";
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  const int kTERM_INT = ad->Convert("TERM");
  int lc = 0;
  string line;
  vector<int> cur_acts;
  while(getline(in, line)) {
    ++lc;
    sents.push_back(Sentence());
    auto& raw = sents.back().raw;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    int termc = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      if (line == kREDUCE) {
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        // Convert NT
        nd->Convert(line.substr(3, line.size() - 4));
	// NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(ad->Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
      } else if (line == kTERM) {
        cur_acts.push_back(kTERM_INT);
      } else {
	cur_acts.push_back(kSHIFT_INT);	
	termc++;
	int term = d->Convert(line.substr(line.find(' ')+1));
	raw.push_back(term);
      }
    }
    actions.push_back(cur_acts);
    if (termc != sents.back().size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      abort();
    }
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}*/


} // namespace parser
