#ifndef DYNET_DICT_H_
#define DYNET_DICT_H_

#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace boost { namespace serialization { class access; } }

namespace dynet {

class Dict {
typedef std::unordered_map<std::string, int> Map;
public:
  Dict() : frozen(false), map_unk(false), unk_id(-1) {
  }

  inline unsigned size() const { return words_.size(); }

  inline bool contains(const std::string& words) {
    return !(d_.find(words) == d_.end());
  }

  void freeze() { frozen = true; }
  bool is_frozen() { return frozen; }

  inline int convert(const std::string& word) {
    auto i = d_.find(word);
    if (i == d_.end()) {
      if (frozen) {
        if (map_unk) {
          return unk_id;
        }
        else {
         std::cerr << map_unk << std::endl;
          std::cerr << "Unknown word encountered: " << word << std::endl;
          throw std::runtime_error("Unknown word encountered in frozen dictionary: " + word);
        }
      }
      words_.push_back(word);
      return d_[word] = words_.size() - 1;
    } else {
      return i->second;
    }
  }
  
  inline const std::string& convert(const int& id) const {
    assert(id < (int)words_.size());
    return words_[id];
  }
  
  void set_unk(const std::string& word) {
    if (!frozen)
      throw std::runtime_error("Please call set_unk() only after dictionary is frozen");
    if (map_unk)
      throw std::runtime_error("Set UNK more than one time");
  
    // temporarily unfrozen the dictionary to allow the add of the UNK
    frozen = false;
    unk_id = convert(word);
    frozen = true;
  
    map_unk = true;
  }

  int get_unk_id() const { return unk_id; }
  const std::vector<std::string> & get_words() const { return words_; }
  
  void clear() { words_.clear(); d_.clear(); }

private:
  bool frozen;
  bool map_unk; // if true, map unknown word to unk_id
  int unk_id; 
  std::vector<std::string> words_;
  Map d_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);

};

std::vector<int> read_sentence(const std::string& line, Dict* sd);
void read_sentence_pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td);

} // namespace dynet

#endif
