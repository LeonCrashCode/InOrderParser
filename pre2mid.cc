#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
using namespace std;

typedef struct Node{
	string label;
	vector<Node*> nodelist;	
	Node(string _label){ label = _label;}
}Node;


void convert(Node* n, const vector<string>& actions, unsigned& index){
	while(index < actions.size()){
		if(actions[index][0] == 'R'){
		//	cout<<actions[index]<<"\n";
			index ++; 
			break;
		}
		else if(actions[index][0] == 'N'){
		//	cout<<actions[index]<<"\n";
			(n->nodelist).push_back(new Node(actions[index++]));
			convert((n->nodelist).back(), actions, index);
		}
		else if(actions[index][0] == 'S'){
		//	cout<<actions[index]<<"\n";
			(n->nodelist).push_back(new Node(actions[index++]));
		}
		else{
			cerr<<"error "<<actions[index]<<"\n";
			exit(0);
		}
	}
}
void show(Node* p){
	cout<<p->label<<"\n";
	for(unsigned i = 0; i < (p->nodelist).size(); i ++){
		show((p->nodelist)[i]);
	}
	if(p->nodelist.size() != 0){
		cout<<"REDUCE\n";
	}
}

void one_mid(Node* p){
	if(p->nodelist.size()==0){
		cout<<p->label<<"\n";
		return;
	}
	one_mid(p->nodelist[0]);
	cout<<p->label<<"\n";
	for(unsigned i = 1; i < p->nodelist.size(); i++){
		one_mid(p->nodelist[i]);
	}
	cout<<"REDUCE\n";
}

int main(int argc, char** argv){
	
	ifstream ifs(argv[1]);
	string line;
	int cnt = 0;
	int lin = 0;
	vector<string> actions;
	while(getline(ifs,line)){
		if(cnt < 5){
                        cout<<line<<"\n";
			cnt++;
                        continue;
                }
		if(line == ""){
			cerr<<lin++<<"\n";
			Node root("root");
			Node *p = &root;
			unsigned index = 0;
			convert(p,actions,index);
			//show(p->nodelist[0]);
			one_mid(p->nodelist[0]);
			cout<<"TERM\n";
			cout<<"\n";
			cnt = 0;
			actions.clear();
			continue;
		}
		actions.push_back(line);
	}
	
	return 0;
}
