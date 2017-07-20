import sys

def tree(acts):
	btree = []
	openidx = []
	wid = 0
	for act in acts:
		if act[0] == 'S':
			tmp = act.split()
			btree.append("("+tmp[1]+" "+tmp[2]+")")
			wid += 1
		elif act[0] == 'P':
			btree.insert(-1,"("+act[3:-1])
			openidx.append(len(btree)-2)
		else:
			tmp = " ".join(btree[openidx[-1]:])+")"
			btree = btree[:openidx[-1]]
			btree.append(tmp)
			openidx = openidx[:-1]
	print btree[0]

if __name__ == "__main__":
	actions = []
	action = []	
	for line in open(sys.argv[1]):
		line = line.strip()
		if line == "":
			actions.append(action[:-1])
			action = []
		else:
			action.append(line)

	for i in range(len(actions)):	
		tree(actions[i]);
