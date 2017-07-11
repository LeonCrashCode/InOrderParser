import sys

def tree(acts):
	btree = []
	openidx = []
	for act in acts:
		if act[0] == 'S':
			btree.append(act.split()[-1])
		elif act[0] == 'N':
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
	info = False	
	for line in open(sys.argv[1]):
		line = line.strip()
		if info == False:
			n = int(line.split("|||")[0])
			v = float(line.split("|||")[1])
			info = True
		else:
			if line == "":
				actions.append([n,v,action[:-1]])
				action = []
				info = False
			else:
				action.append(line)


	for i in range(len(actions)):
		print actions[i][0],"|||",actions[i][1],"|||",
		tree(actions[i][-1]);
