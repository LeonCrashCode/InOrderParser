import sys

def tree(acts, sents):
	btree = []
	openidx = []
	wid = 0
	for act in acts:
		if act[0] == 'S':
			btree.append("(XX "+sents[wid]+")")
			wid += 1
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
	for line in open(sys.argv[1]):
		line = line.strip()
		if line == "":
			actions.append(action[:-1])
			action = []
		else:
			action.append(line)

	surfaces = []
	cnt = 0
	for line in open(sys.argv[2]):
		line = line.strip()
		cnt += 1
		if cnt == 3:
			surfaces.append(line.split())
		if line == "":
			cnt = 0

	assert len(actions) == len(surfaces)

	for i in range(len(surfaces)):	
		tree(actions[i], surfaces[i]);
