Wl = []
BestWl = []
with open("results",encoding="utf8") as f:
	i = 0
	for line in f:
		i+=1
		splitLine = line.split()
		if(i%2 == 1):
			BestWl.append(float(splitLine[2]))
		else:
			Wl.append(float(splitLine[2]))

print("Avg OneShot: ", sum(Wl)/len(Wl))
print("Avg Best: ", sum(BestWl)/len(BestWl))

