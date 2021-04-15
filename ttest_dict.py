
import collections, itertools    
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i in range(5):
    refsets['pos'].add(i)
    refsets['neg'].add(i)
    refsets['neu'].add(i)

for i in refsets.keys():
    print(i)
