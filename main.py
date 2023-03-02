import string
from katz.good_turing import GoodTuring
from katz.back_off import BackOff

with open('data/romeoandjuliet.txt', 'r') as f:
    data = f.readlines()[75:4494]
    data = [d.strip() for d in data]
    data = ' '.join(data)
data = data.translate(str.maketrans('', '', string.punctuation)).lower().split()

#data = data[:1000]

#gt = GoodTuring(data)
#gt.plot_fit()
#quit()
#for word in ['verona', 'shall', 'our']:
#    k = gt.actual_count(word)
#    kstar = gt.expected_count(word)
#    print(word, k, kstar)
    
## Split data into tuples of n
#w = 'verona'
#for n in range(1, 4):
#    print('\nNEW:', n)
#    new_data = [tuple(data[i:i+n]) for i in range(len(data)-n+1)]
#    print(len(new_data))
#    bo = BackOff(new_data)
#    print(bo.all_gt[1].actual_count((w,)))
#    print(bo.all_gt[1].expected_count((w,)))
#    for i in range(1, n+1):
#        print(i, len(bo.all_gt[i].corpus), bo.all_gt[i].corpus[-4:])
#quit()

n = 3
new_data = [tuple(data[i:i+n]) for i in range(len(data)-n+1)]
bo = BackOff(new_data)

phrase = ('i', 'will')
seen, unseen = bo.sort_endings(phrase)
#print(seen)

for w in ['kiss', 'not', 'romeo', 'shakespeare']:

    print('\nSTARTING:', w)
    new_phrase = phrase + (w,)
    c = bo.all_gt[len(new_phrase)].actual_count(new_phrase)
    if c > 0:
        cstar = bo.all_gt[len(new_phrase)].expected_count(new_phrase)
    else:
        cstar = 0
    pbo = bo.get_pbo(w, phrase)
    print(w, c, cstar, pbo)
        
    
"""
TO DO
- Introduce k parameter for back_off
"""
