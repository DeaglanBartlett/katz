import string
from katz.good_turing import GoodTuring
from katz.back_off import BackOff
import katz.process_equations as process_equations
from katz.prior import KatzPrior

with open('data/romeoandjuliet.txt', 'r') as f:
    data = f.readlines()[75:4494]
    data = [d.strip() for d in data]
    data = ' '.join(data)
data = data.translate(str.maketrans('', '', string.punctuation)).lower().split()

#example = 0
#example = 1
example = 2

if example == 0:
    gt = GoodTuring(data)
    gt.plot_fit()

    for word in ['verona', 'shall', 'our']:
        k = gt.actual_count(word)
        kstar = gt.expected_count(word)
        print(word, k, kstar)
    
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

elif example == 1:

    n = 3
    new_data = [tuple(data[i:i+n]) for i in range(len(data)-n+1)]
    bo = BackOff(new_data)

    phrase = ('i', 'will')
    seen, unseen = bo.sort_endings(phrase)
    #print(seen)
    print(phrase)

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
        
elif example == 2:

    basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh"],
                ["+", "-", "*", "/", "pow"]]
    
#    kp = KatzPrior(2, basis_functions, 'data/FeynmanEquations.csv', 'data/NewFeynman.csv', input_delimiter=',')
    kp = KatzPrior(2, basis_functions, 'data/PhysicsEquations.csv', 'data/NewPhysics.csv', input_delimiter=';')
    for eq in ['x0**2', 'x0**3', 'x0*x1', 'sin(x0) + sin(x1)', 'sin(sin(x0+x1))', 'pow(x0, a0)', 'pow(a0, x0)', 'pow(x0, x0)', 'sqrt(x)',
            'a0 + a1*sin(x)', 'a0 - sin(x)*cos(a1)', 'a0 - pow(Abs(cos(a1)),sin(x))', 'pow(Abs(cos(a0)),sin(x))/a1', '(a0 + sin(x))**2/a1']:
        p = kp.logprior(eq)
        print(eq, p)
