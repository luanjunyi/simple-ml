import random
import numpy as np

def grad_check(func, x, verbose=False, force_finish=False):
    rndstate = random.getstate()

    random.setstate(rndstate)  
    _, grad = func(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    max_diff = 0
    h = 1e-4
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]

        x[ix] = old_value - h
        random.setstate(rndstate)
        f1, _ = func(x)

        x[ix] = old_value + h
        random.setstate(rndstate)
        f2, _ = func(x)

        numgrad = (f2 - f1) / (2 * h)
        x[ix] = old_value

        # Compare gradients
        reldiff = np.max(abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix])))
        if reldiff > 1e-5:
            print "Gradient check failed.", reldiff
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            #import pdb; pdb.set_trace()
            if not force_finish:
                return
        else:
            if verbose:
                print reldiff
        max_diff = max(max_diff, reldiff)
        it.iternext() # Step to next dimension

    print "Gradient finished!", max_diff
