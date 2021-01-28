from numba import njit
import numpy as np

# This calculates the full pmf and stores it in result.
# It is much, much quicker to pre-create result and then re-use it!

@njit
def poisson_binomial_pmf(probs,probslen,result):

    result[0] = 1.0-probs[0]
    result[1] = probs[0]
    signal_0,signal_1 = 0.0,0.0


    oldlen = 2
    for i in range(1,probslen):

        # set signal
        signal_0 = probs[i]
        signal_1 = 1.0-probs[i]

        # initialize result and calculate the two edge cases
        result[oldlen] = signal_0 * result[oldlen-1]

        t = result[0]
        result[0] = signal_1*t

        # calculate the interior cases
        for j in range(1,oldlen):
            tmp=result[j]
            result[j] = signal_0 * t + signal_1 * result[j]
            t=tmp

        oldlen += 1

    return result

@njit
def vp_poisson_binomial_pmf(probs,times,probslen,result,cvp=6,dt=4):

    """
    Poisson Binomial PMF given vp>5
    """

    # for j in range(probslen):
    #     for l in range(probslen):
    #         for vp in range(cvp+1):
    #             result[j,l,vp] = 0.

    result[0,0,0] = 1.0-probs[0]
    result[1,0,1] = probs[0]

    tgap = 0

    oldlen = 2
    for i in range(1,probslen):

        # set signal
        probsi = probs[i]
        timesi = times[i]

        # initialize result and calculate the two edge cases
        # new visibility period
        if times[i]-times[i-1]>4:
            tgap=1
        else:
            tgap=0
        for vp in range(1,cvp+1):
            result[oldlen,oldlen-1,vp] = probsi * (result[oldlen-1,oldlen-2,vp-1] * tgap + result[oldlen-1,oldlen-2,vp] * (1-tgap))
        result[oldlen,oldlen-1,cvp] += probsi * result[oldlen-1,oldlen-2,cvp] * tgap

        t0 = result[0].copy()
        t =  result[1].copy()
        result[0,0,0] = (1.0-probsi)*t0[0,0]
        for l in range(i):
            result[1,l,1] = (1-probsi)*t[l,1]
        result[1,i,1] = probsi*t0[0,0]

        # calculate the interior cases
        for j in range(2,oldlen):
            tmp = result[j].copy()
            for l in range(i):
                # new visibility period
                if timesi-times[l]>dt:
                    tgap=1
                else:
                    tgap=0
                for vp in range(1,cvp+1):
                    result[j,i,vp]+=  probsi * (t[l,vp-1] * tgap + t[l,vp] * (1-tgap))
                    result[j,l,vp] =  (1-probsi) * result[j,l,vp]
                result[j,i,cvp] += probsi * t[l,cvp] * tgap
            t = tmp

        oldlen += 1

    return result

@njit
def exp_vp_poisson_binomial_pmf(probs,times,weights,probslen,result,wresult,cvp=6,dt=4):

    """
    Expectation value of Poisson Binomial PMF given vp>5
    """

    # for j in range(probslen):
    #     for l in range(probslen):
    #         for vp in range(cvp+1):
    #             result[j,l,vp] = 0.

    result[0,0,0] = 1.0-probs[0]
    result[1,0,1] = probs[0]

    wresult[0,0,0] = 0.
    wresult[1,0,1] = probs[0] * weights[0]

    tgap = 0

    oldlen = 2
    for i in range(1,probslen):

        # set signal
        probsi = probs[i]
        timesi = times[i]
        weightsi = weights[i]

        # initialize result and calculate the two edge cases
        # new visibility period
        if times[i]-times[i-1]>4:
            tgap=1
        else:
            tgap=0
        for vp in range(1,cvp+1):
            result[oldlen,oldlen-1,vp] = probsi * (result[oldlen-1,oldlen-2,vp-1] * tgap + result[oldlen-1,oldlen-2,vp] * (1-tgap))
            wresult[oldlen,oldlen-1,vp] =probsi * (wresult[oldlen-1,oldlen-2,vp-1] * tgap + wresult[oldlen-1,oldlen-2,vp] * (1-tgap)) + \
                                                   result[oldlen,oldlen-1,vp]*weightsi
        result[oldlen,oldlen-1,cvp] += probsi *  result[oldlen-1,oldlen-2,cvp] * tgap
        wresult[oldlen,oldlen-1,cvp] += probsi *(wresult[oldlen-1,oldlen-2,cvp] * tgap + \
                                                  result[oldlen-1,oldlen-2,cvp] * tgap * weightsi)

        t0 = result[0].copy()
        w0 = wresult[0].copy()
        t =  result[1].copy()
        w =  wresult[1].copy()
        result[0,0,0] = (1.0-probsi)*t0[0,0]
        for l in range(i):
            result[1,l,1] = (1-probsi)*t[l,1]
            wresult[1,l,1] =(1-probsi)*w[l,1]
        result[1,i,1] = probsi*t0[0,0]
        wresult[1,i,1] =probsi*w0[0,0] + result[1,i,1]*weightsi

        # calculate the interior cases
        for j in range(2,oldlen):
            tmp = result[j].copy()
            wmp = wresult[j].copy()
            for l in range(i):
                # new visibility period
                if timesi-times[l]>dt:
                    tgap=1
                else:
                    tgap=0
                for vp in range(1,cvp+1):
                    result[j,i,vp] +=  probsi * (t[l,vp-1] * tgap + t[l,vp] * (1-tgap))
                    wresult[j,i,vp]+=  probsi * ((w[l,vp-1] * tgap + w[l,vp] * (1-tgap)) + \
                                                 (t[l,vp-1] * tgap + t[l,vp] * (1-tgap))*weightsi)
                    result[j,l,vp] =  (1-probsi) * result[j,l,vp]
                    wresult[j,l,vp] = (1-probsi) * wresult[j,l,vp]
                result[j,i,cvp] +=  probsi * t[l,cvp] * tgap
                wresult[j,i,cvp] += probsi * (w[l,cvp] * tgap + \
                                              t[l,cvp] * tgap * weightsi)
            t = tmp
            w = wmp

        oldlen += 1

    return result, wresult


#%% Unit Tests
import unittest, tqdm

class TestPoissonBinomial(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        super(TestPoissonBinomial, self).__init__(*args, **kwargs)

        # Set up model
        n=10
        a = np.linspace(5,14,n)
        p = np.random.rand(n)
        t = np.sort(np.random.rand(n)*100)

        # Randomly sample
        self.nsample=10000
        x_sample = np.random.rand(n,self.nsample).T < p

        self.count = np.zeros((n+1,7))
        self.sum_a = np.zeros((n+1,7))

        for ii in tqdm.tqdm(range(self.nsample)):
            t_sample = t[x_sample[ii]]
            vp = np.sum((t_sample[1:]-t_sample[:-1])>4) + 1
            if vp>6: vp=6

            k = np.sum(x_sample[ii])
            if k==0: vp=0
            if k==1: vp=1

            self.count[k, vp]+= 1
            self.sum_a[k,vp] += np.sum(a[x_sample[ii]])

        self.a=a
        self.t=t
        self.p=p
        self.n=n

    def test_poisson_binomial_pmf(self):

        result = np.zeros(self.n+1)
        poisson_binomial_pmf(self.p,self.n,result)
        output = result*np.sum(self.count)

        self.assertTrue( np.allclose(np.sum(self.count, axis=1), output, atol=5*np.sqrt(output)) )

        return output

    def test_vp_poisson_binomial_pmf(self):

        result = np.zeros((self.n+1,self.n+1,7))
        vp_poisson_binomial_pmf(self.p,self.t,self.n,result)
        output = np.sum(result*np.sum(self.count), axis=1)

        # All values within 5sigma of random samples
        self.assertTrue( np.allclose(self.count, output, atol=5*np.sqrt(output)) )
        self.assertAlmostEqual( result[0,0,0], np.prod(1-self.p), 8)
        self.assertAlmostEqual( result[1,0,1], self.p[0]*np.prod(1-self.p[1:]), 8)
        self.assertAlmostEqual( np.sum(result[self.n,self.n-1,:]), np.prod(self.p), 8)

        return output

    def test_exp_vp_poisson_binomial_pmf(self):

        result = np.zeros((self.n+1,self.n+1,7))
        wresult = np.zeros((self.n+1,self.n+1,7))
        exp_vp_poisson_binomial_pmf(self.p,self.t,self.a,self.n,result,wresult)
        output = np.sum(wresult*np.sum(self.count), axis=1)

        # All values within 5sigma of random samples
        mean_a = np.sum(self.a * self.p) / np.sum(self.p)
        self.assertTrue( np.allclose(self.sum_a/mean_a, output/mean_a, atol=10*np.sqrt(output/mean_a)) )
        #print((self.sum_a/mean_a - output/mean_a)/np.sqrt(output/mean_a))
        self.assertEqual( wresult[0,0,0], 0, 8)
        self.assertAlmostEqual( wresult[1,0,1], self.p[0]*np.prod(1-self.p[1:])*self.a[0], 8)
        self.assertAlmostEqual( np.sum(wresult[self.n,self.n-1,:]),
                                np.prod(self.p)*np.sum(self.a), 8)

        return output


if __name__ == '__main__':
    unittest.main()
