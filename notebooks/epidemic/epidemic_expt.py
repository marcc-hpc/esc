
# import required libraries
import sys
import numpy as np
import scipy
import scipy.integrate

class StochasticSIR:

    def __init__(self,beta,gamma,S,I,R):
        self.S = S
        self.I = I
        self.R = R
        self.beta = beta
        self.gamma = gamma
        self.t = 0.
        self.N = S + I + R
        self.trajectory = np.array([[self.S, self.I, self.R]])
        self.times = None

    def step(self):
        transition = None
        # define rates
        didt = self.beta * self.S * self.I
        drdt = self.gamma * self.I
        total_rate = didt + drdt
        if total_rate == 0.:
            return transition, self.t
        # get a random number
        rand = np.random.random()
        # rates determine the event
        if rand < didt/drdt:
            self.S -= 1
            self.I += 1
            transition = 1
        else:
            self.I -= 1
            self.R += 1
            transition = 2
        # the event happens in the future
        dt = np.random.exponential(1./total_rate,1)[0]
        self.t += dt
        return transition, self.t

    def run(self, T=None, make_traj=True):
        """The Gillespie algorithm."""
        if T is None:
            T = sys.maxsize
        self.times = [0.]
        t0 = self.t
        transition = 1
        while self.t < t0 + T:
            transition, t = self.step()
            if not transition:
                return self.t
            if make_traj: self.trajectory = np.concatenate(
                (self.trajectory, [[self.S,self.I,self.R]]), axis=0)
            self.times.append(self.t)
        return self.tB
    
# roll the models above into a single function
def study_beta(
    beta = .005,
    gamma = 1./2,
    N = 100,
    I = 1,
    n_expts = 1000):
    """Run many experiments for a particular beta 
    to see how many infections spread."""
    S = N-I
    result = np.zeros((n_expts,))
    for expt_num in range(len(result)):
        model = StochasticSIR(beta=beta,gamma=gamma,S=S,I=I,R=R)
        model.run()
        result[expt_num] = model.trajectory[-1][2]
    return result

if __name__=='__main__':
    print(' '.join(sys.argv))
    # testing mode
    mode = 'test' if len(sys.argv)==2 else 'sweep'
    
    # ensure that we have an argument for the seed
    if len(sys.argv)<2:
        raise Exception('you must supply a seed')
    else: seed_val = int(sys.argv[1])

    # initial settings
    N = 1000
    I = 1
    S = N - I
    R = 0
    beta = 0.2
    gamma = 1./10
    
    # parameter sweep settings, ten total
    sweep_global = np.arange(0.001,0.02+0.001,0.002) 
    
    np.random.seed(seed_val)
    n_expts = 100
    if mode=='sweep':
        # if we send a second index it marks the beta parameter in the sweep
        index = int(sys.argv[2])
        beta = sweep_global[index]
    elif mode=='test':
        beta = 0.002
    result = study_beta(beta=beta,n_expts=n_expts)
    if mode=='test':
        print('result: %s'%str(result))
        print('average: %s'%str(result.mean()))
    elif mode=='sweep':
        # write the results to a file
        with open('result_%d.txt'%index,'w') as fp:
            fp.write("n_expts %d\nN %d\nI %d\nS %d\nbeta %.4f\ngamma %.4f\nresult %s\n"%(
                n_expts,N,I,S,beta,gamma,result))
            fp.write('average %s\n'%str(result.mean()))
