import numpy as np
from pymoo.core.sampling import Sampling


class PrecisionSampling(Sampling):
    def _do(self, problem, n_samples,enlarge, model='xaj'):
        precision = 1 
        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            num_steps = np.floor((xu - xl) / precision).astype(int) + 1
            X_int = np.random.randint(0, num_steps, size=(n_samples, problem.n_var))
            X = xl + X_int * precision
        else:
            X = np.random.random((n_samples, problem.n_var))
            X = np.round(X / precision) * precision
        
        if problem.has_bounds():
            X = np.clip(X, xl, xu)
        
        if model == 'xaj':
            X = self._adjust_xaj_model(X, xl, xu, n_samples, precision,enlarge)
        return X

    def _adjust_xaj_model(self, X, xl, xu, n_samples, precision, enlarge):
        # print(X.shape)

        for i in range(n_samples):
            # print(X[i, 9] ,X[i, 10])
            max_sum = 0.95 / 0.7 * enlarge
            while (X[i, 9] + X[i, 10]) >= max_sum or X[i, 9] == 0:
                num_steps_9_11 = np.floor((xu[9:11] - xl[9:11]) / precision).astype(int) + 1
                X_int_9_11 = np.random.randint(0, num_steps_9_11, size=(2,))
                X[i, 9:11] = xl[9:11] + X_int_9_11 * precision
                if (X[i, 9] + X[i, 10]) < max_sum and X[i, 9] > 0:
                    break
            # print(X[:, 9] ,X[:, 10])      
        return X


class HySampling():
    def __init__(self, base_sampler=None, candidates=None,  model = 'xaj'):
        super().__init__()
        if base_sampler is None:
            base_sampler = PrecisionSampling()
        self.base_sampler = base_sampler
        self.candidates = candidates
        self.model = model


        if self.N > 0:
            self._do = self._do_replace
        else:
            self._do = self.base_sampler._do
    @property
    def N(self):
        if self.candidates is None:
            return 0
        else:
            return np.shape(self.candidates)[0]
        
    def _do_replace(self, problem, n_samples, enlarge,**kwargs):
        candidates = np.array(self.candidates)

        if np.shape(candidates)[1] != problem.n_var:
            raise ValueError("Candidates must have dimension equals n_var on axis=1")
        if n_samples == self.N:
            return candidates


        if n_samples > self.N:
            X1 = self.base_sampler._do(problem, n_samples - self.N, enlarge=enlarge, model=self.model, **kwargs)
            X = np.concatenate((candidates, X1), axis=0)
        else:
            X = self.base_sampler._do(problem, n_samples, enlarge=enlarge, model=self.model, **kwargs)
        
        return X
