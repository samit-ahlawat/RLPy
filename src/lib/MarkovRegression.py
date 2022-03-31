import numpy as np
import logging
import scipy

logging.basicConfig(level=logging.DEBUG)


class RegressionResult(object):
    def __init__(self, lastDiff):
        self.lastDiff = lastDiff
        self.history = []
        self.params = None
        self.transitionProb = None
        self.smoothed_marginal_probabilities = None

    def apppendDiff(self, diff):
        self.history.append(diff)

    def summary(self):
        out_str = ",".join([str(p) for p in self.params])
        return out_str

    def addParams(self, params, transitionProb):
        self.params = np.ndarray(params.shape[0]*params.shape[1] + transitionProb.shape[0])
        self.params[0:transitionProb.shape[0]] = transitionProb[:, 0]
        self.params[transitionProb.shape[0]:] = params.ravel(order='F')


class MarkovRegression(object):
    def __init__(self, nstates, family="Gaussian", diffThresh=1E-5, niters=1000, alpha=1E-6):
        self.nStates = nstates
        self.family = family
        self.params = None
        self.transitionProb = None
        self.cumProb = None
        self.diffThreshold = diffThresh
        self.maxIters = niters
        self.learningRate = alpha
        self.initWt = 1E-3
        self.delta = 0.1
        self.logger = logging.getLogger(self.__class__.__name__)


    def initializeParams(self, xvars: tuple):
        self.params = np.random.random((self.nStates, xvars[1]+2)) * self.initWt
        self.params[:, -1] = np.abs(self.params[:, -1])  # variance is positive
        self.transitionProb = np.random.random((self.nStates, self.nStates)) * self.initWt
        self.transitionProb[:, :] = np.add(self.transitionProb, self.transitionProb.T)
        self.transitionProb = np.divide(self.transitionProb, 2.0)
        self.initialProb = np.full(self.nStates, 1.0/self.nStates, dtype=np.float64)
        rowsum = np.sum(self.transitionProb, axis=1)
        for i in range(self.nStates):
            self.transitionProb[i, i] = 1.0 - (rowsum[i] - self.transitionProb[i, i])
        self.cumProb = np.zeros((xvars[0], self.nStates))
        self.cumProb[0, :] = self.initialProb

    def calculateCumProb(self):
        cprob = self.cumProb
        cprob[0, :] = self.initialProb
        for i in range(1, cprob.shape[0]):
            cprob[i, :] = np.einsum("k,kj->j", cprob[i-1, :], self.transitionProb)
            #assert all(cprob[i, :] >= 0.0)
            if np.asarray(cprob[i, :] < 0).sum():
                import pdb; pdb.set_trace()
        sumval = np.sum(cprob, axis=1)
        self.cumProb = np.divide(cprob, sumval[:, np.newaxis])

    def logLikelihood(self, exog: np.ndarray, endog: np.ndarray) -> float:
        v1 = np.subtract(endog[:, np.newaxis], self.params[:, 0])
        bx = np.einsum("ij,kj->ik", exog, self.params[:, 1:-1])
        v1 = np.subtract(v1, bx)
        v1 = np.multiply(v1, v1)
        v1 = np.divide(v1, self.params[:, -1])
        v1 = np.exp(-v1/2.0)
        stddevp = np.sqrt(self.params[:, -1])
        v1 = np.divide(v1, stddevp)
        v1 = np.einsum("ij,ij->i", v1, self.cumProb)
        zero_vals = np.asarray(np.abs(v1) < 1E-9)
        if zero_vals.sum():
            v1[zero_vals] = 1.0
        v1 = np.log(v1)
        v1[np.isnan(v1)] = 0.0
        if np.isnan(v1).sum():
            import pdb; pdb.set_trace()
        return np.sum(v1)

    def gradAscent(self, exog: np.ndarray, endog: np.ndarray):
        self.calculateCumProb()
        base_value = self.logLikelihood(exog, endog)
        newParam = self.params.copy()
        for i in range(self.params.shape[0]):
            for j in range(self.params.shape[1]):
                self.params[i, j] += self.delta
                new_value = self.logLikelihood(exog, endog)
                self.params[i, j] -= self.delta
                newParam[i, j] += self.learningRate * (new_value - base_value)/self.delta

        newTransProb = self.transitionProb.copy()
        for i in range(self.transitionProb.shape[0]):
            assert self.transitionProb[i, i] > 0
            for j in range(i+1, self.transitionProb.shape[1]):
                delta = min(self.transitionProb[i, i], self.transitionProb[j, j]) * 0.1
                if delta < 1E-8:
                    continue
                self.transitionProb[i, j] += delta
                self.transitionProb[j, i] += delta
                self.transitionProb[i, i] -= delta
                self.transitionProb[j, j] -= delta
                self.calculateCumProb()
                new_value = self.logLikelihood(exog, endog)
                self.transitionProb[i, j] -= delta
                self.transitionProb[j, i] -= delta
                self.transitionProb[i, i] += delta
                self.transitionProb[j, j] += delta

                orig_value = newTransProb[i, j]
                newTransProb[i, j] += self.learningRate * (new_value - base_value)/delta
                if newTransProb[i, j] < 0:
                    newTransProb[i, j] = orig_value
                    continue
                newTransProb[j, i] = newTransProb[i, j]
        rowsum = np.sum(newTransProb, axis=1)
        for i in range(newTransProb.shape[0]):
            newTransProb[i, i] = 1.0 - (rowsum[i] - newTransProb[i, i])

        initProb = self.initialProb.copy()
        for i in range(initProb.shape[0]-1):
            self.initialProb[i] += self.delta
            if self.initialProb[i] > 1.0:
                self.initialProb[i] -= self.delta
                continue
            self.calculateCumProb()
            new_value = self.logLikelihood(exog, endog)
            self.initialProb[i] -= self.delta
            orig_value = initProb[i]
            initProb[i] += self.learningRate * (new_value - base_value)/self.delta
            if (initProb[i] < 0) or (initProb[i] > 1):
                initProb[i] = orig_value
        initProb[-1] = 1.0 - np.sum(initProb[0:-1])

        newParam[:, -1] = np.abs(newParam[:, -1])  # variance cannot be negative
        diff = np.subtract(newParam, self.params)
        diff2 = np.subtract(newTransProb, self.transitionProb)
        diff3 = np.subtract(initProb, self.initialProb)
        if np.isnan(newParam).sum():
            self.logger.info("Nan in newParam")
            import pdb; pdb.set_trace()
        self.params[:, :] = newParam
        if np.isnan(newTransProb).sum():
            self.logger.info("NaN in newTransProb")
            import pdb; pdb.set_trace()
        self.transitionProb[:, :] = newTransProb
        if np.isnan(initProb).sum():
            self.logger.info("NaN in initProb")
            import pdb; pdb.set_trace()
        self.initialProb[:] = initProb
        ss = np.sum(np.multiply(diff, diff)) + np.sum(np.multiply(diff2, diff2)) + np.sum(np.multiply(diff3, diff3))
        den = self.transitionProb.shape[0]*self.transitionProb.shape[1] + self.params.shape[0]*self.params.shape[1] + self.nStates
        return ss/den

    def fit(self, exog: np.ndarray, endog: np.ndarray) -> RegressionResult:
        self.initializeParams(exog.shape)
        diff = self.diffThreshold + 1
        iters = 0
        res = RegressionResult(diff)
        while (diff > self.diffThreshold) and (iters < self.maxIters):
            diff = self.gradAscent(exog, endog)
            iters += 1
            res.apppendDiff(diff)
        self.logger.info("Iterations: %d, final residual: %f", iters, diff)
        res.addParams(self.params, self.transitionProb)
        res.smoothed_marginal_probabilities = self.cumProb
        res.transitionProb = self.transitionProb
        return res

    #def fit(self, exog: np.ndarray, endog: np.ndarray) -> RegressionResult
    #    scipy.optimize.minimize(func, x0, jac, method="BFGS")