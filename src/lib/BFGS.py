import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


class BFGS(object):
    ''' BFGS algorithm for minimizing an unconstrained nonlinear function
    min f(x) where x is n dimensional, R^n
    '''

    def __init__(self, ndimension, deriv_func, second_deriv_func, gradient_func, maxiters=1000, threshold=1E-4):
        self.nDim = ndimension
        self.deriv = deriv_func
        self.secondDeriv = second_deriv_func
        self.gradient = gradient_func
        self.maxIters = maxiters
        self.threshold = threshold
        assert callable(gradient_func), "Gradient function needs to be callable"
        assert callable(deriv_func), "Derivative function needs to be callable"
        assert callable(second_deriv_func), "Second derivative function needs to be callable"
        self.logger = logging.getLogger(self.__class__.__name__)

    def lineMinimize(self, x, direction):
        iter_count = 0
        alpha = 0
        x_new = x.copy()
        change = self.threshold + 1
        while (iter_count < self.maxIters) or (abs(change) < self.threshold):
            x_new[:] = x + alpha * direction
            deriv = self.deriv(x_new)
            second_deriv = self.secondDeriv(x_new)
            if abs(second_deriv) < 1E-10:
                break
            change = -deriv/second_deriv
            alpha += change
            iter_count += 1
        return alpha

    def updateInverseHessian(self, hess_inv, increment, grad_change):
        '''
        B_{k+1}^{-1} = B_k^{-1} + \frac{(s_k^Ty_k + y_k^T B_k^{-1}y_k)s_k s_k^T}{s_k^Ty_k} - \frac{B_k^{-1}y_ks_k^{T} +
        s_ky_k^TB_k^{-1}}{s_k^Ty_k}
        s_k = \alpha_k p_k = increment
        y_k = grad_change
        '''
        den = np.dot(increment, grad_change)
        n12 = np.einsum("i,ij,j->", grad_change, hess_inv, grad_change)
        m1 = np.einsum("i,j->ij", increment, increment)
        n2 = np.einsum("ij,j,k->ik", hess_inv, grad_change, increment) + np.einsum("i,j,jk->ik", increment, grad_change, hess_inv)
        hess_inv[:] = hess_inv + (den + n12)/(den * den) * m1 - n2/den

    def solve(self):
        change = 1.0
        x = np.random.random(self.nDim)
        x_new = np.ndarray(self.nDim, dtype=np.float32)
        inverse_hessian = np.eye(self.nDim)
        grad = np.ndarray(self.nDim, dtype=np.float32)
        self.gradient(x, grad)
        grad_new = np.ndarray(self.nDim, dtype=np.float32)
        grad_change = np.ndarray(self.nDim, dtype=np.float32)
        for i in range(self.maxIters):
            if change <= self.threshold:
                break
            # hessian * p = -gradient
            direction = -np.einsum("ij,j->i", inverse_hessian, grad)
            # minimize function along p
            alpha = self.lineMinimize(x, direction)
            # calculate change in x as increment
            increment = alpha * direction
            change = increment * increment
            change = np.sqrt(np.sum(change)/self.nDim)
            x_new[:] = np.add(x, increment)
            # grad_change = grad_new - grad
            self.gradient(x_new, grad_new)
            grad_change[:] = np.subtract(grad_new, grad)
            # update hessian
            self.updateInverseHessian(inverse_hessian, increment, grad_change)
            # update x
            x[:] = x_new
            # update gradient
            grad[:] = grad_new

        self.logger.info("Took %d iterations, with final diff %f" % (i, change))
        return x