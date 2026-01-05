import torch
import torch.nn as nn
import numpy as np

class NotearsLinear(nn.Module):
    def __init__(self, d, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
        super().__init__()
        self.d = d
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        
    def _loss(self, W, X):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if self.loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.size(0) * (R ** 2).sum()
            G_loss = - 1.0 / X.size(0) * X.t() @ R
        elif self.loss_type == 'logistic':
            loss = 1.0 / X.size(0) * (torch.log1p(torch.exp(M)) - X * M).sum()
            G_loss = 1.0 / X.size(0) * X.t() @ (torch.sigmoid(M) - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint."""
        # E = e^{W * W}
        # h(W) = Tr(E) - d = 0
        W_abs = torch.abs(W) # Ensure positive for cycle mixing
        
        # Matrix Exponential (Approximation or Exact)
        # For d=50, matrix exp is feasible.
        E = torch.matrix_exp(W_abs)
        h = torch.trace(E) - self.d
        return h

    def fit(self, X):
        # NOTEARS Augmented Lagrangian
        # X: (n, d)
        n, d = X.shape
        W_est = torch.zeros(d, d, dtype=torch.float32, requires_grad=True)
        
        rho, alpha, h = 1.0, 0.0, np.inf
        
        optimizer = torch.optim.LBFGS([W_est], max_iter=self.max_iter)
        
        for _ in range(self.max_iter):
            def closure():
                optimizer.zero_grad()
                W_est.data = torch.clamp(W_est.data, min=-10, max=10) # Clip for stability? No NOTEARS doesn't limit 
                
                loss, _ = self._loss(W_est, X)
                h_val = self._h(W_est)
                
                # Augmented Lagrangian
                obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + self.lambda1 * torch.abs(W_est).sum()
                obj.backward()
                return obj
            
            optimizer.step(closure)
            
            # Update Dual
            with torch.no_grad():
                h_new = self._h(W_est).item()
            
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
            alpha += rho * h_new
            h = h_new
            if h <= self.h_tol or rho >= self.rho_max:
                break
                
        # Thresholding
        W_final = W_est.detach().numpy()
        W_final[np.abs(W_final) < self.w_threshold] = 0
        return W_final
