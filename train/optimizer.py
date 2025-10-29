import torch
import torch.nn as nn

class Adam:
    """
    Implements Adam optimizer.
    Adam = momentum + RMSProp -> keep two moving averages: momentum mt (mean of grads)
    and velocity vt (mean of square of grads).

    m_t = b1*m_t-1 + (1-b1)*gt (tracks direction of gradient)
    v_t = b2*v_t-1 + (1-b2)*gt**2 (tracks magnitude)

    gradient update:
    theta_t = theta_t-1 - lr* (m_t/(sqrt(v_t) + eps))
    """
    def __init__(self, params, lr:float, beta1:float=0.9, beta2:float=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = list(params)

        self.state = {} #state has m, v buffers per param
        self.t = 0 #step counter
    
    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue
            
            grad = p.grad
            cur_state = self.state.setdefault(p, {})
            m = cur_state.setdefault('m', torch.zeros_like(p))
            v = cur_state.setdefault('v', torch.zeros_like(p))

            #in place multiplication and addition for efficiency
            m.mul_(self.beta1).add_(grad, alpha=1-self.beta1)
            v.mul_(self.beta2).add_(grad**2, alpha=1-self.beta2)

            #de-bias m, v since we start with 0
            m_hat = m/(1-self.beta1**self.t)
            v_hat = v/(1-self.beta2**self.t)

            with torch.no_grad():
                # p_new = p - lr*(m_hat/sqrt(v_hat))
                p.addcdiv_(m_hat, v_hat.sqrt().add_(1e-8), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None #better to set to None than to zero the gradients out as it frees up memory!

if __name__ == "__main__":
    # simple test
    x = torch.tensor([5.0], requires_grad=True)
    opt = Adam([x], lr=0.5)

    for _ in range(10):
        loss = (x - 2) ** 2
        loss.backward()
        opt.step()
        x.grad.zero_()
        print(f"x = {x.item():.4f}, loss = {loss.item():.4f}")








