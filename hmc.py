import torch

def _helper(netG, x_tilde, eps, sigma):
    eps = eps.clone().detach().requires_grad_(True)
    with torch.no_grad():
        G_eps = netG(eps)
    bsz = eps.size(0)
    log_prob_eps = (eps ** 2).view(bsz, -1).sum(1).view(-1, 1)
    log_prob_x = (x_tilde - G_eps)**2 / sigma**2
    log_prob_x = log_prob_x.view(bsz, -1)
    log_prob_x = torch.sum(log_prob_x, dim=1).view(-1, 1)
    logjoint_vect = -0.5 * (log_prob_eps + log_prob_x)
    logjoint_vect = logjoint_vect.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = eps.grad
    return logjoint_vect, logjoint, grad_logjoint

def get_samples(netG, x_tilde, eps_init, sigma, burn_in, num_samples_posterior, 
            leapfrog_steps, stepsize, flag_adapt, hmc_learning_rate, hmc_opt_accept):
    device = eps_init.device
    bsz, eps_dim = eps_init.size(0), eps_init.size(1)
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, eps_dim).to(device)
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        eps = current_eps
        p = torch.randn_like(current_eps)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, current_eps, sigma)
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            eps = eps + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, eps, sigma)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, eps, sigma)  
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).sum(dim=1) 
        current_K = current_K.view(-1, 1) ## should be size of B x 1 
        proposed_K = 0.5 * (p**2).sum(dim=1) 
        proposed_K = proposed_K.view(-1, 1) ## should be size of B x 1 
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K)) 
        accept = accept.float().squeeze() ## should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try: 
            len(ind) > 0
            current_eps[ind, :] = eps[ind, :]  
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        else:
            samples[cnt*bsz : (cnt+1)*bsz, :] = current_eps.squeeze()
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize
