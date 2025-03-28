import os
import torch
import torch.nn.functional as F
import numpy as np
from .sampler import expand_dims

# Separated Beta
class RBFSolverGLQ10Sepbeta:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="data_prediction",
            correcting_x0_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
            scale_dir=None,
            exp_num=0
    ):

        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.predict_x0 = algorithm_type == "data_prediction"
        self.scale_dir = scale_dir
        self.exp_num = exp_num

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """

        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)
    
    def get_kernel_matrix(self, lambdas, beta):
        # (p, 1)
        lambdas = lambdas[:, None]
        # (p, p)
        K = torch.exp(-beta[None, :]**2 * (lambdas - lambdas.T) ** 2)
        return K
    
    def get_integral_vector_beta0(self, lambda_s, lambda_t, lambdas):
        if self.predict_x0:
            return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
        return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)

    def get_integral_vector_closed_form(self, lambda_s, lambda_t, lambdas, beta):
        h = lambda_t - lambda_s
        s = 1/(beta*h)
        
        def log_erf_diff(a, b):
                return torch.log(torch.erfc(b)) + torch.log(1.0-torch.exp(torch.log(torch.erfc(a)) - torch.log(torch.erfc(b))))
    
        r_u = (lambdas - lambda_s) / h
        log_prefactor = lambda_t + torch.log(h) + ((s*h)**2/4 + h*(r_u-1)) + torch.log(0.5*np.sqrt(np.pi)*s)
        upper = (r_u + s**2*h/2)/s
        lower = (r_u + s**2*h/2 - 1)/s
        result = torch.exp(log_prefactor + log_erf_diff(upper, lower))
        return result.float()

    def get_integral_vector_numerical(self, lambda_s, lambda_t, lambdas, beta):
        x = torch.tensor([
            -0.973906528517172,
            -0.865063366688985,
            -0.679409568299024,
            -0.433395394129247,
            -0.148874338981631,
            0.148874338981631,
            0.433395394129247,
            0.679409568299024,
            0.865063366688985,
            0.973906528517172,
            ], device=lambdas.device).to(torch.float64)
        w = torch.tensor([
            0.0666713443086881,
            0.149451349150581,
            0.219086362515982,
            0.269266719309996,
            0.295524224714753,
            0.295524224714753,
            0.269266719309996,
            0.219086362515982,
            0.149451349150581,
            0.0666713443086881,
            ], device=lambdas.device).to(torch.float64)
        
        def f1(lam):
            return torch.exp(lam - beta**2*(lam-lambdas[None, :])**2)
        def f2(lam):
            return (lambda_t-lambda_s)/2 * f1(lam*(lambda_t-lambda_s)/2 + (lambda_s+lambda_t)/2)
        
        # (1, p) = (1, n) @ (n, p)
        result = (w[None, :] @ f2(x[:, None]))[0]
        
        return result.float()

    def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
        #lambda_s, lambda_t, lambdas, beta = map(lambda x: x.double(), (lambda_s, lambda_t, lambdas, beta))

        mask0 = (beta == 0.0)
        safe_beta = torch.where(mask0, torch.ones_like(beta), beta)
        h = lambda_t - lambda_s
        s = 1.0 / (safe_beta * h)
        log_s = torch.log(s)
        mask_neg = (~mask0) & (log_s<0.0)
        mask_pos = (~mask0) & (log_s>=0.0)
        
        val0  = self.get_integral_vector_beta0(lambda_s, lambda_t, lambdas)
        val0 = val0.masked_fill(val0.isnan(), 0.)
        val_cf = self.get_integral_vector_closed_form(lambda_s, lambda_t, lambdas, beta)
        val_cf = val_cf.masked_fill(val_cf.isnan(), 0.)
        val_nm = self.get_integral_vector_numerical(lambda_s, lambda_t, lambdas, beta)
        val_nm = val_nm.masked_fill(val_nm.isnan(), 0.)
     
        result = (val0 * mask0.float()
               + val_cf * mask_neg.float()
               + val_nm * mask_pos.float())
        return result #.float()
        
                
    def get_coefficients(self, lambda_s, lambda_t, lambdas, beta):
        p = len(lambdas)
        # (p,)
        integral1 = self.get_integral_vector(lambda_s, lambda_t, lambdas, beta)
        # (1,)
        integral2 = self.get_integral_vector(lambda_s, lambda_t, lambdas[:1], beta=torch.zeros_like(lambdas[:1]))
        
        # (p+1,)
        integral_aug = torch.cat([integral1, integral2], dim=0)

        # (p, p)
        kernel = self.get_kernel_matrix(lambdas, beta)
        # (p+1, p+1)
        eye = torch.eye(p+1, device=kernel.device)
        kernel_aug = 1 - eye
        kernel_aug[:p, :p] = kernel
        # (p+1,)
        #coefficients = torch.linalg.solve(kernel_aug, integral_aug)
        coefficients = torch.linalg.pinv(kernel_aug) @ integral_aug
        # (p,)
        coefficients = coefficients[:p]
        return coefficients
    
    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, lambdas, p, beta, corrector=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        lambdas : [λ_0, λ_1, ...], scalar list
        beta : beta of RBF kernel
        corrector : True or False
        '''
        
        # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
        # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

        # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
        # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
        coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, beta)

        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        
        data_sum = sum([coeff * data for coeff, data in zip(coeffs, datas)])
        if self.predict_x0:
            next_sample = noise_rates[i+1]/noise_rates[i]*sample + noise_rates[i+1]*data_sum
        else:
            next_sample = signal_rates[i+1]/signal_rates[i]*sample - signal_rates[i+1]*data_sum
        return next_sample
    
    def get_loss_by_target_matching(self, i, steps, target, hist, log_scale, lambdas, p, corrector=False):
        beta = steps / (torch.exp(log_scale) * abs(lambdas[-1] - lambdas[0]))
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])
        coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, beta)
        
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        data_sum = sum([coeff * data for coeff, data in zip(coeffs, datas)])

        if self.predict_x0:
            integral = (torch.exp(lambdas[i+1]) - torch.exp(lambdas[i]))
        else:    
            integral = (torch.exp(-lambdas[i]) - torch.exp(-lambdas[i+1]))
        pred = data_sum / integral

        loss = F.mse_loss(target, pred)
        return loss

    def sample_by_target_matching(self, x, target, steps, skip_type='logSNR', order=3, log_scale_min=-6.0, log_scale_max=6.0, optim_lr=1e-2, optim_steps=1000, exp_num=0):
        lower_order_final = True  # 전체 스텝이 매우 작을 때 마지막 스텝에서 차수를 낮춰서 안정성 확보할지.

        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        assert t_0 > 0 and t_T > 0, "Time range( t_0, t_T )는 0보다 커야 함. (Discrete DPMs: [1/N, 1])"

        # 텐서가 올라갈 디바이스 설정
        device = x.device
        
        # 실제로 사용할 time step array를 구한다.
        # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
        lambdas = torch.tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps], device=device)
        signal_rates = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
        noise_rates = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

        optimal_log_scales = torch.zeros(2, steps, order+1)

        hist = [None for _ in range(steps)]
        hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
        
        grad_tolerance = 1e-4
        from tqdm import tqdm
        for i in tqdm(range(0, steps)):
            if lower_order_final:
                p = min(i+1, steps - i, order)
            else:
                p = min(i+1, order)
                
            # ===predictor===
            # random init
            lowest_loss = float('inf')
            best_init = None
            for init_step in range(10000):
                rand_init = torch.empty(p, device=device).uniform_(log_scale_min, log_scale_max)
                with torch.no_grad():
                    loss_val = self.get_loss_by_target_matching(i, steps, target, hist, rand_init, lambdas, p, corrector=False)
                if loss_val < lowest_loss:
                    lowest_loss = loss_val
                    best_init = rand_init.clone()

            log_scale_p = torch.nn.Parameter(best_init.clone(), requires_grad=True)
            #log_scale_p = torch.nn.Parameter(torch.full((p,), 0.0, device=device), requires_grad=True)
            optimizer_p = torch.optim.Adam([log_scale_p], lr=optim_lr)
            for optim_step in range(optim_steps):
                #print(optim_step, 'log_scale_p :', log_scale_p)
                optimizer_p.zero_grad()
                loss_p = self.get_loss_by_target_matching(i, steps, target, hist, log_scale_p, lambdas, p, corrector=False)
                loss_p.backward(retain_graph=True)

                grad_norm = log_scale_p.grad.data.norm()
                if torch.isnan(grad_norm):
                    #print(f"Gradient is NaN. Skipping step {optim_step}.")
                    break
                if grad_norm < grad_tolerance:
                    #print(f"Stopping early at step {optim_step} because grad norm < {grad_tolerance}")
                    break

                optimizer_p.step()
                log_scale_p.data.clamp_(log_scale_min, log_scale_max)
            log_scale_p = log_scale_p.detach()    
            optimal_log_scales[0, i, :p] = log_scale_p
            
            with torch.no_grad():
                beta = steps / (torch.exp(log_scale_p) * abs(lambdas[-1] - lambdas[0]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                            p=p, beta=beta, corrector=False)
                x_pred = x_pred.detach()
                
            if i == steps - 1:
                x = x_pred
                break
            
            with torch.no_grad():
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                 evaluation = self.model_fn(x_pred, timesteps[i+1])
                 hist[i+1] = evaluation.detach()
                
            # ===corrector===
            # random init
            lowest_loss = float('inf')
            best_init = None
            for init_step in range(10000):
                rand_init = torch.empty(p+1, device=device).uniform_(log_scale_min, log_scale_max)
                with torch.no_grad():
                    loss_val = self.get_loss_by_target_matching(i, steps, target, hist, rand_init, lambdas, p, corrector=True)
                if loss_val < lowest_loss:
                    lowest_loss = loss_val
                    best_init = rand_init.clone()

            log_scale_c = torch.nn.Parameter(best_init.clone(), requires_grad=True)
            #log_scale_c = torch.nn.Parameter(torch.full((p+1,), 0.0, device=device), requires_grad=True)
            optimizer_c = torch.optim.Adam([log_scale_c], lr=optim_lr)
            for optim_step in range(optim_steps):
                #print(optim_step, 'log_scale_c :', log_scale_c)
                optimizer_c.zero_grad()
                loss_c = self.get_loss_by_target_matching(i, steps, target, hist, log_scale_c, lambdas, p, corrector=True)
                loss_c.backward(retain_graph=True)

                grad_norm = log_scale_c.grad.data.norm()
                if torch.isnan(grad_norm):
                    #print(f"Gradient is NaN. Skipping step {optim_step}.")
                    break
                if grad_norm < grad_tolerance:
                    #print(f"Stopping early at step {optim_step} because grad norm < {grad_tolerance}")
                    break

                optimizer_c.step()
                log_scale_c.data.clamp_(log_scale_min, log_scale_max)
                
            log_scale_c = log_scale_c.detach()
            optimal_log_scales[1, i, :p+1] = log_scale_c

            with torch.no_grad():
                beta = steps / (torch.exp(log_scale_c) * abs(lambdas[-1] - lambdas[0]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                            p=p, beta=beta, corrector=True)
                x_corr = x_corr.detach()
            x = x_corr
            
        if self.scale_dir is not None:
            save_file = os.path.join(self.scale_dir, f'NFE={steps},p={order},exp_num={exp_num}.npy')
            np.save(save_file, optimal_log_scales.numpy())
            print(save_file, ' saved!')

        # 최종적으로 x를 반환
        return x, optimal_log_scales

    def load_optimal_log_scales(self, steps, order):
        try:
            load_file = os.path.join(self.scale_dir, f'NFE={steps},p={order},exp_num={self.exp_num}.npy')
            log_scales = np.load(load_file)
        except:
            return None
        print(load_file, 'loaded!')
        return log_scales
    
    def sample(self, x, steps, skip_type='logSNR', order=3, log_scale=0.0):
        # (2, steps, order+1)
        log_scales = self.load_optimal_log_scales(steps, order)
        lower_order_final = True  # 전체 스텝이 매우 작을 때 마지막 스텝에서 차수를 낮춰서 안정성 확보할지.

        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        assert t_0 > 0 and t_T > 0, "Time range( t_0, t_T )는 0보다 커야 함. (Discrete DPMs: [1/N, 1])"

        # 텐서가 올라갈 디바이스 설정
        device = x.device

        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            lambdas = torch.tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps], device=device)
            signal_rates = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
            noise_rates = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)

                # ===predictor===
                # (p,)
                s = torch.full((p,), log_scale, device=device) if log_scales is None else torch.tensor(log_scales[0, i][:p], device=device)
                beta = steps / (torch.exp(s) * abs(lambdas[-1] - lambdas[0]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                s = torch.full((p+1,), log_scale, device=device) if log_scales is None else torch.tensor(log_scales[1, i][:p+1], device=device)
                beta = steps / (torch.exp(s) * abs(lambdas[-1] - lambdas[0]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=True)
                x = x_corr
        # 최종적으로 x를 반환
        return x