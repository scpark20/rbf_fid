import os
import torch
import torch.nn.functional as F
import numpy as np
from .sampler import expand_dims

class RBFSolverQuadSweep:
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
        K = torch.exp(-beta**2 * (lambdas - lambdas.T) ** 2)
        return K

    # def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
    #     from scipy.integrate import quad

    #     # Handle the zero-beta case.
    #     if beta == 0:
    #         if self.predict_x0:
    #             return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
    #         else:    
    #             return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)

    #     vals = []
    #     for lambda_u in lambdas:    
    #         def integrand(lmbd):
    #             return np.exp(lmbd - beta**2 * (lmbd - lambda_u)**2)
    #         val, _ = quad(integrand, float(lambda_s), float(lambda_t))
    #         vals.append(val)

    #     return torch.Tensor(vals, device=lambdas.device)

    # def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
    #     from scipy.integrate import fixed_quad

    #     # Handle the zero-beta case.
    #     if beta == 0:
    #         if self.predict_x0:
    #             return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
    #         else:    
    #             return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)
        
    #     lambda_s = float(lambda_s)
    #     lambda_t = float(lambda_t)
    #     lambdas = lambdas.numpy()
    #     beta = float(beta)

    #     vals = []
    #     for lambda_u in lambdas:
    #         def integrand(lmbd):
    #             return np.exp(lmbd - float(beta)**2 * (lmbd - float(lambda_u))**2)
    #         val, _ = fixed_quad(integrand, lambda_s, lambda_t, n=64)
    #         vals.append(val)

    #     return torch.Tensor(vals, device=lambdas.device)

    # def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
    #     from scipy.integrate import simpson

    #     # Handle the zero-beta case.
    #     if beta == 0:
    #         if self.predict_x0:
    #             return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
    #         else:    
    #             return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)
        
    #     lambda_s = float(lambda_s)
    #     lambda_t = float(lambda_t)
    #     lambdas = lambdas.numpy()
    #     beta = float(beta)

    #     def integrand(lmbd, lambda_u):
    #         return np.exp(lmbd - float(beta)**2 * (lmbd - float(lambda_u))**2)               

    #     x_vals = np.linspace(lambda_s, lambda_t, 500)
    #     vals = []
    #     for lambda_u in lambdas:
    #         val = simpson(integrand(x_vals, lambda_u), x_vals)
    #         vals.append(val)

    #     return torch.Tensor(vals, device=lambdas.device)
    
    def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
        from scipy.integrate import quad

        # Handle the zero-beta case.
        if beta == 0:
            if self.predict_x0:
                return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
            else:    
                return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)
            
        h = lambda_t - lambda_s
        s = 1/(beta*h)
        vals = []
        for lambda_u in lambdas:
            r_u = (lambda_u - lambda_s) / h
            def integrand(r):
                return np.exp((r-1)*h - ((r-r_u)/s)**2)
            val, _ = quad(integrand, 0, 1)
            vals.append(np.exp(lambda_t)*h*val)

        return torch.Tensor(vals, device=lambdas.device)

    def get_coefficients(self, lambda_s, lambda_t, lambdas, beta):
        p = len(lambdas)
        # (p,)
        integral1 = self.get_integral_vector(lambda_s, lambda_t, lambdas, beta)
        #print('integral1 :', lambda_s, beta, integral1)
        # (1,)
        integral2 = self.get_integral_vector(lambda_s, lambda_t, lambdas[:1], beta=0)
        
        # (p+1,)
        integral_aug = torch.cat([integral1, integral2], dim=0)

        # (p, p)
        kernel = self.get_kernel_matrix(lambdas, beta)
        eye = torch.eye(p+1, device=kernel.device)
        kernel_aug = 1 - eye
        kernel_aug[:p, :p] = kernel
        # (p,)
        coefficients = (integral_aug[None, :] @ torch.linalg.inv(kernel_aug))[0, :p]    
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
        beta = steps / (np.exp(log_scale) * abs(lambdas[-1] - lambdas[0]))
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

    def sample_by_target_matching(self, x, target, steps, skip_type='logSNR', order=3, log_scale_min=-6.0, log_scale_max1=1.0, log_scale_max2=6.0, log_scale_num=100, exp_num=0):
        lower_order_final = True  # 전체 스텝이 매우 작을 때 마지막 스텝에서 차수를 낮춰서 안정성 확보할지.

        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        assert t_0 > 0 and t_T > 0, "Time range( t_0, t_T )는 0보다 커야 함. (Discrete DPMs: [1/N, 1])"

        # 텐서가 올라갈 디바이스 설정
        device = x.device
        x_original = x

        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            lambdas = torch.Tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps])
            signal_rates = torch.Tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps])
            noise_rates = torch.Tensor([self.noise_schedule.marginal_std(t) for t in timesteps])

            log_scale_max_list = np.arange(log_scale_max1, log_scale_max2+1.0, 1.0)
            x_list = []
            recon_list = []
            optimal_log_scales_list = []
            for log_scale_max in log_scale_max_list:
                x = x_original
                log_scales = np.linspace(log_scale_min, log_scale_max, log_scale_num)
                optimal_log_scales_p = []
                optimal_log_scales_c = []

                hist = [None for _ in range(steps)]
                hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
                
                for i in range(0, steps):
                    if lower_order_final:
                        p = min(i+1, steps - i, order)
                    else:
                        p = min(i+1, order)
                        
                    # ===predictor===
                    pred_losses = []
                    for log_scale in log_scales:
                        loss = self.get_loss_by_target_matching(i, steps, target, hist, log_scale, lambdas, p, corrector=False)
                        pred_losses.append(loss)

                    optimal_log_scale = log_scales[torch.stack(pred_losses).argmin()]
                    optimal_log_scales_p.append(optimal_log_scale)
                    beta = steps / (np.exp(optimal_log_scale) * abs(lambdas[-1] - lambdas[0]))
                    x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                                p=p, beta=beta, corrector=False)
                    
                    if i == steps - 1:
                        x = x_pred
                        break
                    
                    # predictor로 구한 x_pred를 이용해서 model_fn 평가
                    hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                    
                    # ===corrector===
                    corr_losses = []
                    for log_scale in log_scales:
                        loss = self.get_loss_by_target_matching(i, steps, target, hist, log_scale, lambdas, p, corrector=True)
                        corr_losses.append(loss)

                    optimal_log_scale = log_scales[torch.stack(corr_losses).argmin()]
                    optimal_log_scales_c.append(optimal_log_scale)
                    beta = steps / (np.exp(optimal_log_scale) * abs(lambdas[-1] - lambdas[0]))
                    x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                                p=p, beta=beta, corrector=True)
                    x = x_corr
                
                x_list.append(x.detach())
                recon_list.append(F.mse_loss(x, target))
                optimal_log_scales_p = np.array(optimal_log_scales_p)
                optimal_log_scales_c = np.array(optimal_log_scales_c + [0.0])
                optimal_log_scales = np.stack([optimal_log_scales_p, optimal_log_scales_c], axis=0)
                optimal_log_scales_list.append(optimal_log_scales)

        min_index = torch.stack(recon_list).argmin()
        x = x_list[min_index]
        optimal_log_scales = optimal_log_scales_list[min_index]

        if self.scale_dir is not None:
            save_file = os.path.join(self.scale_dir, f'NFE={steps},p={order},exp_num={exp_num}.npy')
            np.save(save_file, optimal_log_scales)
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
            lambdas = torch.Tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps])
            signal_rates = torch.Tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps])
            noise_rates = torch.Tensor([self.noise_schedule.marginal_std(t) for t in timesteps])
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)

                # ===predictor===
                s = log_scale if log_scales is None else log_scales[0, i]
                beta = steps / (np.exp(s) * abs(lambdas[-1] - lambdas[0]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                s = log_scale if log_scales is None else log_scales[1, i]
                beta = steps / (np.exp(s) * abs(lambdas[-1] - lambdas[0]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=True)
                x = x_corr
        # 최종적으로 x를 반환
        return x