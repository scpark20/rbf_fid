import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.integrate import quad
from .sampler import expand_dims
import matplotlib.pyplot as plt

class RBFSolverClosed:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="data_prediction",
            correcting_x0_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
            scale_dir=None,
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
        K = torch.exp(-beta**2 * (lambdas - lambdas.T)**2)
        return K

    def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
        # Handle the zero-beta case.
        if beta == 0:
            if self.predict_x0:
                return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
            else:    
                return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)

        def log_erf_diff(a, b):
            return torch.log(torch.erfc(b)) + torch.log(1.0-torch.exp(torch.log(torch.erfc(a)) - torch.log(torch.erfc(b))))
        
        lambda_s = lambda_s.to(dtype=torch.float64)
        lambda_t = lambda_t.to(dtype=torch.float64)
        lambdas  = lambdas.to(dtype=torch.float64)
        beta     = beta.to(dtype=torch.float64)
    
        h = lambda_t - lambda_s
        s = 1/(beta*h)
        r_u = (lambdas - lambda_s) / h
        
        log_prefactor = lambda_t + torch.log(h) + ((s*h)**2/4 + h*(r_u-1)) + torch.log(0.5*np.sqrt(np.pi)*s)
        upper = (r_u + s**2*h/2)/s
        lower = (r_u + s**2*h/2 - 1)/s
        ret = torch.exp(log_prefactor + log_erf_diff(upper, lower))
        
        # if torch.isnan(ret).any():
        #     print('integral1 :', lambda_s, lambda_t, torch.log(s), ret)
        return ret.to(dtype=torch.float32)

    def get_coefficients(self, lambda_s, lambda_t, lambdas, beta):
        p = len(lambdas)
        # (p,)
        integral1 = self.get_integral_vector(lambda_s, lambda_t, lambdas, beta)
        # (1,)
        integral2 = self.get_integral_vector(lambda_s, lambda_t, lambdas[:1], beta=torch.zeros_like(beta))
        
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

    def sample_by_target_matching(self, x, target, steps, skip_type='logSNR', order=3, log_scale_min=-6.0, log_scale_max=6.0, log_scale_num=100, optim_lr=1e-1, optim_steps=100):
        # x : start noise to sample
        # target : target image to sample
        # log_scale_max : absolute value of log scale to search
        # log_scale_num : # of log scale in [-log_scale, log_scale] to search
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
        log_scales = torch.linspace(log_scale_min, log_scale_max, log_scale_num, device=device)
        optimal_log_scales_p = []
        optimal_log_scales_c = []

        hist = [None for _ in range(steps)]
        with torch.no_grad():
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
        
        for i in range(0, steps):
            if lower_order_final:
                p = min(i+1, steps - i, order)
            else:
                p = min(i+1, order)
                
            # ===predictor===
            # Line Search
            with torch.no_grad():
                pred_losses = []
                for log_scale in log_scales:
                    loss = self.get_loss_by_target_matching(i, steps, target, hist, log_scale, lambdas, p, corrector=False)
                    pred_losses.append(loss)

            # plt.plot(log_scales.data.data.cpu().numpy(), torch.stack(pred_losses).data.cpu().numpy())
            # plt.show()
            optimal_log_scale_p = log_scales[torch.stack(pred_losses).argmin()]
            optimal_log_scales_p.append(optimal_log_scale_p.item())

            # # Gradient Descent
            # log_scale_p = torch.nn.Parameter(torch.tensor(log_scale_min, device=device), requires_grad=True)
            # optimizer_p = torch.optim.Adam([log_scale_p], lr=optim_lr)
            # for _ in range(optim_steps):
            #     optimizer_p.zero_grad()
            #     loss_p = self.get_loss_by_target_matching(i, steps, target, hist, log_scale_p, lambdas, p, corrector=False)
            #     loss_p.backward()
            #     optimizer_p.step()
                
            # plt.figure(figsize=(10, 3))
            # plt.title('Predictor log-scale')
            # plt.plot(log_scales.data.cpu().numpy(), torch.stack(pred_losses).data.cpu().numpy(), label="Loss over log_scales")
            # plt.axvline(x=optimal_log_scale_p.item(), color='g', linestyle='-.', label="Line search best log_scale")
            # plt.axvline(x=log_scale_p.item(), color='r', linestyle='--', label="Adam-based log_scale")
            # plt.ylim([0, 3e-1])
            # plt.xlabel("log_scale")
            # plt.ylabel("MSE Loss")
            # plt.grid()
            # plt.legend()
            # plt.show()        
            
            with torch.no_grad():
                beta = steps / (torch.exp(optimal_log_scale_p) * abs(lambdas[-1] - lambdas[0]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                                p=p, beta=beta, corrector=False)
                
            if i == steps - 1:
                x = x_pred
                break
            
            with torch.no_grad():
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
            
            # ===corrector===
            # Line Search
            with torch.no_grad():
                corr_losses = []
                for log_scale in log_scales:
                    loss = self.get_loss_by_target_matching(i, steps, target, hist, log_scale, lambdas, p, corrector=True)
                    corr_losses.append(loss)
            optimal_log_scale_c = log_scales[torch.stack(corr_losses).argmin()]
            optimal_log_scales_c.append(optimal_log_scale_c.item())

            # # Gradient Descent
            # log_scale_c = torch.nn.Parameter(torch.tensor(log_scale_min, device=device), requires_grad=True)
            # optimizer_c = torch.optim.Adam([log_scale_c], lr=optim_lr)
            # for _ in range(optim_steps):
            #     optimizer_c.zero_grad()
            #     loss_c = self.get_loss_by_target_matching(i, steps, target, hist, log_scale_c, lambdas, p, corrector=True)
            #     loss_c.backward()
            #     optimizer_c.step()
                
            # plt.figure(figsize=(10, 3))
            # plt.title('Corrector log-scale')
            # plt.plot(log_scales.data.cpu().numpy(), torch.stack(corr_losses).data.cpu().numpy(), label="Loss over log_scales")
            # plt.axvline(x=optimal_log_scale_c.item(), color='g', linestyle='-.', label="Line search best log_scale")
            # plt.axvline(x=log_scale_c.item(), color='r', linestyle='--', label="Adam-based log_scale")
            # plt.ylim([0, 3e-1])
            # plt.xlabel("log_scale")
            # plt.ylabel("MSE Loss")
            # plt.grid()
            # plt.legend()
            # plt.show()

            with torch.no_grad():
                beta = steps / (torch.exp(optimal_log_scale_c) * abs(lambdas[-1] - lambdas[0]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                                p=p, beta=beta, corrector=True)
                x = x_corr
    
        optimal_log_scales_p = np.array(optimal_log_scales_p)
        optimal_log_scales_c = np.array(optimal_log_scales_c + [0.0])
        optimal_log_scales = np.stack([optimal_log_scales_p, optimal_log_scales_c], axis=0)

        if self.scale_dir is not None:
            optimal_file = os.path.join(self.scale_dir, f'NFE={steps},p={order}.npy')
            np.save(optimal_file, optimal_log_scales)
            print(optimal_file, ' saved!')

        return x, optimal_log_scales

    def sample_by_target_matching_grad(self, x, target, steps, skip_type='logSNR', order=3, log_scale_min=-6.0, log_scale_max=6.0, optim_lr=1e-1, optim_steps=100):
        # x : start noise to sample
        # target : target image to sample
        # log_scale_max : absolute value of log scale to search
        # log_scale_num : # of log scale in [-log_scale, log_scale] to search

        log_scale_min = -log_scale_max
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
        optimal_log_scales_p = []
        optimal_log_scales_c = []

        hist = [None for _ in range(steps)]
        with torch.no_grad():
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
        
        for i in range(0, steps):
            if lower_order_final:
                p = min(i+1, steps - i, order)
            else:
                p = min(i+1, order)
                
            # ===predictor===
            # Gradient Descent
            log_scale_p = torch.nn.Parameter(torch.tensor(log_scale_min, device=device), requires_grad=True)
            optimizer_p = torch.optim.Adam([log_scale_p], lr=optim_lr)
            for _ in range(optim_steps):
                optimizer_p.zero_grad()
                loss_p = self.get_loss_by_target_matching(i, steps, target, hist, log_scale_p, lambdas, p, corrector=False)
                loss_p.backward()
                optimizer_p.step()
            #log_scale_p = log_scale_p.detach().clamp(log_scale_min, log_scale_max)
            optimal_log_scales_p.append(log_scale_p.item())
            #print('log_scale_p :', log_scale_p.item())

            with torch.no_grad():
                beta = steps / (torch.exp(log_scale_p) * abs(lambdas[-1] - lambdas[0]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                                p=p, beta=beta, corrector=False)
                
            if i == steps - 1:
                x = x_pred
                break
            
            with torch.no_grad():
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
            
            # ===corrector===
            # Gradient Descent
            log_scale_c = torch.nn.Parameter(torch.tensor(log_scale_min, device=device), requires_grad=True)
            optimizer_c = torch.optim.Adam([log_scale_c], lr=optim_lr)
            for _ in range(optim_steps):
                optimizer_c.zero_grad()
                loss_c = self.get_loss_by_target_matching(i, steps, target, hist, log_scale_c, lambdas, p, corrector=True)
                loss_c.backward()
                optimizer_c.step()
            #log_scale_c = log_scale_c.detach().clamp(log_scale_min, log_scale_max)
            optimal_log_scales_c.append(log_scale_c.item())
            #print('log_scale_c :', log_scale_c.item())
                
            with torch.no_grad():
                beta = steps / (torch.exp(log_scale_c) * abs(lambdas[-1] - lambdas[0]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                                p=p, beta=beta, corrector=True)
                x = x_corr

        optimal_log_scales_p = np.array(optimal_log_scales_p)
        optimal_log_scales_c = np.array(optimal_log_scales_c + [0.0])
        optimal_log_scales = np.stack([optimal_log_scales_p, optimal_log_scales_c], axis=0)

        if self.scale_dir is not None:
            optimal_file = os.path.join(self.scale_dir, f'NFE={steps},p={order}.npy')
            np.save(optimal_file, optimal_log_scales)
            print(optimal_file, ' saved!')

        return x, optimal_log_scales

    def load_optimal_log_scales(self, NFE, order):
        try:
            load_file = os.path.join(self.scale_dir, f'NFE={NFE},p={order}.npy')
            log_scales = np.load(load_file)
        except:
            return None
        #print(load_file, 'loaded!')
        return log_scales
    
    def sample(self, x, steps, skip_type='logSNR', order=3, log_scale=1.0):
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
                s = log_scale if log_scales is None else log_scales[0, i]
                s = torch.tensor(s, device=device)
                beta = steps / (torch.exp(s) * abs(lambdas[-1] - lambdas[0]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                s = log_scale if log_scales is None else log_scales[1, i]
                s = torch.tensor(s, device=device)
                beta = steps / (torch.exp(s) * abs(lambdas[-1] - lambdas[0]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=True)
                x = x_corr
        # 최종적으로 x를 반환
        return x