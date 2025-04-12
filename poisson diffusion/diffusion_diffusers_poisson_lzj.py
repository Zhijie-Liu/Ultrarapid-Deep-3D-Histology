import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import core_lzj


def tensor2show16bit(img):
    img1 = img[0][0].detach().cpu().numpy()
    img2 = ((img1 + 1) / 2 * 65535)
    # img2[img2 < 0] = 0
    # img2[img2 > 255] = 255
    img3 = img2.astype(np.uint16)
    img4 = Image.fromarray(img3)

    return img4


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    re = out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    return re


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # noise = torch.randn_like(x_0)
        lambda_value = 10
        poisson_noise = np.random.poisson(lambda_value, size=x_0.size()).astype(np.float32)
        mean_poisson = np.mean(poisson_noise)
        std_poisson = np.std(poisson_noise)
        poisson_noise_standardized = (poisson_noise - mean_poisson) / std_poisson
        noise = torch.from_numpy(poisson_noise_standardized).to(x_0.device)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t)[0], noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = []
        # posterior_log_var_clipped = extract(
        #     self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)[0]
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)[0]
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)[0]
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            del eps

        else:
            raise NotImplementedError(self.mean_type)
        # x_0 = torch.clip(x_0, -1., 1.)

        torch.cuda.empty_cache()
        return model_mean, model_log_var

    def forward(self, x_T, t_current):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(t_current)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                # noise = torch.randn_like(x_t)
                lambda_value = 10
                poisson_noise = np.random.poisson(lambda_value, size=x_t.size()).astype(np.float32)
                mean_poisson = np.mean(poisson_noise)
                std_poisson = np.std(poisson_noise)
                poisson_noise_standardized = (poisson_noise - mean_poisson) / std_poisson
                noise = torch.from_numpy(poisson_noise_standardized).to(x_t.device)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
            # img = tensor2show16bit(torch.clip(x_t, -1, 1))
            # core_lzj.check_folder_existence('step')
            # img.save(os.path.join('step', time_step.__str__() + '.tif'))
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class DDPMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas_t', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer('alphas_t', 1. - self.betas_t)
        self.register_buffer('alphas_t_bar', torch.cumprod(self.alphas_t, dim=0))
        self.register_buffer('alphas_prev_bar', F.pad(self.alphas_t_bar[:-1], (1, 0), value=1.0))

    def sample_one_step(self, x_t, time_step: int):
        """
        Calculate $x_{t-1}$ according to $x_t$
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        epsilon = self.model(x_t, t)[0]
        mean = extract(torch.sqrt(1.0 / self.alphas_t), t, x_t.shape) * x_t - extract((1.0 - self.alphas_t) / torch.sqrt(self.alphas_t * (1.0 - self.alphas_t_bar)), t, x_t.shape) * epsilon
        var = extract((1.0 - self.alphas_t) * (1.0 - self.alphas_prev_bar) / (1.0 - self.alphas_t_bar), t, x_t.shape)
        # mean, var = self.cal_mean_variance(x_t, t)
        lambda_value = 10
        poisson_noise = np.random.poisson(lambda_value, size=x_t.size()).astype(np.float32)
        mean_poisson = np.mean(poisson_noise)
        std_poisson = np.std(poisson_noise)
        poisson_noise_standardized = (poisson_noise - mean_poisson) / std_poisson
        z = torch.from_numpy(poisson_noise_standardized).to(x_t.device) if time_step > 0 else 0

        # z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        return x_t_minus_one

    def forward(self, x_t, t_current):

        x = [x_t]
        with tqdm(reversed(range(t_current)), colour="#6565b5", total=t_current) as sampling_steps:
            for time_step in sampling_steps:
                x_t = self.sample_one_step(x_t, time_step)
        return torch.clip(x_t, -1, 1)


class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas_t', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer('alphas_t', 1. - self.betas_t)
        self.register_buffer('alphas_t_bar', torch.cumprod(self.alphas_t, dim=0))
        # self.register_buffer('alphas_prev_bar', F.pad(self.alphas_t_bar[:-1], (1, 0), value=1.0))

    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        """
        Calculate $x_{t-1}$ according to $x_t$
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t_bar = extract(self.alphas_t_bar, t, x_t.shape)
        alpha_prev_bar = extract(self.alphas_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_t = self.model(x_t, t)[0]

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_prev_bar) / (1 - alpha_t_bar) * (1 - alpha_t_bar / alpha_prev_bar))
        # z = torch.randn_like(x_t)
        lambda_value = 10
        poisson_noise = np.random.poisson(lambda_value, size=x_t.size()).astype(np.float32)
        mean_poisson = np.mean(poisson_noise)
        std_poisson = np.std(poisson_noise)
        poisson_noise_standardized = (poisson_noise - mean_poisson) / std_poisson
        z = torch.from_numpy(poisson_noise_standardized).to(x_t.device) if prev_time_step > 0 else 0
        # z = torch.randn_like(x_t) if prev_time_step > 0 else 0
        x_t_minus_one = (
                torch.sqrt(alpha_prev_bar / alpha_t_bar) * x_t +
                (torch.sqrt(1 - alpha_prev_bar - sigma_t ** 2) - torch.sqrt(
                    (alpha_prev_bar * (1 - alpha_t_bar)) / alpha_t_bar)) * epsilon_t +
                sigma_t * z
        )
        return x_t_minus_one

    def forward(self, x_t, t_current, steps, method="linear", eta=0.0):
        if t_current!=0:
            if method == "linear":
                a = t_current // steps - 1
                b = t_current - 1
                # time_steps = np.asarray(list(range(0, t_current, a)))
                time_steps = (np.linspace(a, b, steps)).astype(np.int32)
                time_steps_prev = np.concatenate([[0], time_steps[:-1]])
                with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
                    for i in sampling_steps:
                        x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)
                        img = tensor2show16bit(torch.clip(x_t, -1, 1))
                        core_lzj.check_folder_existence('step v2')
                        img.save(os.path.join('step v2', time_steps[i].__str__() + '.tif'))
            elif method == "interval":
                time_steps = np.flipud(np.arange(t_current - 1, 0, -steps))
                time_steps_prev = np.concatenate([[0], time_steps[:-1]])
                with tqdm(reversed(range(0, time_steps_prev.__len__())), colour="#6565b5", total=time_steps_prev.__len__()) as sampling_steps:
                    for i in sampling_steps:
                        x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)
                        # img = tensor2show16bit(torch.clip(x_t, -1, 1))
                        # core_lzj.check_folder_existence('step v2')
                        # img.save(os.path.join('step v2', sampling_steps.__str__() + '.tif'))
            else:
                raise NotImplementedError(f"sampling method {method} is not implemented!")

            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            # time_steps = time_steps + 1
            # previous sequence



        return torch.clip(x_t, -1, 1)

def poisson_noise_torch(size):
    lambda_value = 10
    poisson_noise = np.random.poisson(lambda_value, size=size).astype(np.float32)
    mean_poisson = np.mean(poisson_noise)
    std_poisson = np.std(poisson_noise)
    poisson_noise_standardized = (poisson_noise - mean_poisson) / std_poisson
    x0 = torch.from_numpy(poisson_noise_standardized)
    return x0



