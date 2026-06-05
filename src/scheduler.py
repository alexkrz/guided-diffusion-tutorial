# This code is from https://www.kaggle.com/code/vikramsandu/guided-diffusion-by-openai-from-scratch/notebook
# It is an adapted and simplified version from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py

import math

import numpy as np
import torch
import torch.nn.functional as F


def sample_timesteps(num_total_timesteps: int = 1000, num_sample_timesteps: int = 1000):
    """
    Uniformaly sample num_sample_timesteps timesteps given
    total number of timesteps.
    Later we will utilize for speed up sampling.
    """
    # Get New sampled timesteps (Uniform stepping)
    step_size = num_total_timesteps // num_sample_timesteps
    use_timesteps = np.arange(0, num_total_timesteps, step_size)

    # Make sure to append final timestep if not there already
    if (num_total_timesteps - 1) not in use_timesteps:
        use_timesteps = np.append(use_timesteps, num_total_timesteps - 1)

    return use_timesteps


# ---------------------------------------------------------------------------


def cosine_noise_schedular(
    num_original_timesteps: int = 1000, num_sampling_timesteps: int = 1000, max_beta: float = 0.999
) -> np.ndarray:
    """
    Calculate noise controlling parameter betas for all time steps
    according to the cosine noise scheduler.

    Parameters:
    num_original_timesteps (int): Number of original diffusion timesteps (used during training).
    num_sampling_timesteps (int): Number of timesteps used for sampling.
    max_beta (float): Maximum value for beta.

    Returns:
    array: Beta values for each timestep.
    array: timesteps used to calculate betas.

    NOTE: Set "num_sampling_timesteps" = "num_original_timesteps" during training.
    """
    # s is set to 0.008 by Authors
    s = 0.008

    # Calculate alpha bars
    alpha_bars = [
        math.cos(((t / num_original_timesteps + s) / (1 + s)) * (math.pi / 2)) ** 2
        for t in range(num_original_timesteps + 1)
    ]

    # Calculate betas
    betas = np.array([min(1 - a_t / a_tminus1, max_beta) for a_t, a_tminus1 in zip(alpha_bars[1:], alpha_bars[:-1])])

    # Modify Betas as described in the "Improved Sampling Speed" section above.
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    # Get New sampled timesteps (Uniform stepping)
    use_timesteps = sample_timesteps(num_original_timesteps, num_sampling_timesteps)

    # Calculate new betas according to new timesteps.
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod

    return np.array(new_betas), use_timesteps


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class GuidedDiffusionProcess:
    """
    Utilities for training and sampling diffusion model.
    Ported directly from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
    and modified for simplicity.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_sampling_timesteps: int = 1000,
        classifier_guidance: bool = False,  # True for Classifier Guidance during sampling only.
        classifier_scale=1.0,  # Weight given to the classifier gradient
    ):

        # Classifier Guidance.
        self.classifier_guidance = classifier_guidance
        self.classifier_scale = classifier_scale

        # Betas using Cosine Noise Schedular
        self.betas, self.use_timesteps = cosine_noise_schedular(num_timesteps, num_sampling_timesteps)
        self.betas = torch.tensor(self.betas, dtype=torch.float32)
        self.use_timesteps = torch.tensor(self.use_timesteps, dtype=torch.float32)

        # Diffusion Process Coefficients.
        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.one_minus_alphas_cumprod)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)

        # calculation for forward process "posterior" q(x_{t-1} | x_{t}).
        self.posterior_mean_coef1 = (self.sqrt_alphas_cumprod_prev * self.betas) / self.one_minus_alphas_cumprod
        self.posterior_mean_coef2 = (
            self.sqrt_alphas * (1.0 - self.alphas_cumprod_prev)
        ) / self.one_minus_alphas_cumprod
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev)) / (self.one_minus_alphas_cumprod)

        # Calculation for Variation Lower Bound Loss
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]))
        )

    def add_noise(self, x0, noise, t):
        """
        calculate x_t for all timesteps given input x0 using
        x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * epsilon
        where epsilon ~ N(0, 1). i.e., Forward Process

        Parameters:
        x0: Batch of Input Images of shape -> (B, C, H, W)
        noise: Batch of Noise sampled from N(0, 1) of shape -> (B, C, H, W)
        t: Batch of timesteps of shape -> (B, 1)

        Return:
        xt: Batch of output images at timesteps t obtained
        according to Diffusion Forward Process of shape -> (B, C, H, W)
        """
        # Mapping to Device + Broadcast
        sqrt_alphas_cumprod = self._coeff_broadcasting(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod = self._coeff_broadcasting(self.sqrt_one_minus_alphas_cumprod, t)

        # Return
        return (sqrt_alphas_cumprod * x0) + (sqrt_one_minus_alphas_cumprod * noise)

    def q_posterior_mean_variance(self, x0, xt, t):
        """
        Calculate Posterior Forward Mean and Variance of the
        Diffusion Process. i.e., q(x_(t-1)|x_t) Later we will use
        this in defining Variational Lower Bound (VLB) loss.

        Parameters:
        x0: Batch of Input Images of shape -> (B, C, H, W)
        xt: Batch of images at timesteps t -> (B, C, H, W)
        t: Batch of timesteps of shape -> (B, 1)

        Return:
        posterior_mean, posterior_variance, and posterior_log_variance_clipped
        of the shape -> (B, C, H, W)
        """

        posterior_mean_coef1 = self._coeff_broadcasting(self.posterior_mean_coef1, t)
        posterior_mean_coef2 = self._coeff_broadcasting(self.posterior_mean_coef2, t)
        posterior_variance = self._coeff_broadcasting(self.posterior_variance, t)
        posterior_log_variance_clipped = self._coeff_broadcasting(self.posterior_log_variance_clipped, t)

        # Posterior Mean
        posterior_mean = (posterior_mean_coef1 * x0) + (posterior_mean_coef2 * xt)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, y):
        """
        Calculation of Mean and Variance of p(x_(t-1)|x_t) using
        model noise prediction.

        Params
        --------
        model: U-net model which predicts the noise
        x: signal at time step t of shape -> (B, C, H, W)
        t: Batch of timesteps of shape -> (B, 1)
        y: Batch of class labels of shape -> (B, 1)

        Return
        --------
        mean: the model mean output
        variance: the model variance output
        log_variance: the log of variance
        x0: prediction of x_start
        """
        B, C, H, W = x.shape

        out = model(x, self.scale_timestep(t), y)  # Class conditional

        # Get Noise pred and Variance from out
        noise_pred, model_var_values = torch.split(out, C, dim=1)

        # Calculate x0
        sqrt_recip_alphas_cumprod = self._coeff_broadcasting(self.sqrt_recip_alphas_cumprod, t)
        sqrt_recipm1_alphas_cumprod = self._coeff_broadcasting(self.sqrt_recipm1_alphas_cumprod, t)
        pred_xstart = (sqrt_recip_alphas_cumprod * x) - (sqrt_recipm1_alphas_cumprod * noise_pred)
        pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)

        # Calculate Mean
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        # Calculate Variance
        min_log_var = self._coeff_broadcasting(self.posterior_log_variance_clipped, t)
        max_log_var = self._coeff_broadcasting(torch.log(self.betas), t)
        frac = (model_var_values + 1) / 2  # Convert the model_var_values from [-1, 1] -> [0, 1]
        model_log_variance = (frac * max_log_var) + ((1 - frac) * min_log_var)
        model_variance = torch.exp(model_log_variance)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, t, y, classifier=None):
        """
        sample x_{t-1} from the model at given timestep.
        """
        out = self.p_mean_variance(model, x.float(), t, y)

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0

        # Shift the mean to Sigma * gradient
        if self.classifier_guidance:
            gradient = self._calc_gradient(x, self.scale_timestep(t), y, classifier) * self.classifier_scale
            out["mean"] = out["mean"].float() + out["variance"] * gradient.float()

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Predict epsilon from x_t and predicted x_start.
        """
        sqrt_recip_alphas_cumprod = self._coeff_broadcasting(self.sqrt_recip_alphas_cumprod, t)
        sqrt_recipm1_alphas_cumprod = self._coeff_broadcasting(self.sqrt_recipm1_alphas_cumprod, t)
        return (sqrt_recip_alphas_cumprod * x_t - pred_xstart) / sqrt_recipm1_alphas_cumprod

    def ddim_sample(self, model, x, t, y, classifier=None, eta=0.0):
        """
        DDIM step: sample x_{t-1} from x_t.
        eta=0 -> deterministic DDIM, eta>0 -> stochastic DDIM.
        """
        out = self.p_mean_variance(model, x.float(), t, y)
        pred_xstart = out["pred_xstart"]

        eps = self._predict_eps_from_xstart(x, t, pred_xstart)
        alpha_bar = self._coeff_broadcasting(self.alphas_cumprod, t)
        alpha_bar_prev = self._coeff_broadcasting(self.alphas_cumprod_prev, t)

        # Optional classifier guidance in score-space (Song et al.)
        if self.classifier_guidance:
            gradient = self._calc_gradient(x, self.scale_timestep(t), y, classifier) * self.classifier_scale
            eps = eps - torch.sqrt(1.0 - alpha_bar) * gradient.float()
            pred_xstart = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)

        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * torch.sqrt(1.0 - (alpha_bar / alpha_bar_prev))
        )

        noise = torch.randn_like(x)
        mean_pred = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(
            torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)
        ) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": pred_xstart}

    def _vb_terms_bpd(self, model, x_start, x_t, t, y):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x0=x_start, xt=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, y)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, noise, y):
        """
        MSE Loss and Variational Lower Bound Loss.
        """
        x_t = self.add_noise(x_start, noise, t)
        model_output = model(x_t.float(), self.scale_timestep(t), y)

        # Model Output and Variance
        B, C, H, W = x_start.shape
        model_output, model_var_values = torch.split(model_output, C, dim=1)

        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
        vlb_loss = self._vb_terms_bpd(model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, y=y)[
            "output"
        ]

        # Divide by 1000 for equivalence with initial implementation.
        # Without a factor of 1/1000, the VB term hurts the MSE term.
        # vlb_loss *= 1 / 1000.0

        # MSE Loss
        mse_loss = mean_flat((noise - model_output) ** 2)

        return {"mse_loss": mse_loss, "vlb_loss": vlb_loss}

    def scale_timestep(self, timestep):
        """
        This is to make sure we are feeding correct timestep
        to the model. (comes handy during sampling with lower timesteps)
        """
        return self.use_timesteps.to(timestep.device)[timestep]

    @staticmethod
    def _calc_gradient(x, t, y, classifier):
        """
        Calculate Classifier Gradient of the image x at timestep t.
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0]

    @staticmethod
    def _coeff_broadcasting(coeff, timesteps):
        """
        Device Mapping and Broadcasting of the Diffusion
        Process Coefficients to the same shape as Input tensors
        (B, C, H, W).
        """

        # Make coef on the same device as timesteps
        coeff = coeff.to(timesteps.device)[timesteps]

        # Broadcast
        coeff = coeff[:, None, None, None]

        return coeff
