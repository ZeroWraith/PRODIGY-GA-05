import argparse
import atexit
import io
import json
import platform
import sys
import webbrowser # Kept for potential future use, though not used in this non-web version
import copy
import time
import warnings
# asyncio, aiohttp, and torch.multiprocessing.Queue are removed as they were for web interface
from dataclasses import dataclass, is_dataclass
from functools import partial
import numpy as np
from PIL import Image, ImageCms
from tifffile import TIFF, TiffWriter
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import torch.multiprocessing as mp # Kept for mp.set_start_method if needed, but not for WebInterface queue
from tqdm import tqdm

# --- Global sRGB Profile ---
from pathlib import Path

# In Colab, files are typically uploaded to the root working directory (/content/).
# So, we assume 'sRGB Profile.icc' will be directly accessible by its name.
try:
    srgb_profile = Path('sRGB Profile.icc').read_bytes()
except FileNotFoundError:
    print("Error: 'sRGB Profile.icc' not found. Please upload it to your Colab session.")
    sys.exit(1)

# --- From sqrtm.py (Matrix Square Root functions) ---
def sqrtm_ns(a, num_iters=10):
    """Newton-Schulz iteration for matrix square root."""
    if a.ndim < 2:
        raise RuntimeError('tensor of matrices must have at least 2 dimensions')
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError('tensor must be batches of square matrices')
    if num_iters < 0:
        raise RuntimeError('num_iters must not be negative')
    norm_a = a.pow(2).sum(dim=[-2, -1], keepdim=True).sqrt()
    y = a / norm_a
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype) * 3
    z = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    z = z.repeat([*a.shape[:-2], 1, 1])
    for i in range(num_iters):
        t = (eye - z @ y) / 2
        y = y @ t
        z = t @ z
    return y * norm_a.sqrt()


class _MatrixSquareRootNSLyap(torch.autograd.Function):
    """Autograd function for Newton-Schulz matrix square root (Lyapunov-based backward pass)."""
    @staticmethod
    def forward(ctx, a, num_iters, num_iters_backward):
        z = sqrtm_ns(a, num_iters)
        ctx.save_for_backward(z, torch.tensor(num_iters_backward))
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, num_iters = ctx.saved_tensors
        norm_z = z.pow(2).sum(dim=[-2, -1], keepdim=True).sqrt()
        a = z / norm_z
        eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype) * 3
        q = grad_output / norm_z
        for i in range(num_iters):
            eye_a_a = eye - a @ a
            q = q = (q @ eye_a_a - a.transpose(-2, -1) @ (a.transpose(-2, -1) @ q - q @ a)) / 2
            if i < num_iters - 1:
                a = a @ eye_a_a / 2
        return q / 2, None, None


def sqrtm_ns_lyap(a, num_iters=10, num_iters_backward=None):
    """Matrix square root using Newton-Schulz with Lyapunov-based backward pass."""
    if num_iters_backward is None:
        num_iters_backward = num_iters
    if num_iters_backward < 0:
        raise RuntimeError('num_iters_backward must not be negative')
    return _MatrixSquareRootNSLyap.apply(a, num_iters, num_iters_backward)


class _MatrixSquareRootEig(torch.autograd.Function):
    """Autograd function for matrix square root using eigenvalue decomposition."""
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    """Matrix square root using eigenvalue decomposition."""
    if a.ndim < 2:
        raise RuntimeError('tensor of matrices must have at least 2 dimensions')
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError('tensor must be batches of square matrices')
    return _MatrixSquareRootEig.apply(a)


# --- From style_transfer.py (Core Style Transfer Logic) ---
@dataclass
class STIterate:
    """Dataclass to hold information about each style transfer iteration."""
    w: int
    h: int
    i: int
    i_max: int
    loss: float
    time: float
    gpu_ram: int


class VGGFeatures(nn.Module):
    """VGG19 feature extractor for style transfer."""
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1]
        # which are then normalized according to this transform.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # Load pre-trained VGG19 features
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model) # Default to CPU

        # Reduces edge artifacts by changing padding mode of the first conv layer.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Change pooling type and rescale activations if not max pooling.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.model.eval() # Set model to evaluation mode
        self.model.requires_grad_(False) # Freeze model parameters

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        """Helper to change padding mode of a Conv2d layer."""
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        """Calculates minimum input size required for the VGG layers."""
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]: # MaxPool2d layers in VGG19
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices_map):
        """Distributes VGG layers across specified devices."""
        for i, layer in enumerate(self.model):
            if i in devices_map:
                device = torch.device(devices_map[i])
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        """Forward pass through VGG to extract features from specified layers."""
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        input = self.normalize(input)
        for i in range(max(layers) + 1):
            input = self.model[i](input.to(self.devices[i]))
            if i in layers:
                feats[i] = input
        return feats


class ScaledMSELoss(nn.Module):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def extra_repr(self):
        return f'eps={self.eps:g}'

    def forward(self, input, target):
        diff = input - target
        return diff.pow(2).sum() / diff.abs().sum().add(self.eps)


class ContentLoss(nn.Module):
    """Content loss module (uses ScaledMSELoss)."""
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    def forward(self, input):
        return self.loss(input, self.target)


class ContentLossMSE(nn.Module):
    """Content loss module (uses standard MSELoss)."""
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = nn.MSELoss()

    def forward(self, input):
        return self.loss(input, self.target)


class StyleLoss(nn.Module):
    """Style loss module using Gram matrix and ScaledMSELoss."""
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    @staticmethod
    def get_target(target):
        """Computes the Gram matrix for style representation."""
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. (2015) and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input):
        return self.loss(self.get_target(input), self.target)


def eye_like(x):
    """Returns an identity matrix with the same shape and device as x."""
    return torch.eye(x.shape[-2], x.shape[-1], dtype=x.dtype, device=x.device).expand_as(x)


class StyleLossW2(nn.Module):
    """Wasserstein-2 style loss."""
    def __init__(self, target, eps=1e-4):
        super().__init__()
        # Use sqrtm_ns_lyap from sqrtm.py
        self.sqrtm = partial(sqrtm_ns_lyap, num_iters=12)
        mean, srm = target
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * eps
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)
        self.register_buffer('cov_sqrt', self.sqrtm(cov))
        self.register_buffer('eps', mean.new_tensor(eps))

    @staticmethod
    def get_target(target):
        """Compute the mean and second raw moment of the target activations."""
        mean = target.mean([-2, -1])
        srm = torch.einsum('...chw,...dhw->...cd', target, target) / (target.shape[-2] * target.shape[-1])
        return mean, srm

    @staticmethod
    def srm_to_cov(mean, srm):
        """Compute the covariance matrix from the mean and second raw moment."""
        return srm - torch.einsum('...c,...d->...cd', mean, mean)

    def forward(self, input):
        mean, srm = self.get_target(input)
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * self.eps
        mean_diff = torch.mean((mean - self.mean) ** 2)
        sqrt_term = self.sqrtm(self.cov_sqrt @ cov @ self.cov_sqrt)
        cov_diff = torch.diagonal(self.cov + cov - 2 * sqrt_term, dim1=-2, dim2=-1).mean()
        return mean_diff + cov_diff


class TVLoss(nn.Module):
    """L2 total variation loss (nine point stencil) for image smoothing."""
    def forward(self, input):
        input = F.pad(input, (1, 1, 1, 1), 'replicate')
        s1, s2 = slice(1, -1), slice(2, None)
        s3, s4 = slice(None, -1), slice(1, None)
        d1 = (input[..., s1, s2] - input[..., s1, s1]).pow(2).mean() / 3
        d2 = (input[..., s2, s1] - input[..., s1, s1]).pow(2).mean() / 3
        d3 = (input[..., s4, s4] - input[..., s3, s3]).pow(2).mean() / 12
        d4 = (input[..., s4, s3] - input[..., s3, s4]).pow(2).mean() / 12
        return 2 * (d1 + d2 + d3 + d4)


class SumLoss(nn.ModuleList):
    """Combines multiple loss functions into a single sum."""
    def __init__(self, losses, verbose=False):
        super().__init__(losses)
        self.verbose = verbose

    def forward(self, *args, **kwargs):
        losses = [loss(*args, **kwargs) for loss in self]
        if self.verbose:
            for i, loss in enumerate(losses):
                print(f'({i}): {loss.item():g}')
        return sum(loss.to(losses[-1].device) for loss in losses)


class Scale(nn.Module):
    """Applies a scaling factor to the output of a module."""
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class LayerApply(nn.Module):
    """Applies a module to a specific layer's output from a dictionary of features."""
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'(layer): {self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


class EMA(nn.Module):
    """A bias-corrected exponential moving average, as in Kingma et al. (Adam)."""
    def __init__(self, input, decay):
        super().__init__()
        self.register_buffer('value', torch.zeros_like(input))
        self.register_buffer('decay', torch.tensor(decay))
        self.register_buffer('accum', torch.tensor(1.))
        self.update(input)

    def get(self):
        return self.value / (1 - self.accum)

    def update(self, input):
        self.accum *= self.decay
        self.value *= self.decay
        self.value += (1 - self.decay) * input


def size_to_fit(size, max_dim, scale_up=False):
    """Calculates new dimensions to fit an image within max_dim while preserving aspect ratio."""
    w, h = size
    if not scale_up and max(h, w) <= max_dim:
        return w, h
    new_w, new_h = max_dim, max_dim
    if h > w:
        new_w = round(max_dim * w / h)
    else:
        new_h = round(max_dim * h / w)
    return new_w, new_h


def gen_scales(start, end):
    """Generates a list of scales for multi-scale style transfer."""
    scale = end
    i = 0
    scales = set()
    while scale >= start:
        scales.add(scale)
        i += 1
        scale = round(end / pow(2, i/2))
    return sorted(scales)


def interpolate(*args, **kwargs):
    """Wrapper for F.interpolate to suppress warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.interpolate(*args, **kwargs)


def scale_adam(state, shape):
    """Prepares an Adam optimizer state dict to warm-start at a new image scale."""
    state = copy.deepcopy(state)
    for group in state['state'].values():
        exp_avg, exp_avg_sq = group['exp_avg'], group['exp_avg_sq']
        group['exp_avg'] = interpolate(exp_avg, shape, mode='bicubic')
        group['exp_avg_sq'] = interpolate(exp_avg_sq, shape, mode='bilinear').relu_()
        if 'max_exp_avg_sq' in group:
            max_exp_avg_sq = group['max_exp_avg_sq']
            group['max_exp_avg_sq'] = interpolate(max_exp_avg_sq, shape, mode='bilinear').relu_()
    return state


class StyleTransfer:
    """Main class for performing neural style transfer."""
    def __init__(self, devices=['cpu'], pooling='max'):
        self.devices = [torch.device(device) for device in devices]
        self.image = None
        self.average = None

        # Default content and style layers follow Gatys et al. (2015).
        self.content_layers = [22]
        self.style_layers = [1, 6, 11, 20, 29]

        # The weighting of the style layers differs from Gatys et al. (2015) and Johnson et al.
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        self.model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling)

        # Device distribution plan for VGG layers
        if len(self.devices) == 1:
            device_plan = {0: self.devices[0]}
        elif len(self.devices) == 2:
            device_plan = {0: self.devices[0], 5: self.devices[1]} # Distribute across two GPUs
        else:
            raise ValueError('Only 1 or 2 devices are supported for VGG layer distribution.')

        self.model.distribute_layers(device_plan)

    def get_image_tensor(self):
        """Returns the current averaged generated image as a detached tensor."""
        if self.average is not None:
            return self.average.get().detach()[0].clamp(0, 1)
        return None

    def get_image(self, image_type='pil'):
        """Returns the generated image in the specified format."""
        image_tensor = self.get_image_tensor()
        if image_tensor is not None:
            if image_type.lower() == 'pil':
                return TF.to_pil_image(image_tensor)
            elif image_type.lower() == 'np_uint16':
                arr = image_tensor.cpu().movedim(0, 2).numpy()
                return np.uint16(np.round(arr * 65535))
            else:
                raise ValueError("image_type must be 'pil' or 'np_uint16'")
        return None

    def stylize(self, content_image, style_images, *,
                style_weights=None,
                content_weight: float = 0.015,
                tv_weight: float = 2.,
                optimizer: str = 'adam',
                min_scale: int = 128,
                end_scale: int = 512,
                iterations: int = 500,
                initial_iterations: int = 1000,
                step_size: float = 0.02,
                avg_decay: float = 0.99,
                init: str = 'content',
                style_scale_fac: float = 1.,
                style_size: int = None,
                callback=None):
        """
        Performs the neural style transfer process.

        Args:
            content_image (PIL.Image): The content image.
            style_images (list of PIL.Image): List of style images.
            style_weights (list of float, optional): Relative weights for each style image.
            content_weight (float): Weight for the content loss.
            tv_weight (float): Weight for the total variation (smoothing) loss.
            optimizer (str): Optimizer to use ('adam' or 'lbfgs').
            min_scale (int): Minimum image dimension during multi-scale optimization.
            end_scale (int): Final image dimension for optimization.
            iterations (int): Number of iterations per scale (after initial scale).
            initial_iterations (int): Number of iterations for the first (smallest) scale.
            step_size (float): Learning rate for Adam optimizer.
            avg_decay (float): EMA decay rate for iterate averaging.
            init (str): Initialization method for the generated image ('content', 'gray', 'uniform', 'normal', 'style_stats').
            style_scale_fac (float): Relative scale of the style image to the content image.
            style_size (int, optional): Fixed scale for style images, overrides style_scale_fac if set.
            callback (callable, optional): A function to call after each iteration.
        """
        min_scale = min(min_scale, end_scale)
        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)

        if style_weights is None:
            style_weights = [1 / len(style_images)] * len(style_images)
        else:
            weight_sum = sum(abs(w) for w in style_weights)
            style_weights = [weight / weight_sum for weight in style_weights]
        if len(style_images) != len(style_weights):
            raise ValueError('style_images and style_weights must have the same length')

        tv_loss = Scale(LayerApply(TVLoss(), 'input'), tv_weight)

        scales = gen_scales(min_scale, end_scale)

        # Initialize the generated image based on the chosen method
        cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_image.resize((cw, ch), Image.BICUBIC))[None]
        elif init == 'gray':
            self.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5 # A gray-ish random image
        elif init == 'uniform':
            self.image = torch.rand([1, 3, ch, cw]) # Uniform random image
        elif init == 'normal':
            self.image = torch.empty([1, 3, ch, cw])
            nn.init.trunc_normal_(self.image, mean=0.5, std=0.25, a=0, b=1) # Truncated normal
        elif init == 'style_stats':
            # Initialize based on mean and variance of style images
            means, variances = [], []
            for i, image in enumerate(style_images):
                my_image = TF.to_tensor(image)
                means.append(my_image.mean(dim=(1, 2)) * style_weights[i])
                variances.append(my_image.var(dim=(1, 2)) * style_weights[i])
            means = sum(means)
            variances = sum(variances)
            channels = []
            for mean, variance in zip(means, variances):
                channel = torch.empty([1, 1, ch, cw])
                nn.init.trunc_normal_(channel, mean=mean, std=variance.sqrt(), a=0, b=1)
                channels.append(channel)
            self.image = torch.cat(channels, dim=1)
        else:
            raise ValueError("init must be one of 'content', 'gray', 'uniform', 'normal', 'style_stats'")
        self.image = self.image.to(self.devices[0])

        opt = None # Optimizer will be initialized per scale

        # Multi-scale optimization loop
        for scale in scales:
            if self.devices[0].type == 'cuda':
                torch.cuda.empty_cache() # Clear CUDA cache for memory efficiency

            cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
            content = TF.to_tensor(content_image.resize((cw, ch), Image.BICUBIC))[None]
            content = content.to(self.devices[0])

            # Resize the generated image to the current scale
            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.average = EMA(self.image, avg_decay) # Initialize Exponential Moving Average
            self.image.requires_grad_() # Enable gradient computation for the image

            print(f'Processing content image ({cw}x{ch})...')
            content_feats = self.model(content, layers=self.content_layers)
            content_losses = []
            for layer, weight in zip(self.content_layers, content_weights):
                target = content_feats[layer]
                content_losses.append(Scale(LayerApply(ContentLossMSE(target), layer), weight))

            style_targets, style_losses = {}, []
            for i, image in enumerate(style_images):
                # Determine style image size
                if style_size is None:
                    sw, sh = size_to_fit(image.size, round(scale * style_scale_fac))
                else:
                    sw, sh = size_to_fit(image.size, style_size)
                style = TF.to_tensor(image.resize((sw, sh), Image.BICUBIC))[None]
                style = style.to(self.devices[0])
                print(f'Processing style image ({sw}x{sh})...')
                style_feats = self.model(style, layers=self.style_layers)
                # Aggregate style targets (means and covariance matrices) for multiple styles
                for layer in self.style_layers:
                    target_mean, target_cov = StyleLossW2.get_target(style_feats[layer])
                    target_mean *= style_weights[i]
                    target_cov *= style_weights[i]
                    if layer not in style_targets:
                        style_targets[layer] = target_mean, target_cov
                    else:
                        style_targets[layer][0].add_(target_mean)
                        style_targets[layer][1].add_(target_cov)

            for layer, weight in zip(self.style_layers, self.style_weights):
                target = style_targets[layer]
                style_losses.append(Scale(LayerApply(StyleLossW2(target), layer), weight))

            crit = SumLoss([*content_losses, *style_losses, tv_loss]) # Combined loss function

            # Optimizer setup
            if optimizer == 'adam':
                opt2 = optim.Adam([self.image], lr=step_size, betas=(0.9, 0.99))
                # Warm-start Adam if not the first scale
                if scale != scales[0] and opt is not None:
                    opt_state = scale_adam(opt.state_dict(), (ch, cw))
                    opt2.load_state_dict(opt_state)
                opt = opt2
            elif optimizer == 'lbfgs':
                opt = optim.LBFGS([self.image], max_iter=1, history_size=10)
            else:
                raise ValueError("optimizer must be one of 'adam', 'lbfgs'")

            if self.devices[0].type == 'cuda':
                torch.cuda.empty_cache()

            def closure():
                """Closure for L-BFGS optimizer."""
                feats = self.model(self.image)
                loss = crit(feats)
                loss.backward()
                return loss

            actual_its = initial_iterations if scale == scales[0] else iterations
            for i in range(1, actual_its + 1):
                opt.zero_grad()
                loss = opt.step(closure) # Perform optimization step
                # Enforce box constraints [0, 1] for pixel values
                if optimizer != 'lbfgs': # L-BFGS handles constraints differently
                    with torch.no_grad():
                        self.image.clamp_(0, 1)
                self.average.update(self.image) # Update EMA of the image

                # Callback for progress reporting and saving intermediate images
                if callback is not None:
                    gpu_ram = 0
                    for device in self.devices:
                        if device.type == 'cuda':
                            gpu_ram = max(gpu_ram, torch.cuda.max_memory_allocated(device))
                    callback(STIterate(w=cw, h=ch, i=i, i_max=actual_its, loss=loss.item(),
                                       time=time.time(), gpu_ram=gpu_ram))

            # Initialize next scale with the previous scale's averaged iterate.
            with torch.no_grad():
                self.image.copy_(self.average.get())

        return self.get_image()


# --- CLI-related functions (adapted for Colab) ---
def prof_to_prof(image, src_prof, dst_prof, **kwargs):
    """Converts an image between ICC color profiles."""
    src_prof = io.BytesIO(src_prof)
    dst_prof = io.BytesIO(dst_prof)
    return ImageCms.profileToProfile(image, src_prof, dst_prof, **kwargs)


def load_image(path, proof_prof=None):
    """Loads an image, handling ICC profiles and converting to RGB."""
    src_prof = dst_prof = srgb_profile
    try:
        image = Image.open(path)
        if 'icc_profile' in image.info:
            src_prof = image.info['icc_profile']
        else:
            image = image.convert('RGB') # Assume sRGB if no profile
        if proof_prof is None:
            if src_prof == dst_prof:
                return image.convert('RGB')
            # For Colab, proof_prof would also need to be a local file path
            proof_prof_bytes = Path(proof_prof).read_bytes()
            cmyk = prof_to_prof(image, src_prof, proof_prof_bytes, outputMode='CMYK')
            return prof_to_prof(cmyk, proof_prof_bytes, dst_prof, outputMode='RGB')
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_pil(path, image):
    """Saves a PIL image to disk."""
    try:
        kwargs = {'icc_profile': srgb_profile}
        if path.suffix.lower() in {'.jpg', '.jpeg'}:
            kwargs['quality'] = 95
            kwargs['subsampling'] = 0
        elif path.suffix.lower() == '.webp':
            kwargs['quality'] = 95
        image.save(path, **kwargs)
    except (OSError, ValueError) as err:
        print_error(err)
        sys.exit(1)


def save_tiff(path, image):
    """Saves a NumPy array image as TIFF with sRGB profile."""
    tag = ('InterColorProfile', TIFF.DATATYPES.BYTE, len(srgb_profile), srgb_profile, False)
    try:
        with TiffWriter(path) as writer:
            writer.save(image, photometric='rgb', resolution=(72, 72), extratags=[tag])
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_image(path, image):
    """Saves an image (PIL or NumPy) to the specified path."""
    path = Path(path)
    tqdm.write(f'Writing image to {path}.')
    if isinstance(image, Image.Image):
        save_pil(path, image)
    elif isinstance(image, np.ndarray) and path.suffix.lower() in {'.tif', '.tiff'}:
        save_tiff(path, image)
    else:
        raise ValueError('Unsupported combination of image type and extension')


def get_safe_scale(w, h, dim):
    """Computes a safe end_scale for content image given GPU memory constraints."""
    return int(pow(w / h if w > h else h / w, 1/2) * dim)


def print_error(err):
    """Prints an error message to stderr in red."""
    print('\033[31m{}:\033[0m {}'.format(type(err).__name__, err), file=sys.stderr)


class Callback:
    """Callback class to report style transfer progress and save intermediate images."""
    def __init__(self, st, args, image_type='pil'): # web_interface removed
        self.st = st
        self.args = args
        self.image_type = image_type
        # self.web_interface = web_interface # Removed
        self.iterates = []
        self.progress = None

    def __call__(self, iterate):
        """Called after each iteration."""
        self.iterates.append(iterate.__dict__) # Store as dict for JSON serialization
        if iterate.i == 1:
            self.progress = tqdm(total=iterate.i_max, dynamic_ncols=True)
        msg = 'Size: {}x{}, iteration: {}, loss: {:g}'
        tqdm.write(msg.format(iterate.w, iterate.h, iterate.i, iterate.loss))
        self.progress.update()
        # if self.web_interface is not None: # Removed web interface calls
        #     self.web_interface.put_iterate(iterate, self.st.get_image_tensor())
        if iterate.i == iterate.i_max:
            self.progress.close()
            # Save final image only if it's the last scale
            if max(iterate.w, iterate.h) == self.args.end_scale:
                pass # No web_interface.put_done()
        elif iterate.i % self.args.save_every == 0:
            # Save intermediate image
            save_image(self.args.output, self.st.get_image(self.image_type))

    def close(self):
        """Closes the progress bar."""
        if self.progress is not None:
            self.progress.close()

    def get_trace(self):
        """Returns the full trace of arguments and iteration data."""
        return {'args': self.args.__dict__, 'iterates': self.iterates}


def main():
    """Main function to parse arguments and run the neural style transfer."""
    # setup_exceptions() and fix_start_method() are often not needed in Colab
    # as the environment is pre-configured.
    # For robust multiprocessing, 'spawn' is generally recommended for PyTorch.
    # Colab often defaults to 'spawn' or handles it.
    # if platform.system() != 'Windows':
    #     mp.set_start_method('spawn', force=True) # Already handled in __main__ block

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Helper to get default values and types from StyleTransfer.stylize method
    def arg_info(arg):
        defaults = StyleTransfer.stylize.__kwdefaults__
        annotations = StyleTransfer.stylize.__annotations__
        return {'default': defaults.get(arg), 'type': annotations.get(arg)}

    p.add_argument('content', type=str, help='the content image file path')
    p.add_argument('styles', type=str, nargs='+', metavar='style', help='the style image file path(s)')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output image file path')
    p.add_argument('--style-weights', '-sw', type=float, nargs='+', default=None,
                   metavar='STYLE_WEIGHT', help='the relative weights for each style image')
    p.add_argument('--devices', type=str, default=[], nargs='+',
                   help='the device names to use (e.g., "cuda:0", "cpu"; omit for auto-detect)')
    p.add_argument('--random-seed', '-r', type=int, default=0,
                   help='the random seed for reproducibility')
    p.add_argument('--content-weight', '-cw', **arg_info('content_weight'),
                   help='the content weight')
    p.add_argument('--tv-weight', '-tw', **arg_info('tv_weight'),
                   help='the smoothing weight (total variation loss)')
    p.add_argument('--optimizer', **arg_info('optimizer'),
                   choices=['adam', 'lbfgs'],
                   help='the optimizer to use')
    p.add_argument('--min-scale', '-ms', **arg_info('min_scale'),
                   help='the minimum scale (max image dim), in pixels, for multi-scale optimization')
    p.add_argument('--end-scale', '-s', type=str, default='512',
                   help='the final scale (max image dim), in pixels (can end with "+" for safe scale)')
    p.add_argument('--iterations', '-i', **arg_info('iterations'),
                   help='the number of iterations per scale (after initial scale)')
    p.add_argument('--initial-iterations', '-ii', **arg_info('initial_iterations'),
                   help='the number of iterations on the first (smallest) scale')
    p.add_argument('--save-every', type=int, default=50,
                   help='save the image every SAVE_EVERY iterations (per scale)')
    p.add_argument('--step-size', '-ss', **arg_info('step_size'),
                   help='the step size (learning rate) for Adam optimizer')
    p.add_argument('--avg-decay', '-ad', **arg_info('avg_decay'),
                   help='the EMA decay rate for iterate averaging')
    p.add_argument('--init', **arg_info('init'),
                   choices=['content', 'gray', 'uniform', 'normal', 'style_stats'],
                   help='the initial image initialization method')
    p.add_argument('--style-scale-fac', **arg_info('style_scale_fac'),
                   help='the relative scale of the style image to the content image')
    p.add_argument('--style-size', **arg_info('style_size'),
                   help='the fixed scale of the style at different content scales (overrides style-scale-fac)')
    p.add_argument('--pooling', type=str, default='max', choices=['max', 'average', 'l2'],
                   help='the model\'s pooling mode (VGG feature extractor)')
    p.add_argument('--proof', type=str, default=None,
                   help='the ICC color profile (CMYK) for soft proofing the content and styles (file path)')
    # Removed --web, --host, --port, --browser arguments

    args = p.parse_args()

    # Load content and style images
    content_img = load_image(args.content, args.proof)
    style_imgs = [load_image(img, args.proof) for img in args.styles]

    # Determine output image type based on extension
    image_type = 'pil'
    if Path(args.output).suffix.lower() in {'.tif', '.tiff'}:
        image_type = 'np_uint16'

    # Setup devices (CUDA if available, otherwise CPU)
    devices = [torch.device(device) for device in args.devices]
    if not devices:
        devices = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')]
    if len(set(device.type for device in devices)) != 1:
        print_error(ValueError('Devices must all be the same type (e.g., all cuda or all cpu).'))
        sys.exit(1)
    # The original code supported 1 or 2 devices for VGG distribution.
    # This check is kept for consistency with the model's distribute_layers logic.
    if not 1 <= len(devices) <= 2:
        print_error(ValueError('Only 1 or 2 devices are supported for VGG layer distribution.'))
        sys.exit(1)
    print('Using devices:', ' '.join(str(device) for device in devices))

    if devices[0].type == 'cpu':
        print('CPU threads:', torch.get_num_threads())
    if devices[0].type == 'cuda':
        for i, device in enumerate(devices):
            props = torch.cuda.get_device_properties(device)
            print(f'GPU {i} type: {props.name} (compute {props.major}.{props.minor})')
            print(f'GPU {i} RAM:', round(props.total_memory / 1024 / 1024), 'MB')

    # Handle '+' in end_scale for safe scaling
    end_scale_str = str(args.end_scale).rstrip('+')
    try:
        end_scale = int(end_scale_str)
    except ValueError:
        print_error(ValueError(f"Invalid value for --end-scale: '{args.end_scale}'. Must be an integer or integer followed by '+'."))
        sys.exit(1)

    if str(args.end_scale).endswith('+'):
        end_scale = get_safe_scale(*content_img.size, end_scale)
    args.end_scale = end_scale # Update args with resolved end_scale

    # web_interface is removed, so this block is no longer needed
    # web_interface = None
    # if args.web:
    #     web_interface = WebInterface(args.host, args.port)
    #     atexit.register(web_interface.close)

    # Set random seed for reproducibility
    for device in devices:
        torch.tensor(0).to(device)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed) # Also set numpy seed for consistency if used

    print('Loading model...')
    st = StyleTransfer(devices=devices, pooling=args.pooling)
    # Callback no longer takes web_interface argument
    callback = Callback(st, args, image_type=image_type)
    atexit.register(callback.close) # Ensure callback resources are closed on exit

    # Browser opening logic removed as web interface is gone
    # url = f'http://{args.host}:{args.port}/'
    # if args.web:
    #     if args.browser:
    #         try:
    #             webbrowser.get(args.browser).open(url)
    #         except webbrowser.Error as e:
    #             print_error(f"Could not open browser '{args.browser}': {e}")
    #     elif args.browser is None:
    #         try:
    #             webbrowser.open(url)
    #         except webbrowser.Error as e:
    #             print_error(f"Could not open default browser: {e}")

    # Prepare arguments for stylize method
    defaults = StyleTransfer.stylize.__kwdefaults__
    st_kwargs = {k: getattr(args, k) for k in defaults if hasattr(args, k)}

    # Perform style transfer
    try:
        st.stylize(content_img, style_imgs, **st_kwargs, callback=callback)
    except KeyboardInterrupt:
        print("\nStyle transfer interrupted by user.")
    except Exception as e:
        print_error(e)
        sys.exit(1)

    # Save final output image
    output_image = st.get_image(image_type)
    if output_image is not None:
        save_image(args.output, output_image)
        # Display the image in Colab after saving
        try:
            from IPython.display import Image as DisplayImage, display
            display(DisplayImage(filename=str(args.output)))
            print(f"Generated image displayed and saved as '{args.output}'")
        except ImportError:
            print(f"Generated image saved as '{args.output}'")
    else:
        print_error(RuntimeError("No output image generated."))

    # Save trace data to JSON
    try:
        with open('trace.json', 'w') as fp:
            json.dump(callback.get_trace(), fp, indent=4)
        print("Trace data saved to 'trace.json'.")
    except Exception as e:
        print_error(f"Error saving trace data: {e}")


if __name__ == '__main__':
    # Ensure multiprocessing context is set up correctly for all OS
    # This is crucial when using multiprocessing.Process, especially with PyTorch.
    # Colab environments typically handle this well, but 'spawn' is a safe default.
    if platform.system() != 'Windows': # 'spawn' is default on Windows, but not on Unix/macOS
        mp.set_start_method('spawn', force=True)
    main()

