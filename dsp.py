import torch


def mulaw_encode(x, quantization_channels, scale_to_int=True):
    """x_mu = mulaw_encode(x, quantization_channels, scale_to_int=True)
    Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding
    input
    -----
       x (Tensor): Input tensor, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)
        
    output
    ------
       x_mu: tensor, int64, Input after mu-law encoding
    """
    # mu
    mu = quantization_channels - 1.0

    # no check on the value of x
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    if scale_to_int:
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding
    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels
        input_int: Bool
          True: convert x_mu (int) from int to float, before mu-law decode
          False: directly decode x_mu (float) 
           
    Returns:
        Tensor: Input after mu-law decoding (float-value waveform (-1, 1))
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    if input_int:
        x = ((x_mu) / mu) * 2 - 1.0
    else:
        x = x_mu
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x
