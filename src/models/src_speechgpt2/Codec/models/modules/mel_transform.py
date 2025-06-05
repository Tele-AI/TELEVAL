
import librosa
import torch
import torch.nn.functional as F


class MelSpectrogram(torch.nn.Module):
    """Calculate Mel-spectrogram."""

    def __init__(
        self,
        mode,
        fs=22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann_window",
        num_mels=80,
        fmin=80,
        fmax=7600, # ! 一定要是频率的一半 ！！！
        normalized=False,
        onesided=True,
        eps=1e-10,
        log_base=10.0,
    ):
        """Initialize MelSpectrogram module."""
        super().__init__()
        assert mode != None
        self.mode = mode
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        if win_length is not None:
            self.win_length = win_length
        else:
            self.win_length = fft_size
        self.normalized = normalized
        self.onesided = onesided
        self.register_buffer("window", getattr(torch, window)(self.win_length))
        self.eps = eps

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(
            sr=fs,
            n_fft=fft_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f"log_base: {log_base} is not supported.")
    
        self.pad_length = self.win_length - self.hop_size
        pad_buffer = torch.zeros(1, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)
        self.is_buffer_empty = True  # 一开始 buffer 为空

    def forward(self, x):
        """Calculate Mel-spectrogram.

        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, C, T).

        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).

        """
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            x = x.reshape(-1, x.size(2))
            
        if self.mode == "noncausal":
            x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, pad_mode="replicate")
            x_power = x_stft.real ** 2 + x_stft.imag ** 2
            x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps)).transpose(2, 1) # (B, D, T') -> (B, T', D)
            x_mel = torch.matmul(x_amp, self.melmat)
            x_mel = torch.clamp(x_mel, min=self.eps)

        elif self.mode == "causal":
            x = F.pad(x, (self.pad_length, 0), mode='replicate')
            x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, center=False, pad_mode="replicate")
            x_power = x_stft.real ** 2 + x_stft.imag ** 2
            x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps)).transpose(2, 1) # (B, D, T') -> (B, T', D)
            x_mel = torch.matmul(x_amp, self.melmat)
            x_mel = torch.clamp(x_mel, min=self.eps)
        else:
            assert False


        return self.log(x_mel).transpose(1, 2) # (B, D, T')


    def inference(self, x):
        assert self.mode == "causal"
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            assert x.shape[1] == 1 # channel 必须是 1，不然要改 pad_buffer
            x = x.reshape(-1, x.size(2))
            
        if self.is_buffer_empty: # 如果是第一帧, 则和训练时 forward 一样用最边上的值来 pad
            self.is_buffer_empty = False 
            x = F.pad(x, (self.pad_length, 0), mode='replicate') 
        else: # 否则不是第一帧, 那么上一帧已经 pad 过了,用上一帧的结果
            x = torch.cat((self.pad_buffer, x), -1)

        self.pad_buffer = x[:, x.shape[-1] - self.pad_length:]
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, center=False, pad_mode="replicate")
        x_power = x_stft.real ** 2 + x_stft.imag ** 2
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps)).transpose(2, 1) # (B, D, T') -> (B, T', D)
        x_mel = torch.matmul(x_amp, self.melmat)
        x_mel = torch.clamp(x_mel, min=self.eps)
        return self.log(x_mel).transpose(1, 2) # (B, D, T')
        

    def reset_buffer(self, batch_size):
        # self.pad_buffer.zero_()
        device = self.melmat.device
        self.pad_buffer = torch.zeros(batch_size, self.pad_length, device=device)
        self.is_buffer_empty = True
    
    # 禁止在第一帧时调用
    def set_buffer(self, pad_buffer):
        assert pad_buffer != None
        self.pad_buffer = pad_buffer
        self.is_buffer_empty = False