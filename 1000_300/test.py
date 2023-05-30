from skimage.metrics import peak_signal_noise_ratio
import torch as th
import torch.cuda.amp as amp

class TestModel:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def test(self):
        psnr_scores = []
        with th.no_grad():
            for (low_res, high_res) in self.test_loader:
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                with amp.autocast():
                    outputs = self.model(low_res)
                # Remove the extra channel dimension from outputs
                outputs = outputs[:, 0, :, :]  # Select the first channel
                # Squeeze the tensors to remove the batch_size dimension
                outputs = outputs.squeeze(0)
                high_res = high_res.squeeze(0)
                # Calculate PSNR and SSIM
                psnr = peak_signal_noise_ratio(high_res.cpu().numpy(), outputs.cpu().numpy(), data_range=1.0)
                psnr_scores.append(psnr)
                # Free up GPU memory
                del low_res, high_res, outputs
                th.cuda.empty_cache()
        return psnr_scores