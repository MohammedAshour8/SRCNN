from skimage.metrics import peak_signal_noise_ratio
import torch as th
import torch.cuda.amp as amp
from skimage.metrics import structural_similarity as ssim

class TestModel:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def training_test(self, test_loader):
        psnr_scores = []
        ssim_scores = []
        with th.no_grad():
            for (low_res, high_res) in test_loader:
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                with amp.autocast():
                    outputs = self.model(low_res)
                # Remove the extra channel dimension from outputs
                outputs = outputs[:, 0, :, :]
                # Squeeze the tensors to remove the batch_size dimension
                outputs = outputs.squeeze(0)
                high_res = high_res.squeeze(0)
                # Calculate the PSNR score and SSIM score
                psnr = peak_signal_noise_ratio(high_res.cpu().numpy(), outputs.cpu().numpy(), data_range=1.0)
                ssim_score = ssim(high_res.cpu().numpy(), outputs.cpu().numpy(), data_range=1.0)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim_score)
                # Free up GPU memory
                del low_res, high_res, outputs
                th.cuda.empty_cache()
        return psnr_scores, ssim_scores
    
    def single_test(self, high_res, prediction):
        prediction = prediction[0, 0, :, :]
        high_res = high_res[:, :]
        psnr = peak_signal_noise_ratio(high_res, prediction, data_range=1.0)
        ssim_score = ssim(high_res, prediction, data_range=1.0, multichannel=True)
        return psnr, ssim_score