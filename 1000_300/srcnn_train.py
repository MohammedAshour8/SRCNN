import torch as th
from chlorophyll_dataset import ChlorophyllDataset
from image_aligner import ImageAligner
from tqdm import tqdm
from model import SRCNN
import torch.cuda.amp as amp
import argparse

parser = argparse.ArgumentParser(description='SRCNN Training')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--epochs', default=1200, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--in_channels', default=2, type=int, help='Number of input channels')
parser.add_argument('--device', default='cuda', type=str, help='Device to use (cuda or cpu)')
parser.add_argument('--save_model', default='model.pth', type=str, help='Path to save the trained model')
parser.add_argument('--low_res_path', default='../archivos_prueba/1km_300m/1km/', type=str, help='Path to the low resolution images')
parser.add_argument('--high_res_path', default='../archivos_prueba/1km_300m/300m/', type=str, help='Path to the high resolution images')

args = parser.parse_args()

aligner = ImageAligner(args.low_res_path, args.high_res_path)
aligner.align_images()

# Load the data
batch_size = args.batch_size
train_dataset = ChlorophyllDataset(args.low_res_path, args.high_res_path)

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = th.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Create the model
device = th.device(args.device if th.cuda.is_available() else 'cpu')
model = SRCNN(in_channels=args.in_channels).to(device)
criterion = th.nn.MSELoss().to(device)
optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

model.train()

# Train the model for 10 epochs
epochs = args.epochs
array_loss = []
scaler = amp.GradScaler()
for epoch in tqdm(range(epochs)):
    for (low_res, high_res) in train_loader:
        low_res, high_res = low_res.to(device), high_res.unsqueeze(1).to(device)
        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Free up GPU memory
        del low_res, high_res, outputs
        th.cuda.empty_cache()
    array_loss.append(loss.item())


# Save the model
th.save(model.state_dict(), args.save_model)