from torch.utils.tensorboard import SummaryWriter
from model import SRCNN
import torch as th

# Create the model
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = SRCNN(in_channels=2).to(device)
criterion = th.nn.MSELoss().to(device)
optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

writer = SummaryWriter()

writer.add_graph(model, th.rand(1, 2, 750, 750).to(device))
writer.close()