import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import OneHotCategorical


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20*256)
        self.fc3 = nn.Linear(20*256, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(h1).view(len(x), 20, 256), -1)
        return probs

    def decode(self, z):
        h3 = F.relu(self.fc3(z.view(len(z), 20*256)))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        # For convenience we use torch.distributions to sample and compute the values of interest for the distribution see (https://pytorch.org/docs/stable/distributions.html) for more details.
        probs = self.encode(x.view(-1, 784))
        m = OneHotCategorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return self.decode(action), log_prob, entropy


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, log_prob, entropy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduce=False).sum(-1)
    # We make the assumption that q(z1,..,zd|x) = q(z1|x)...q(zd|x)
    log_prob = log_prob.sum(-1)
    # The Reinforce loss is just log_prob*loss
    reinforce_loss = torch.sum(log_prob*BCE.detach())
    # If the prior on the latent is uniform then the KL is just the entropy of q(z|x)
    # We add reinforce_loss - reinforce_loss.detach() so we can backpropagate through the encoder with REINFORCE but it doesn't modify the loss.
    loss = BCE.sum() + reinforce_loss - reinforce_loss.detach() - entropy.sum()
    return loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, log_prob, entropy = model(data)
        loss = loss_function(recon_batch, data, log_prob, entropy)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, log_prob, entropy = model(data)
            test_loss += loss_function(recon_batch, data, log_prob, entropy).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        m = OneHotCategorical(torch.ones(256)/256.)
        sample = m.sample((64, 20))
        sample = sample.to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
