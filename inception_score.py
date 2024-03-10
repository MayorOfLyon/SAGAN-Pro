import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.models as models
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # # Load inception model    
    inception_model = models.inception_v3(pretrained=False, transform_input=False)
    inception_model.to('cuda')
    checkpoint = torch.load('inception_v3_google-0cc3c7bd.pth', map_location='cuda:0')
    inception_model.load_state_dict(checkpoint)
    inception_model.eval()
    
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    # print("preds shape is ",preds.shape)

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)

def get_inception_score(dataloader):
    print ("inception score is ",inception_score(dataloader, cuda=True, batch_size=32, resize=True, splits=10))
    
# if __name__ == '__main__':
#     from torchvision import datasets, transforms
#     dataloader = torch.utils.data.DataLoader(
#         datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
#             transforms.Resize(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ])),
#         batch_size=32, shuffle=True)
#     get_inception_score(dataloader)