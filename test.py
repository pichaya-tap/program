import torch

print('torch version: {:}'.format(torch.__version__))
print('cuda arch list version: {:}'.format(torch.cuda.get_arch_list()))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Device: {:}'.format(device))

print('CUDA available: {:}'.format(torch.cuda.is_available()))