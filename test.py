import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

devNumber = torch.cuda.current_device()
print(devNumber)

devName=torch.cuda.get_device_name(devNumber)
print(devName)