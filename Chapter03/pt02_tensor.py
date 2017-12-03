import torch
a = torch.ones(3)
print("\n01==============")
print(a)

print("\n02==============")
b = a.numpy()
print(b)

print("\n03==============")
a.add_(1)
print(a)
print(b)
