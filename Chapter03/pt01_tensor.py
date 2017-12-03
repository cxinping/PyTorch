import torch
z = torch.Tensor(4, 5)
print("\n01==============")
print(z)

print("\n02==============")
print(z.size())

y = torch.rand(4, 5)#产生一个5行3列的矩阵
print("\n03==============")
print(z + y)

print("\n04==============")
print(torch.add(z, y))

print("\n05==============")
result = torch.Tensor(4, 5)
torch.add(z, y, out=result)
print(result)
