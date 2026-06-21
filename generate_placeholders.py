import torch, os
os.makedirs('data/placeholder', exist_ok=True)
torch.save((torch.randn(500,24), torch.randint(0,4,(500,))), 'data/placeholder/part1.pt')
torch.save((torch.randn(400,1,128,100), torch.randint(0,4,(400,))), 'data/placeholder/part2.pt')
torch.save((torch.randint(1,50,(300,15)), torch.randint(1,10,(300,8))), 'data/placeholder/part3.pt')
print("? Placeholders generated in data/placeholder/")
