Per caricare il modello addestrato:

net_resnet32 = resnet32()
state_dict = torch.load(f"pytorch_resnet_cifar10/pretrained_models/resnet32-trained.th", map_location=device)['state_dict']
clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
net_resnet32.load_state_dict(clean_state_dict)