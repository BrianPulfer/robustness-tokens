import torchvision.transforms as tt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = tt.Normalize(IMAGENET_MEAN, IMAGENET_STD)
unnormalize = tt.Normalize(
    [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)], [1 / s for s in IMAGENET_STD]
)

to_pil = tt.ToPILImage()

transform = tt.Compose([tt.Resize((224, 224)), tt.ToTensor(), normalize])

to_img = tt.Compose([unnormalize, to_pil])
