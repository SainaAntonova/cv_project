import torchvision.transforms as T

preprocessing_func = T.Compose(
    [
        T.Resize((227,227)),
        T.ToTensor()
    ]
)

def preprocess(img):
    return preprocessing_func(img)