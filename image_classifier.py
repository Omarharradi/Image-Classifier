import torch
from torchvision import models, transforms
from PIL import Image

resnet = models.resnet101(pretrained=True)
resnet.eval()  

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_image(image_path):
    img = Image.open(image_path)
    img_t = preprocess(img)

    batch_t = torch.unsqueeze(img_t, 0)

    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    return labels[index[0]], percentage[index[0]].item()

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    
    try:
        label, confidence = classify_image(image_path)
        print(f"Predicted label: {label}, Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"An error occurred: {e}")
