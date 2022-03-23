#This file is a temporary record of how data is proceeded

##Load data, which returns a tensor:face_tens, one resized-image: face
# this function fetech image by Image.open(), then by doing convert('RGB')
##to convert it as RGB image,
import torchvision.transforms as transforms
tf = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
#tf doing two things, one is to transfer the image to tensor, the other is to normalize
#the tensor; tf.to(device) is to specify the device to conduct this transform(speed up the whole process)
def load_data(img_path, device):
    face = Image.open(img_path).convert('RGB')
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(device)
    return face_tens, face

#等比例缩小图片
def resize_shorter_side(img, min_length):
    """
    Resize the shorter side of img to min_length while
    preserving the aspect ratio.
    """
    ow, oh = img.size
    mult = 8
    if ow < oh:
        if ow == min_length and oh % mult == 0:
            return img, (ow, oh)
        w = min_length
        h = int(min_length * oh / ow)
    else:
        if oh == min_length and ow % mult == 0:
            return img, (ow, oh)
        h = min_length
        w = int(min_length * ow / oh)
    return img.resize((w, h), Image.BICUBIC), (w, h)