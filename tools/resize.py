from PIL import Image


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


def resize_img(img, w, h):
    return img.resize((w, h), Image.BICUBIC), (w, h)


def crop_img(img, box):
    return img.crop(box), box
