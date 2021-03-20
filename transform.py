import math
import torch
from torch import nn
from torch.nn import functional as F
from _utils import ImageList


def expand_boxes(boxes, scale):
    
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    M = mask.shape[-1]
    scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])]

    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):

    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    res = [
        paste_mask_in_image(m[0], b, im_h, im_w)
        for m, b in zip(masks, boxes)]

    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))

    return ret


def _resize_image_and_masks(image, self_min_size, self_max_size, target):

    im_shape = torch.tensor(image.shape[-2:]) # (width, height)

    # find shorter edge 
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))

    # find scale factor
    scale_factor = self_min_size / min_size

    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size

    image = F.interpolate(image[None], scale_factor=scale_factor, 
                          mode='bilinear', recompute_scale_factor=True,
                          align_corners=False)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = F.interpolate(mask[:, None].float(), scale_factor=scale_factor, 
                             recompute_scale_factor=True)[:, 0].byte()
        target["masks"] = mask

    return image, target


def resize_boxes(boxes, original_size, new_size):

    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)]

    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1) # = unsqueeze

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()

        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std


    # normalize and resize images 
    def forward(self,
                images,        
                targets=None  
                ):

        images = [img for img in images]
    
        for i in range(len(images)):
            
            image = images[i]
            target_index = targets[i] if targets is not None else None

            # normalized image
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image

            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images] # get width, height of a image
        images = self.batch_images(images)
        image_sizes_list = []

        # get image size to compose ImageList type
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets


    # normalize image 
    def normalize(self, image):
        
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]


    def resize(self, image, target):
        
        h, w = image.shape[-2:]
        size = float(self.min_size[-1])

        image, target = _resize_image_and_masks(image, size, float(self.max_size), target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:]) # boxes, original size, newsize
        target["boxes"] = bbox

        return image, target


    def batch_images(self, images, size_divisible=32):

        the_list = [list(img.shape) for img in images] # ex) [[32, 32], [64, 64], ...]
        maxes = the_list[0]

        for sublist in the_list[1:]: # sublist : [image width, image height]
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        
        max_size = maxes # max width, height of an image(list type)

        stride = float(size_divisible)
        max_size = list(max_size)

        # turn maximum float image size into integer
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size # ex) [1, 3, 800, 1300]

        batched_imgs = images[0].new_full(batch_shape, 0)

        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs


    def postprocess(self,
                    result,               
                    image_shapes,         
                    original_image_sizes  
                    ):

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

        return result


if __name__ == "__main__":
    transfrom = GeneralizedRCNNTransform(min_size=800, max_size=1300, 
                                        image_mean=[1, 1, 1], 
                                        image_std=[1, 1, 1])
    images = torch.randint(256, (1, 3, 1400, 600))
    print(images.shape)
    output = transfrom(images)
    imagelist, _ = output
    print(imagelist.tensors.shape)