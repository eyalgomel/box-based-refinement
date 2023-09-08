from pathlib import Path
import numpy as np
import torch

# Some of this code is borrowed from https://github.com/talshaharabany/what-is-where-by-looking


def get_root_path():
    current_file = Path(__file__)
    return current_file.parents[1].resolve().as_posix()


def no_tuple(a):
    out = []
    for item in a:
        out.append(item[0])
    return out


def interpret(image, text, model, device, index=None):
    logits_per_image, logits_per_text = model(image, text)
    logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=False)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    for i, blk in enumerate(image_attn_blocks):
        if i <= 10:
            continue
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.detach().reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode="bilinear")
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    del image_attn_blocks, R, one_hot, grad, cam
    torch.cuda.empty_cache()
    return image_relevance


def interpret_batch(image, text, model, device, index=None, ground=False):
    bs = image.shape[0]
    logits_per_image, logits_per_text = model(image, text)
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((bs, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[:, index] = 1
    if ground:
        one_hot = np.eye(bs, dtype=np.float32)
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).cuda()
    one_hot = torch.sum(one_hot * logits_per_image, dim=1).mean()
    model.zero_grad()
    one_hot.backward(retain_graph=False)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = (
        torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype)
        .to("cuda:" + str(image.get_device()))
        .repeat(bs, 1, 1)
    )
    for i, blk in enumerate(image_attn_blocks):
        if i <= 10:
            continue
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(bs, -1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(bs, -1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=1)
        R += torch.matmul(cam, R)
    R[:, 0, 0] = 0
    image_relevance = R[:, 0, 1:].detach().clone()

    dim = int(image_relevance.shape[1] ** 0.5)
    image_relevance = image_relevance.reshape(bs, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode="bilinear")

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance


def interpret_new(images, texts, model, device):
    bs = images.shape[0]
    batch_size = texts.shape[0]
    logits_per_image, logits_per_text = model(images, texts)
    logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < 11:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:].detach().clone()

    dim = int(image_relevance.shape[1] ** 0.5)
    image_relevance = image_relevance.reshape(bs, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode="bilinear")

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance
