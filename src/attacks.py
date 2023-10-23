import torch
from torch.nn.functional import cosine_similarity, mse_loss

from data.transforms import normalize, unnormalize


def psnr_to_mse(psnr):
    return 10 ** (-psnr / 10)


def quantize(batch):
    imgs = [unnormalize(img).clamp(0, 1) for img in batch]
    imgs = [normalize(img) for img in imgs]
    batch = torch.stack(imgs).cuda()
    return batch


def pgd_attack(model, batch, steps=350, lr=4e-4, max_mse=1e-4, eps=None, verbose=False):
    model.eval()
    with torch.no_grad():
        target = model(batch).detach()
        img_target = unnormalize(batch).detach()

    l, u = batch.min(), batch.max()
    batch_adv = (batch + lr * torch.randn_like(batch)).detach().clone()
    batch_adv.requires_grad = True
    for step in range(steps):
        # Minimize cosine similarity an MSE
        pred = model(batch_adv)
        img_pred = unnormalize(batch_adv)

        f_loss = cosine_similarity(pred, target).mean()
        i_loss = mse_loss(img_pred, img_target).mean()
        loss = f_loss + i_loss

        loss.backward()
        batch_adv.data = batch_adv.data - lr * batch_adv.grad.sign()
        batch_adv.grad.zero_()

        # Clamp perturbation
        if eps is not None:
            delta = batch_adv - batch
            delta = torch.clamp(delta, -eps, eps)
            batch_adv = batch + delta

        # Stay within range
        batch_adv = torch.clamp(batch_adv, l, u).clone().detach()
        batch_adv.requires_grad = True

        # Print progress
        if verbose and ((step + 1) % (steps // 10) == 0 or step == 0):
            print(f"Step {step+1}/{steps}: {loss.item():.3f}")

        # Stop if MSE is too high
        if max_mse is not None:
            if i_loss >= max_mse:
                break

    # Quantizing
    batch_adv = quantize(batch_adv)
    return batch_adv.clone().detach()
