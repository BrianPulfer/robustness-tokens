import torch
from torch.nn.functional import cosine_similarity, mse_loss

from data.transforms import normalize, to_pil, to_tensor, unnormalize


def psnr_to_mse(psnr):
    return 10 ** (-psnr / 10)


def validate(batch):
    imgs = [to_pil(unnormalize(img).clamp(0, 1)) for img in batch]
    imgs = [normalize(to_tensor(img)) for img in imgs]
    batch = torch.stack(imgs).cuda()
    return batch


def pgd_attack(
    model, batch, steps=30, lr=1e-2, max_mse=1e-4, eps=8 / 255, verbose=False
):
    model.eval()
    with torch.no_grad():
        target = model(batch).detach()
        img_target = unnormalize(batch).detach()

    lower, upper = batch.min(), batch.max()
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

        # Projection
        if eps is not None:
            batch_adv = batch + torch.clamp(batch_adv - batch, -eps, eps)

        # Stay within range
        batch_adv = torch.clamp(batch_adv, lower, upper).clone().detach()
        batch_adv.requires_grad = True

        # Print progress
        if verbose and ((step + 1) % (steps // 10) == 0 or step == 0):
            print(f"Step {step+1}/{steps}: {loss.item():.3f}")

        # Stop if MSE is too high
        if max_mse is not None:
            if i_loss >= max_mse:
                break

    # Getting valid image tensors
    batch_adv = validate(batch_adv)
    return batch_adv.clone().detach()
