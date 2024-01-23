import torch
from torch.nn.functional import cosine_similarity, cross_entropy, mse_loss

from data.transforms import to_tensor, to_pil, normalize, unnormalize


def validate(batch):
    imgs = [to_pil(unnormalize(img).clamp(0, 1)) for img in batch]
    imgs = [normalize(to_tensor(img)) for img in imgs]
    batch = torch.stack(imgs).cuda()
    return batch


def pgd_attack(
    model,
    batch,
    target=None,
    steps=30,
    lr=1e-2,
    max_mse=1e-4,
    eps=8 / 255,
    verbose=False,
    ignore_index=-100,
    **forward_kwargs,
):
    model.eval()
    img = unnormalize(batch).detach()

    on_features = target is None
    objective = (
        cosine_similarity
        if on_features
        else lambda x, y: -cross_entropy(x, y, ignore_index=ignore_index)
    )

    if on_features:
        with torch.no_grad():
            target = model(batch, **forward_kwargs).detach()

    lower, upper = batch.min(), batch.max()
    batch_adv = (batch + lr * torch.randn_like(batch)).detach().clone()
    batch_adv.requires_grad = True
    for step in range(steps):
        # Minimize cosine similarity and MSE
        pred = model(batch_adv, **forward_kwargs)
        img_adv = unnormalize(batch_adv)

        f_loss = objective(pred, target).mean()
        i_loss = mse_loss(img_adv, img).mean()
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


if __name__ == "__main__":
    from data.utils import get_loaders_fn

    device = "cuda"
    model = (
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc").to(device).eval()
    )

    loader, _ = get_loaders_fn("imagenet")(batch_size=16)

    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y)

        with torch.no_grad():
            y_hat = model(x.to(device))
            y_hat_adv = model(x_adv.to(device))

            print(f"Accuracy: {(y_hat.argmax(dim=-1) == y).float().mean()}.")
            print(f"Accuracy adv: {(y_hat_adv.argmax(dim=-1) == y).float().mean()}.")
        break
