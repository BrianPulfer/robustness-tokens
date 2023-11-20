from attacks.pgd import pgd_attack

SUPPORTED_ATTACKS = ["pgd"]


def get_attack(model, **kwargs):
    name = kwargs["name"]

    if name == "pgd":

        def attack(model, batch):
            mode = model.enable_robust
            model.enable_robust = False
            adv_batch = pgd_attack(
                model, batch, **{k: v for k, v in kwargs.items() if k != "name"}
            )
            model.enable_robust = mode
            return adv_batch

        return attack

    raise KeyError(f"Attack {name} not supported. Pick one of {SUPPORTED_ATTACKS}")
