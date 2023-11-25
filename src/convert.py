from argparse import ArgumentParser
import torch


def convert_checkpoint(sd):
    # Replace all of "model." with nothing
    sd = {k.replace("model.", ""): v for k, v in sd.items()}

    # Append "rtokens" to "register_tokens"
    if "register_tokens" not in sd:
        sd["register_tokens"] = sd["rtokens"]
    else:
        sd["register_tokens"] = torch.cat([sd["register_tokens"], sd["rtokens"]], dim=1)

    # Remove "rtokens"
    del sd["rtokens"]

    return sd


def main(args):
    sd = torch.load(args["checkpoint"], map_location="cpu")
    sd_new = convert_checkpoint(sd)

    # Save the converted model
    torch.save(sd_new, args["output"])
    print(f'Converted model saved to {args["output"]}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint file path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="store path of converted model"
    )
    args = vars(parser.parse_args())
    print(args)
    main(args)
