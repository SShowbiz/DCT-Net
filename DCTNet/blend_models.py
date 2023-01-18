import argparse
import math
import torch


def blend_models(par1, par2, level, blend_width=None, verbose=True):
    """ref: https://github.com/justinpinkney/stylegan2/blob/master/blend_models.py#L48-L89
    Blend layers between two StyleGAN2 models (for rosinality's version: https://github.com/rosinality/stylegan2-pytorch)
    par1: state_dict of low res model
    par2: state_dict of high res model
    level: [0:4, 1:8, 2:16, 3:32, 4:64, 5:128, 6:256, 7:512, 8:1024]
    """
    par_new = par1.copy()
    for k in par1.keys():
        params = k.split('.')
        if params[0] == 'style':
            continue
        if params[0] == 'convs':
            position = int(params[1]) // 2 + 1
        elif params[0] == 'to_rgbs':
            position = int(params[1])
        elif params[0] == 'noises':
            idx = int(params[1].split('_')[-1])
            position = (idx + 1) // 2
        else:
            position = 0

        x = position - level
        if blend_width:
            exponent = -x / blend_width
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1 if x > 1 else 0

        if verbose:
            print(f"Blending {k} by {y}")

        tmp = par2[k] * y + par1[k] * (1 - y)
        par_new[k] = tmp

    return par_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--blend_width", type=float, default=None)

    args = parser.parse_args()

    Gs = torch.load(args.model_path)['Gs']
    Gt_ema = torch.load(args.model_path)['gt_ema']
    Gt = torch.load(args.model_path)['Gt']

    model_result = blend_models(Gs, Gt_ema, args.level, args.blend_width)
    model_name = args.model_path.split('.')[0]
    torch.save(
        {"gt_ema": model_result, "Gs": Gs},
        f'{model_name}_level{args.level}_width{args.blend_width}.pth' if args.blend_width is not None else f'{model_name}_level{args.level}_hard.pth'
    )