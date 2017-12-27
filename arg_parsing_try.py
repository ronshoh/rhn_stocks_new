import argparse

ap = argparse.ArgumentParser()

ap.add_argument("--depth", type=int, nargs=1, default=None)
ap.add_argument("--adaptive_optimizer", type=str, nargs=1, default=None)
ap.add_argument("--decay_epochs", type=int, nargs='*', default=None)
ap.add_argument("--lr_decay", type=float, nargs='*', default=None)
ap.add_argument("--lr", type=float, nargs='*', default=None)


args = ap.parse_args()

for arg in vars(args):
    if getattr(args, arg) is not None:
        if len(getattr(args, arg)) == 1:
            print(arg, getattr(args, arg)[0])
        else:
            print(arg, getattr(args, arg))




class Config():
    depth = 4
    decay_epochs = 4
    lr_decay = 4.2
    adaptive_optimizer = "fg"



print(Config.__dict__)


for arg in vars(args):
    if getattr(args, arg) is not None:
        if len(getattr(args, arg)) == 1:
            setattr(Config, arg, getattr(args, arg)[0])
        else:
            setattr(Config, arg, getattr(args, arg))

print(Config.__dict__)








# for arg in args.__dict__:
#     if args.__dict__[arg] is not None:
#         print(arg + '==' + str(args.__dict__[arg]))
