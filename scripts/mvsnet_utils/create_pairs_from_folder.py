from os.path import join, basename, splitext
from glob import glob
from argparse import ArgumentParser

# arguments
parser = ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--pattern", type=str, default=".png")
parser.add_argument("--n_views", type=int, default=3)
args = parser.parse_args()

# find imgs
imgs = glob(join(args.folder, "*" + args.pattern))
for i in range(len(imgs)):
    imgs[i] = splitext(basename(imgs[i]))[0]
imgs.sort()

# find target imgs
window_half = int(args.n_views / 2)
target_imgs = imgs[window_half:-window_half]

# find source imgs
source_img_lists = []
for i in range(len(target_imgs)):
    source_img_lists.append(
        imgs[i : i + args.n_views]
    )  # add all imgs in window as source
    source_img_lists[-1].remove(target_imgs[i])  # remove target img from sources

out = join(args.folder, "pair.txt")
with open(out, "w") as f:
    f.write(str(len(target_imgs)) + "\n")  # how many
    for target_img, source_imgs in zip(target_imgs, source_img_lists):
        # write current target
        f.write(target_img + "\n")

        # write num of source imgs
        f.write(str(len(source_imgs)) + " ")

        # join source imgs and scores in alternating order
        scores = [str(0.0)] * len(source_imgs)  # TODO: scores are zero for now
        sources_and_scores = [None] * 2 * len(source_imgs)
        sources_and_scores[::2] = source_imgs
        sources_and_scores[1::2] = scores

        # write source imgs and scores
        f.write(" ".join(sources_and_scores) + "\n")
