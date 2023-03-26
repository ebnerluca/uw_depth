from os.path import dirname

pairs_file = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/MVSNet/pair.txt"

# read file
with open(pairs_file, "r") as f:
    lines = f.read().split("\n")

# store tuples
tuples = []
n_tuples = int(lines[0])
for i in range(n_tuples):
    pr_name = "%08d_init.pfm" % i
    gt_name = lines[1 + i * 2] + "_render.tif"
    tuples.append((pr_name, gt_name))

# write tuples
with open(dirname(pairs_file) + "/img_names.txt", "w") as f:
    for tuple in tuples:
        f.write(f"{tuple[0]} {tuple[1]}\n")
