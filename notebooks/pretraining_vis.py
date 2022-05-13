import sys
sys.path.append('../')
from models.encoder_only import EncoderOnly
from datasets.ms_dataset import MsDataModule
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA

# load model
checkpoint_dir = "/workspace/volume/tensorboard/lightning_logs/pretrain/checkpoints/last.ckpt"
model = EncoderOnly()
model.load_from_checkpoint(checkpoint_dir)

# get test encodings
data_module = MsDataModule(batch_size = 128)
data_module.setup("test")
ccmslids = []
encodings = None
# NOTE: more memory efficient way to do this, just stop early for now because I'm lazy
max_batches = 9
print("GENERATING ENCODINGS")
for batchnum ,(ms, fp, ids) in enumerate(tqdm(data_module.test_dataloader())):
    if batchnum >= max_batches: break
    ccmslids.append(ids)
    curr = model.encode(ms).squeeze(1)
    if encodings is not None:
        encodings = torch.cat((curr, encodings))
    else:
        encodings = curr

ccmslids = [item for batch in ccmslids for item in batch]
encodings = encodings.detach().numpy()
pca = PCA(n_components = 3)
pca.fit(encodings)
encodings = pca.transform(encodings)
id_to_pathway = json.load(open("/workspace/volume/data/id_to_pathway.json"))
id_to_superclass = json.load(open("/workspace/volume/data/id_to_superclass.json"))

pathway_to_points = {}
superclass_to_points = {}
for encoding, ccmslid in zip(encodings, ccmslids):
    pathways = id_to_pathway[ccmslid[:-4]]
    for pathway in pathways:
        if pathway not in pathway_to_points:
            pathway_to_points[pathway] = [[],[],[]]
        pathway_to_points[pathway][0].append(float(encoding[0]))
        pathway_to_points[pathway][1].append(float(encoding[1]))
        pathway_to_points[pathway][2].append(float(encoding[2]))

    superclasses = id_to_superclass[ccmslid[:-4]]
    for superclass in superclasses:
        if superclass not in superclass_to_points:
            superclass_to_points[superclass] = [[],[],[]]
        superclass_to_points[superclass][0].append(float(encoding[0]))
        superclass_to_points[superclass][1].append(float(encoding[1]))
        superclass_to_points[superclass][2].append(float(encoding[2]))

with open("pathway_to_points.json", "w") as f:
    json.dump(pathway_to_points, f)
with open("superclass_to_points.json", "w") as f:
    json.dump(superclass_to_points, f)

# PLOTTING
def plot(cluster_to_points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for cluster, dimensions in cluster_to_points.items():
        print(f"{cluster}: {len(dimensions[0])}")
        cluster_to_points[cluster] = ax.scatter(dimensions[0], dimensions[1], dimensions[2], label=cluster)
    legend = plt.legend(loc="upper right")
    labels = legend.get_texts()
    for label in labels:
        label.set_picker(True)

    def handle_pick(event):
        text = event.artist
        is_visible = cluster_to_points[text.get_text()].get_visible()
        cluster_to_points[text.get_text()].set_visible(not is_visible)
        fig.canvas.draw()

    plt.connect('pick_event', lambda event: handle_pick(event))
    plt.show()
    plt.savefig("test.png")


plot(superclass_to_points)