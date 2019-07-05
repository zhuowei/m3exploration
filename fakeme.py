import foolbox
import m3inference
import torch
import numpy as np
import json
# Load static data for one user
with open("../m3webdemo/static/m3/xbox.json", "r") as infile:
    user_data = json.load(infile)
user_data["input"]["img_path"] = "../m3webdemo/" + user_data["input"]["img_path"]

# Load the image

dataset = m3inference.dataset.M3InferenceDataset([user_data["input"]], use_img=True)
dataset_tensors = list(dataset[0])
start_image = dataset_tensors[-1].numpy()

class M3ModifiedModel(m3inference.full_model.M3InferenceModel):
    def forward(self, data, label=None):
        newdata = dataset_tensors[:-1] + [data[0].float()]
        newdataloader = torch.utils.data.DataLoader(newdata, batch_size=1)
        output = super().forward(newdataloader, label)
        print(output)
        return output[2] # non-corp/corp

m3model = M3ModifiedModel(device="cpu")
m3model.eval()
m3model.load_state_dict(torch.load("../m3webdemo/full_model.mdl", map_location="cpu"))

print("Loaded state")
num_classes = 2 # corp/noncorp
fmodel = foolbox.models.PyTorchModel(m3model, bounds=(0, 1), num_classes=num_classes)
#one = fmodel.forward_one(start_image)
target_class = 0 # non-corp
criterion = foolbox.criteria.TargetClassProbability(target_class, p=0.99)
attack = foolbox.attacks.LBFGSAttack(fmodel, criterion)
adversarial = attack(start_image, 1)
