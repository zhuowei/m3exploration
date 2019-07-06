import foolbox
import m3inference
import torch
import numpy as np
import json
import os

import imagewriter

# Load static data for one user

username = "2zhuowei"

# download user data with m3inference if needed

m3cachedir = "cachedir"

username_file = m3cachedir + "/" + username + ".json"
if not os.path.exists(username_file):
    print("Generating initial inference with m3inference")
    m3twitter = m3inference.M3Twitter(cache_dir=m3cachedir, model_dir="./")
    m3twitter.infer_screen_name(username)

with open(username_file, "r") as infile:
    user_data = json.load(infile)

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
        return output[1] # <18, 19-29, 30-39, >=40

m3model = M3ModifiedModel(device="cpu")
m3model.eval()
m3model.load_state_dict(torch.load("./full_model.mdl", map_location="cpu"))

print("Loaded state")
num_classes = 4 # 4 age groups
fmodel = foolbox.models.PyTorchModel(m3model, bounds=(0, 1), num_classes=num_classes)

# get model prediction for the current age group

forward_result = fmodel.forward_one(start_image)
cur_max = -1
cur_class = -1
for i in range(len(forward_result)):
    if forward_result[i] >= cur_max:
        cur_max = forward_result[i]
        cur_class = i
print(cur_class)

# run the attack, targeting one other age group

class TargetClassProbabilityPostSoftmax(foolbox.criteria.TargetClassProbability):
    # our nn already includes a softmax, so the target class doesn't need to do it
    def is_adversarial(self, predictions, label):
        return predictions[self.target_class()] > self.p

target_class = 3 # >=40
criterion = TargetClassProbabilityPostSoftmax(target_class, p=0.9)
attack = foolbox.attacks.LBFGSAttack(fmodel, criterion)
adversarial = attack(start_image, 1, maxiter=20)

# write output in pickle and png format
import pickle
with open("out.pickle", "wb") as outfile:
	pickle.dump(adversarial, outfile)
imagewriter.writeresult(adversarial, "output_" + username + ".png")

print("Adversarial output generated at output_" + username + ".png!")
