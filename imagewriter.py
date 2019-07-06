import pickle
#import PIL.Image
import torch
import torchvision.utils
import numpy as np
def writeresult(result, outfilename):
    #result2 = result.reshape((224, 224, 3))
    #result3 = np.uint8(result2*255)
    #im = PIL.Image.fromarray(result3)
    #im.save(outfilename)
    torchvision.utils.save_image(torch.from_numpy(result), outfilename)

def main():
    with open("out.pickle", "rb") as infile:
    	indata = pickle.load(infile)
    writeresult(indata, "output.png")

if __name__ == "__main__":
    main()
