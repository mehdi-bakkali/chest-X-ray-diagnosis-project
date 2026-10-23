"""Create a Grad-CAM heatmap for one label and image (qualitative only)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from .labels import LABELS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='outputs/best_model.pt'); p.add_argument('--image',required=True); p.add_argument('--label',required=True,choices=LABELS); p.add_argument('--out',default='outputs/gradcam.png'); args=p.parse_args()
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=models.densenet121(weights=None); model.classifier=nn.Linear(model.classifier.in_features,len(LABELS))
    ckpt=torch.load(args.checkpoint,map_location='cpu'); model.load_state_dict(ckpt['model_state_dict']); model.to(device).eval()
    activations=[]; gradients=[]; target_layer=model.features.denseblock4
    target_layer.register_forward_hook(lambda m,i,o: activations.append(o))
    target_layer.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0]))
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    original=Image.open(args.image).convert('RGB').resize((224,224)); x=tfm(original).unsqueeze(0).to(device)
    idx=LABELS.index(args.label); model.zero_grad(); logits=model(x); score=logits[0,idx]; score.backward()
    acts=activations[0].detach(); grads=gradients[0].detach(); weights=grads.mean(dim=(2,3),keepdim=True); cam=torch.relu((weights*acts).sum(dim=1,keepdim=True)); cam=torch.nn.functional.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0].cpu().numpy(); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(6,6)); plt.imshow(original); plt.imshow(cam,alpha=0.4,cmap='jet'); plt.axis('off'); plt.title(f'Grad-CAM: {args.label} (exploratory)'); plt.tight_layout(); plt.savefig(out,dpi=160); print('Saved',out)
    print('Grad-CAM is a qualitative visualization, not proof of a diagnosis or model validity.')
if __name__=='__main__': main()
