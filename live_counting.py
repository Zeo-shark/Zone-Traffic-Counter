import torch
import matplotlib as plt
from cannet import CANNet
from torchvision import transforms
import argparse
import cv2
import copy
import os

def main_live(args):
    capture= cv2.VideoCapture(args.stream)
    num_gpus= torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'
    model= CANNet().to(device)
    model.load_state_dict(torch.load(args.weights,map_location=torch.device(device)))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    model.eval()