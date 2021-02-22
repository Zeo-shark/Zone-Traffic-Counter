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
    while(capture.isOpened()):
        ret,frame= capture.read()
        if (ret):
            key= cv2.waitKey(1) & 0xFF

            copy_frame = copy.copy(frame)

            copy_frame = transform(copy_frame)

            # putting the batch dimension on tensor
            copy_frame = copy_frame.unsqueeze(0)
            copy_frame = copy_frame.to(device)
            result = model(copy_frame)

            crowd_count = int(result.data.sum().item())

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Crowd Couting: " + str(crowd_count), (10, 35), font, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("Frame", frame)