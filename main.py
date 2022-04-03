import argparse
import os
import sys
import math
import PIL
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable

from pwc.utils import estimate
from drn.utils import predict_flow
from tools.visualize import visualize_flow_heatmap, visualize_merge_heatmap, visualize_flow_heatmap_batched, visualize_merge_heatmap_batched
from tools.resize import resize_img, resize_shorter_side
from tools.face_detection import detect_face
from prediction.model import GeneratorConfig, DiscriminatorConfig, Generator, Discriminator
from prediction.losses import GeneratorLossFunction, DiscriminatorLossFunction
from dataset import GANDataset, gan_collate


def train(
    basepath: str, 
    outputpath: str, 
    no_crop: bool = False,
    batch_size: int = 2,
    num_workers: int = 8,
    num_epochs: int = 8,
    log_frequency: int = 50,
    learn_rate: float = 1e-4,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    gen_config = GeneratorConfig()
    dis_config = DiscriminatorConfig()

    generator = Generator(gen_config).to(device)
    discriminator = Discriminator(dis_config).to(device)

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        batch_size=batch_size,
        shaffle=True,
        num_workers=num_workers,
        collate_fn=gan_collate,
    )

    gen_loss = GeneratorLossFunction()
    dis_loss = DiscriminatorLossFunction()

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    for epoch in range(num_epochs):
        for idx, (modified, original, modified_path, original_path, modified_images, original_images) in tqdm(enumerate(dataloader)):
            real = Variable(torch.ones(original.shape[0], 1), requires_grad=False)
            fake = Variable(torch.zeros(modified.shape[0], 1), requires_grad=False)

            generator.train()
            predicted_flow = generator(modified.to(device))

            with torch.no_grad():
                generator.eval()
                flow, gt_flow, modified_nps, original_nps = generator.inference(
                    dataloader, predicted_flow, modified_path, original_path, modified_images, original_images)

            # (1) Update D network
            d_loss = dis_loss(discriminator(gt_flow), discriminator(flow), real, fake)
            dis_optimizer.zero_grad()
            d_loss.backward()
            dis_optimizer.step()

            # (2) Update G network
            classify_fake = discriminator(flow)
            gen_optimizer.zero_grad()
            classify_fake.backward()
            gen_optimizer.step()

            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(dataloader)}]: " + 
                    f"D Loss: {d_loss.item():.4f} " +
                    f"G Loss: {classify_fake.item():.4f} "
                )

                flow_output_path = f"{outputpath}/{epoch}_{idx}_flow.png"
                visualize_flow_heatmap_batched(flow, flow_output_path)

                merge_output_path = f"{outputpath}/{epoch}_{idx}_merge.png"
                visualize_merge_heatmap_batched(modified_nps, flow, merge_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify", required=True, help="the input of modified image")
    parser.add_argument("--origin", required=True, help="the input of original image")
    parser.add_argument("--model", required=True, help="the path to the drn model")
    parser.add_argument("--no_crop", action="store_true",
        help="do not use a face detector, instead run on the full input image")
    parser.add_argument("--output_dir", required=True, 
        help="the output directory of visualization")
    args = parser.parse_args()

    # DRN testing
    predicted_flow = predict_flow(args.modify, args.no_crop, model_path=args.model).cpu().numpy()
    predicted_flow = np.transpose(predicted_flow, (1, 2, 0))
    h, w, d = predicted_flow.shape

    if args.no_crop:
        img = Image.open(args.modify).convert('RGB')
    else:
        img, box = detect_face(args.modify)
    modified = resize_img(img, w, h)[0]
    modified_np = np.asarray(modified)
    visualize_flow_heatmap(predicted_flow, os.path.join(args.output_dir, 'drn_flow_heatmap.jpg'))
    visualize_merge_heatmap(modified_np, predicted_flow, os.path.join(args.output_dir, 'drn_merge_heatmap.jpg'))

    # PWC testing
    flow = estimate(args.modify, args.origin, args.no_crop, box, w, h).cpu().numpy()
    flow = np.transpose(flow, (1, 2, 0))
    fh, fw, fd = flow.shape

    assert(h == fh)
    assert(w == fw)
    print(f"flow: h = {fh}, w = {fw}, d = {d}, fd = {fd}")

    visualize_flow_heatmap(flow, os.path.join(args.output_dir, 'pwc_flow_heatmap.jpg'), 7.0)
    visualize_merge_heatmap(modified_np, flow, os.path.join(args.output_dir, 'pwc_merge_heatmap.jpg'), 7.0)
