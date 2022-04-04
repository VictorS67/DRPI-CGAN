import argparse
import os
import sys
import math
import PIL
import torch
import random
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
from prediction.modules import ConvGANDiscriminator
from prediction.losses import GeneratorLossFunction, DiscriminatorLossFunction
from dataset import GANDataset, gan_collate


def overfit(
    basepath: str, 
    outputpath: str, 
    no_crop: bool = False,
    num_iterations: int = 700,
    log_frequency: int = 50,
    learn_rate: float = 1e-4,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(outputpath, exist_ok=True)

    gen_config = GeneratorConfig()
    dis_config = DiscriminatorConfig()

    # print("set configs!")

    generator = Generator(gen_config).to(device)
    discriminator = ConvGANDiscriminator(dis_config).to(device)

    # print("set gan!")

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        collate_fn=gan_collate,
    )

    # print("set dataset!")

    gen_loss = GeneratorLossFunction()
    dis_loss = GeneratorLossFunction()

    # print("set loss function!")

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    # print("set optimizer!")
    modified, modified_data, original_data = next(iter(dataloader))
    for idx in tqdm(range(num_iterations)):
        # print(f"training step: {idx}")
        generator.zero_grad()
        generator.train()

        predicted_flow = generator(modified.to(device))

        generator.eval()
        flow, gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)

        # (1) Update D network
        discriminator.zero_grad()
        discriminator.train()

        classify_real = discriminator(gt_flow.to(device))
        classify_fake = discriminator(flow.to(device))

        real_loss = dis_loss(classify_real, True)
        real_loss.backward()

        fake_loss = dis_loss(classify_fake, False)
        fake_loss.backward()

        d_loss = real_loss + fake_loss
        dis_optimizer.step()

        # (2) Update G network
        discriminator.eval()
        classify_fake = gen_loss(discriminator(flow.to(device)), True)

        classify_fake.backward()
        gen_optimizer.step()

        if (idx + 1) % log_frequency == 0:
            print(
                f"Epoch [{idx}/{num_iterations}]: " + 
                f"D Loss: {d_loss.item():.4f} " +
                f"G Loss: {classify_fake.item():.4f} "
            )

            flow_output_path = f"{outputpath}/{idx}_flow.png"
            visualize_flow_heatmap_batched(flow.cpu().numpy(), flow_output_path, max_flow_mag=50.0)

            merge_output_path = f"{outputpath}/{idx}_merge.png"
            visualize_merge_heatmap_batched(modified_nps, flow.cpu().numpy(), merge_output_path, max_flow_mag=50.0)

            flow_output_path = f"{outputpath}/{idx}_flow_gt.png"
            visualize_flow_heatmap_batched(gt_flow.cpu().numpy(), flow_output_path)

            merge_output_path = f"{outputpath}/{idx}_merge_gt.png"
            visualize_merge_heatmap_batched(modified_nps, gt_flow.cpu().numpy(), merge_output_path)


def train(
    basepath: str, 
    outputpath: str, 
    no_crop: bool = False,
    batch_size: int = 1,
    num_workers: int = 8,
    num_epochs: int = 8,
    log_frequency: int = 50,
    learn_rate: float = 1e-4,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
    os.makedirs(outputpath, exist_ok=True)
    
    gen_config = GeneratorConfig()
    dis_config = DiscriminatorConfig()

    # print("set configs!")

    generator = Generator(gen_config).to(device)
    discriminator = ConvGANDiscriminator(dis_config).to(device)

    # print("set gan!")

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=gan_collate,
    )

    # print("set dataset!")

    gen_loss = GeneratorLossFunction()
    dis_loss = DiscriminatorLossFunction()

    # print("set loss function!")

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    # print("set optimizer!")

    for epoch in range(num_epochs):
        # print(f"epoch {epoch}:")
        for idx, (modified, modified_data, original_data) in tqdm(enumerate(dataloader)):
            # print(f"training step: {idx}")

            generator.train()
            predicted_flow = generator(modified.to(device))

            # print(f"predicted_flow: {predicted_flow.shape}")

            # real = Variable(torch.ones(modified.shape[0], 1), requires_grad=False)
            # fake = Variable(torch.zeros(modified.shape[0], 1), requires_grad=False)
            # print(f"real: {real.shape}, fake: {fake.shape}")

            with torch.no_grad():
                generator.eval()
                flow, gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)
                # print(f"flow: {flow.shape}, gt_flow: {gt_flow.shape}, modified_nps: {modified_nps.shape}")

            # (1) Update D network
            d_loss = Variable(dis_loss(discriminator(gt_flow), discriminator(flow)), requires_grad=True)
            dis_optimizer.zero_grad()
            d_loss.backward()
            dis_optimizer.step()

            # print(f"d_loss: {d_loss.shape}")

            # (2) Update G network
            classify_fake = Variable(gen_loss(discriminator(flow), True), requires_grad=True)
            gen_optimizer.zero_grad()
            classify_fake.backward()
            gen_optimizer.step()

            # print(f"classify_fake: {classify_fake.shape}")

            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(dataloader)}]: " + 
                    f"D Loss: {d_loss.item():.4f} " +
                    f"G Loss: {classify_fake.item():.4f} "
                )

                # flow_output_path = f"{outputpath}/{epoch}_{idx}_flow.png"
                # visualize_flow_heatmap_batched(flow.cpu().numpy(), flow_output_path)

                # print("visualize_flow_heatmap_batched completed!")

                # merge_output_path = f"{outputpath}/{epoch}_{idx}_merge.png"
                # visualize_merge_heatmap_batched(modified_nps, flow.cpu().numpy(), merge_output_path)

                # print("visualize_merge_heatmap_batched completed!")


if __name__ == "__main__":
    import fire

    fire.Fire()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--modify", required=True, help="the input of modified image")
    # parser.add_argument("--origin", required=True, help="the input of original image")
    # parser.add_argument("--model", required=True, help="the path to the drn model")
    # parser.add_argument("--no_crop", action="store_true",
    #     help="do not use a face detector, instead run on the full input image")
    # parser.add_argument("--output_dir", required=True, 
    #     help="the output directory of visualization")
    # args = parser.parse_args()

    # # DRN testing
    # predicted_flow = predict_flow(args.modify, args.no_crop, model_path=args.model).cpu().numpy()
    # predicted_flow = np.transpose(predicted_flow, (1, 2, 0))
    # h, w, d = predicted_flow.shape

    # if args.no_crop:
    #     img = Image.open(args.modify).convert('RGB')
    # else:
    #     img, box = detect_face(args.modify)
    # modified = resize_img(img, w, h)[0]
    # modified_np = np.asarray(modified)
    # visualize_flow_heatmap(predicted_flow, os.path.join(args.output_dir, 'drn_flow_heatmap.jpg'))
    # visualize_merge_heatmap(modified_np, predicted_flow, os.path.join(args.output_dir, 'drn_merge_heatmap.jpg'))

    # # PWC testing
    # flow = estimate(args.modify, args.origin, args.no_crop, box, w, h).cpu().numpy()
    # flow = np.transpose(flow, (1, 2, 0))
    # fh, fw, fd = flow.shape

    # assert(h == fh)
    # assert(w == fw)
    # print(f"flow: h = {fh}, w = {fw}, d = {d}, fd = {fd}")

    # visualize_flow_heatmap(flow, os.path.join(args.output_dir, 'pwc_flow_heatmap.jpg'), 7.0)
    # visualize_merge_heatmap(modified_np, flow, os.path.join(args.output_dir, 'pwc_merge_heatmap.jpg'), 7.0)
