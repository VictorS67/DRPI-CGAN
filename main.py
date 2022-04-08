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

    generator = Generator(gen_config).to(device)
    discriminator = ConvGANDiscriminator(dis_config).to(device)

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        collate_fn=gan_collate,
    )

    gen_loss = GeneratorLossFunction()
    dis_loss = GeneratorLossFunction()

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    modified, modified_data, original_data = next(iter(dataloader))
    for idx in tqdm(range(num_iterations)):
        # (1) Update G network
        generator.zero_grad()
        discriminator.eval()
        generator.train()

        # 1.1 Get noise from modified image
        noise = Variable(modified.to(device))

        # 1.2 Generate fake flow from the noise
        predicted_flow = generator(noise)  # [batch_size x D x H x W]
        # print(f"predicted_flow: {predicted_flow.requires_grad}")
        classify_fool = discriminator(predicted_flow)  # [batch_size x D x H' x W']
        fool = Variable(torch.ones_like(classify_fool).float().to(device), requires_grad=False)  # [batch_size x D x H' x W']

        # 1.3 Compute the generator loss on the fake flow
        G_loss = gen_loss(classify_fool, fool)
        # G_loss.register_hook(lambda grad: print(f"G_loss: {grad}"))

        #  1.4 Compute the gradients and run SGD on generator's parameters
        G_loss.backward()
        gen_optimizer.step()

        # (2) Update D network
        discriminator.zero_grad()
        generator.eval()
        discriminator.train()

        # 2.1 Get noise from modified image
        noise = Variable(modified.to(device))

        # 2.2 Generate fake flow from the noise
        predicted_flow = generator(noise)  # [batch_size x D x H x W]
        classify_fake = discriminator(predicted_flow.detach())  # [batch_size x D x H' x W']
        fake = Variable(torch.zeros_like(classify_fake).float().to(device), requires_grad=False)  # [batch_size x D x H' x W']

        # 2.3 Compute the discriminator loss on the fake flow
        D_fake_loss = dis_loss(classify_fake, fake)
        # D_fake_loss.register_hook(lambda grad: print(f"D_fake_loss: {grad}"))
        # D_fake_loss.backward()

        # 2.4 Form the GT flow by PWC-Net
        with torch.no_grad():
            gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)

        gt_flow = Variable(gt_flow.to(device))  # [batch_size x D x H x W]
        classify_real = discriminator(gt_flow)  # [batch_size x D x H' x W']
        real = Variable(torch.ones_like(classify_real).float().to(device), requires_grad=False)  # [batch_size x D x H' x W']

        # 2.5 Compute the discriminator loss on the GT flow
        D_real_loss = dis_loss(classify_real, real)
        # D_real_loss.register_hook(lambda grad: print(f"D_real_loss: {grad}"))
        # D_real_loss.backward()

        # 2.6 Compute the total discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2

        # 2.7 Compute the gradients and run SGD on discriminator's parameters
        D_total_loss.backward()
        dis_optimizer.step()

        if (idx + 1) % log_frequency == 0:
            print(
                f"Epoch [{idx}/{num_iterations}]: " + 
                f"D Loss: {D_total_loss.item():.4f} " +
                f"G Loss: {G_loss.item():.4f} "
            )

            flow_output_path = f"{outputpath}/{idx}_flow.png"
            # print(f"predicted_flow: {np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)).shape}")
            # print(f"modified_nps: {modified_nps.shape}")
            visualize_flow_heatmap_batched(np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path, max_flow_mag=50.0)

            merge_output_path = f"{outputpath}/{idx}_merge.png"
            visualize_merge_heatmap_batched(modified_nps, np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path, max_flow_mag=50.0)

            flow_output_path = f"{outputpath}/{idx}_flow_gt.png"
            visualize_flow_heatmap_batched(np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path)

            merge_output_path = f"{outputpath}/{idx}_merge_gt.png"
            visualize_merge_heatmap_batched(modified_nps, np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path)


def train(
    basepath: str, 
    outputpath: str, 
    no_crop: bool = False,
    batch_size: int = 1,
    num_workers: int = 8,
    num_epochs: int = 8,
    log_frequency: int = 24,
    learn_rate: float = 1e-4,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    os.makedirs(outputpath, exist_ok=True)

    gen_config = GeneratorConfig()
    dis_config = DiscriminatorConfig()

    generator = Generator(gen_config).to(device)
    discriminator = ConvGANDiscriminator(dis_config).to(device)

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=gan_collate,
    )

    gen_loss = GeneratorLossFunction()
    dis_loss = GeneratorLossFunction()

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    for epoch in range(num_epochs):
        for idx, (modified, modified_data, original_data) in tqdm(enumerate(dataloader)):
            # (1) Update G network
            generator.zero_grad()
            discriminator.eval()
            generator.train()

            # 1.1 Get noise from modified image
            noise = Variable(modified.to(device))

            # 1.2 Generate fake flow from the noise
            predicted_flow = generator(noise)  # [batch_size x D x H x W]
            # print(f"predicted_flow: {predicted_flow.requires_grad}")
            classify_fool = discriminator(predicted_flow)  # [batch_size x D x H' x W']
            fool = Variable(torch.ones_like(classify_fool).float().to(device), requires_grad=False)  # [batch_size x D x H' x W']

            # 1.3 Compute the generator loss on the fake flow
            G_loss = gen_loss(classify_fool, fool)
            # G_loss.register_hook(lambda grad: print(f"G_loss: {grad}"))

            #  1.4 Compute the gradients and run SGD on generator's parameters
            G_loss.backward()
            gen_optimizer.step()

            # (2) Update D network
            discriminator.zero_grad()
            generator.eval()
            discriminator.train()

            # 2.1 Get noise from modified image
            noise = Variable(modified.to(device))

            # 2.2 Generate fake flow from the noise
            predicted_flow = generator(noise)  # [batch_size x D x H x W]
            classify_fake = discriminator(predicted_flow.detach())  # [batch_size x D x H' x W']
            fake = Variable(torch.zeros_like(classify_fake).float().to(device), requires_grad=False)  # [batch_size x D x H' x W']

            # 2.3 Compute the discriminator loss on the fake flow
            D_fake_loss = dis_loss(classify_fake, fake)
            # D_fake_loss.register_hook(lambda grad: print(f"D_fake_loss: {grad}"))
            # D_fake_loss.backward()

            # 2.4 Form the GT flow by PWC-Net
            with torch.no_grad():
                gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)

            gt_flow = Variable(gt_flow.to(device))  # [batch_size x D x H x W]
            classify_real = discriminator(gt_flow)  # [batch_size x D x H' x W']
            real = Variable(torch.ones_like(classify_real).float().to(device), requires_grad=False)  # [batch_size x D x H' x W']

            # 2.5 Compute the discriminator loss on the GT flow
            D_real_loss = dis_loss(classify_real, real)
            # D_real_loss.register_hook(lambda grad: print(f"D_real_loss: {grad}"))
            # D_real_loss.backward()

            # 2.6 Compute the total discriminator loss
            D_total_loss = (D_real_loss + D_fake_loss) / 2

            # 2.7 Compute the gradients and run SGD on discriminator's parameters
            D_total_loss.backward()
            dis_optimizer.step()

            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(dataloader)}]: " + 
                    f"D Loss: {D_total_loss.item():.4f} " +
                    f"G Loss: {G_loss.item():.4f} "
                )

                flow_output_path = f"{outputpath}/{epoch}_{idx}_flow.png"
                visualize_flow_heatmap_batched(np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path, max_flow_mag=50.0)

                merge_output_path = f"{outputpath}/{epoch}_{idx}_merge.png"
                visualize_merge_heatmap_batched(modified_nps, np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path, max_flow_mag=50.0)

                flow_output_path = f"{outputpath}/{epoch}_{idx}_flow_gt.png"
                visualize_flow_heatmap_batched(np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path)

                merge_output_path = f"{outputpath}/{epoch}_{idx}_merge_gt.png"
                visualize_merge_heatmap_batched(modified_nps, np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path)


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
