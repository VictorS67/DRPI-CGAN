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
from drn.utils import preprocess as drn_preprocess
from tools.visualize import *
from tools.resize import resize_img, resize_shorter_side, crop_img
from tools.face_detection import detect_face
from tools.warp import warp
from prediction.model import GeneratorConfig, DiscriminatorConfig, Generator, Discriminator
from prediction.modules import ConvGANDiscriminator
from prediction.losses import BCELossFunction, LSLossFunction, MixedGenLossFunction
from dataset import GANDataset, gan_collate


def overfit(
    basepath: str, 
    outputpath: str, 
    no_crop: bool = False,
    num_iterations: int = 5000,
    log_frequency: int = 100,
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

    gen_loss = MixedGenLossFunction()
    dis_loss = BCELossFunction()

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    modified, original, modified_data, original_data = next(iter(dataloader))
    for idx in tqdm(range(num_iterations)):
        original = Variable(original.to(device))
        # print(f"original: {original.shape} - {original.requires_grad}")

        # (1) Update G network
        generator.zero_grad()
        discriminator.eval()
        generator.train()

        # 1.1 Get noise from modified image
        noise = Variable(modified.to(device))
        # print(f"noise: {noise.shape}")

        # 1.2 Generate fake flow from the noise
        predicted_flow = generator(noise)  # [batch_size x D x H x W]
        # print(f"gradients: {predicted_flow.grad}")
        # predicted_flow_fool = predicted_flow
        # print(f"predicted_flow: {predicted_flow.shape}")
        # print(f"predicted_flow: {predicted_flow.requires_grad}")

        with torch.no_grad():
            gt_flow_epe, _ = generator.inference(predicted_flow, modified_data, original_data, no_crop)

        predicted_flow_warpped = warp(noise, predicted_flow)
        predicted_flow_fool = original - predicted_flow_warpped
        # predicted_flow_G = predicted_flow_fool

        _, _, H, W = predicted_flow_fool.shape
        sampled_point = np.random.choice(np.arange(min(H, W) - 64), 1).item()
        predicted_flow_G = predicted_flow_fool[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]

        # predicted_flow_epe_patched = predicted_flow[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
        # gt_flow_epe_patched = gt_flow_epe[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]

        # predicted_flow_warpped_patched = predicted_flow_warpped[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
        # original_patched = original[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]

        classify_fool = discriminator(predicted_flow_G)  # [batch_size x 1 x 1 x 1]
        fool = Variable(torch.ones_like(classify_fool).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

        # 1.3 Compute the generator loss on the fake flow
        # G_loss = gen_loss(classify_fool, fool)
        G_loss = gen_loss(predicted_flow, gt_flow_epe, predicted_flow_warpped, original, classify_fool, fool)

        # 1.4 Compute the gradients and run SGD on generator's parameters
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
        # predicted_flow_fake = predicted_flow

        predicted_flow_fake = original - warp(noise, predicted_flow)
        # predicted_flow_D_fake = predicted_flow_fake

        _, _, H, W = predicted_flow_fake.shape
        sampled_point = np.random.choice(np.arange(min(H, W) - 64), 1).item()
        predicted_flow_D_fake = predicted_flow_fake[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
        classify_fake = discriminator(predicted_flow_D_fake.detach())  # [batch_size x 1 x 1 x 1]
        fake = Variable(torch.zeros_like(classify_fake).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

        # 2.3 Compute the discriminator loss on the fake flow
        D_fake_loss = dis_loss(classify_fake, fake)
        # D_fake_loss.register_hook(lambda grad: print(f"D_fake_loss: {grad}"))
        # D_fake_loss.backward()

        # 2.4 Form the GT flow by PWC-Net
        with torch.no_grad():
            gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)

        gt_flow = Variable(gt_flow.to(device))  # [batch_size x D x H x W]
        # gt_flow_real = gt_flow

        gt_flow_real = original - warp(noise, gt_flow)
        # gt_flow_D_real = gt_flow_real

        gt_flow_D_real = gt_flow_real[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
        classify_real = discriminator(gt_flow_D_real)  # [batch_size x 1 x 1 x 1]
        real = Variable(torch.ones_like(classify_real).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

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
            visualize_flow_heatmap_batched(np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path, max_flow_mag=7.0)

            merge_output_path = f"{outputpath}/{idx}_merge.png"
            visualize_merge_heatmap_batched(modified_nps, np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path, max_flow_mag=7.0)

            flow_output_path = f"{outputpath}/{idx}_flow_gt.png"
            visualize_flow_heatmap_batched(np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path)

            merge_output_path = f"{outputpath}/{idx}_merge_gt.png"
            visualize_merge_heatmap_batched(modified_nps, np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path)

            wrap_output_path = f"{outputpath}/{idx}_wrap.png"
            visualize_warp_batched(modified_nps, np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), wrap_output_path)


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

    gen_loss = BCELossFunction()
    dis_loss = BCELossFunction()

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

            # 1.4 Compute the gradients and run SGD on generator's parameters
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
                visualize_flow_heatmap_batched(np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path, max_flow_mag=7.0)

                merge_output_path = f"{outputpath}/{epoch}_{idx}_merge.png"
                visualize_merge_heatmap_batched(modified_nps, np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path, max_flow_mag=7.0)

                flow_output_path = f"{outputpath}/{epoch}_{idx}_flow_gt.png"
                visualize_flow_heatmap_batched(np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), flow_output_path)

                merge_output_path = f"{outputpath}/{epoch}_{idx}_merge_gt.png"
                visualize_merge_heatmap_batched(modified_nps, np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1)), merge_output_path)

                wrap_output_path = f"{outputpath}/{epoch}_{idx}_wrap.png"
                visualize_warp_batched(modified_nps, np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1)), wrap_output_path)


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
    # predicted_flow = predict_flow(args.modify, args.no_crop, model_path=args.model).detach().cpu().numpy()
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
    # visualize_warp(modified_np, predicted_flow, os.path.join(args.output_dir, 'drn_wrapped.jpg'))

    # # PWC testing
    # flow = estimate(args.modify, args.origin, args.no_crop, box, w, h).detach()
    # flow = flow.cpu().numpy()
    # flow = np.transpose(flow, (1, 2, 0))
    # fh, fw, fd = flow.shape

    # assert(h == fh)
    # assert(w == fw)
    # print(f"flow: h = {fh}, w = {fw}, d = {d}, fd = {fd}")

    # visualize_flow_heatmap(flow, os.path.join(args.output_dir, 'pwc_flow_heatmap.jpg'), 7.0)
    # visualize_merge_heatmap(modified_np, flow, os.path.join(args.output_dir, 'pwc_merge_heatmap.jpg'), 7.0)
    # visualize_warp(modified_np, flow, os.path.join(args.output_dir, 'pwc_wrapped.jpg'))

    # o_img = Image.open(args.origin).convert('RGB')
    # if not args.no_crop:
    #     o_img, _ = crop_img(o_img, box)
    # o_img = resize_img(o_img, w, h)[0]
    # o_img.save(os.path.join(args.output_dir, 'reshaped_original.jpg'), quality=90)
    # modified.save(os.path.join(args.output_dir, 'reshaped_modified.jpg'), quality=90)
