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
from typing import Optional
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
from prediction.losses import BCELossFunction, LSLossFunction, MixedGenLossFunction, LossConfig
from prediction.metrics import EvaluationMetric
from dataset import GANDataset, gan_collate


def overfit(
    basepath: str,
    outputpath: str,
    no_crop: bool = False,
    num_iterations: int = 3000,
    log_frequency: int = 100,
    learn_rate: float = 1e-4,
    # dataset_path: Optional[str] = None,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(outputpath, exist_ok=True)

    gen_config = GeneratorConfig()
    dis_config = DiscriminatorConfig()

    generator = Generator(gen_config).to(device)
    discriminator = ConvGANDiscriminator(dis_config).to(device)

    # if not dataset_path:
    gan_dataset = GANDataset(basepath, no_crop)
    #     torch.save(gan_dataset, f"{outputpath}/dataset_path.pth")
    # else:
    #     gan_dataset = torch.load(dataset_path, map_location="cpu")

    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        collate_fn=gan_collate,
    )

    gen_loss_config = LossConfig()
    dis_loss_config = LossConfig()

    gen_loss = MixedGenLossFunction(gen_loss_config)
    dis_loss = BCELossFunction(dis_loss_config)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    G_losses, epe_losses, warp_losses, gan_losses, D_losses, D_fake_losses, D_real_losses = [], [], [], [], [], [], []

    modified, original, modified_data, original_data = next(iter(dataloader))
    for idx in tqdm(range(num_iterations)):
        original = Variable(original.to(device))

        # (2) Update D network
        discriminator.zero_grad()
        generator.eval()
        discriminator.train()

        # 2.1 Get noise from modified image
        noise = Variable(modified.to(device))

        # 2.2 Generate fake flow from the noise
        predicted_flow = generator(noise)  # [batch_size x D x H x W]

        predicted_flow_fake = original - warp(noise, predicted_flow)
        predicted_flow_D_fake = predicted_flow_fake

        # _, _, H, W = predicted_flow_fake.shape
        # sampled_point = np.random.choice(np.arange(min(H, W) - 64), 1).item()
        # predicted_flow_D_fake = predicted_flow_fake[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
        classify_fake = discriminator(predicted_flow_D_fake.detach())  # [batch_size x 1 x 1 x 1]
        fake = Variable(torch.zeros_like(classify_fake).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

        # 2.3 Compute the discriminator loss on the fake flow
        D_fake_loss, D_fake_loss_data = dis_loss(classify_fake, fake)
        D_fake_losses.append(D_fake_loss.item())

        # 2.4 Form the GT flow by PWC-Net
        with torch.no_grad():
            gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)

        gt_flow = Variable(gt_flow.to(device))  # [batch_size x D x H x W]

        gt_flow_real = original - warp(noise, gt_flow)
        gt_flow_D_real = gt_flow_real

        # gt_flow_D_real = gt_flow_real[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
        classify_real = discriminator(gt_flow_D_real)  # [batch_size x 1 x 1 x 1]
        real = Variable(torch.ones_like(classify_real).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

        # 2.5 Compute the discriminator loss on the GT flow
        D_real_loss, D_real_loss_data = dis_loss(classify_real, real)
        D_real_losses.append(D_real_loss.item())

        # 2.6 Compute the total discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_losses.append(D_total_loss.item())

        # 2.7 Compute the gradients and run SGD on discriminator's parameters
        D_total_loss.backward()
        dis_optimizer.step()

        # (1) Update G network
        generator.zero_grad()
        discriminator.eval()
        generator.train()

        # 1.1 Get noise from modified image
        noise = Variable(modified.to(device))

        # 1.2 Generate fake flow from the noise
        predicted_flow = generator(noise)  # [batch_size x D x H x W]

        predicted_flow_warpped = warp(noise, predicted_flow)
        predicted_flow_fool = original - predicted_flow_warpped
        predicted_flow_G = predicted_flow_fool

        # _, _, H, W = predicted_flow_fool.shape
        # sampled_point = np.random.choice(np.arange(min(H, W) - 64), 1).item()
        # predicted_flow_G = predicted_flow_fool[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]

        classify_fool = discriminator(predicted_flow_G)  # [batch_size x 1 x 1 x 1]
        fool = Variable(torch.ones_like(classify_fool).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

        # 1.3 Compute the generator loss on the fake flow
        G_loss, G_loss_data = gen_loss(predicted_flow, gt_flow, predicted_flow_warpped, original, classify_fool, fool)
        G_losses.append(G_loss.item())
        epe_losses.append(G_loss_data.loss_epe)
        warp_losses.append(G_loss_data.loss_warp)
        gan_losses.append(G_loss_data.loss_gan)

        # 1.4 Compute the gradients and run SGD on generator's parameters
        G_loss.backward()
        gen_optimizer.step()

        if (idx + 1) % log_frequency == 0:
            print(
                f"Epoch [{idx}/{num_iterations}]: " +
                f"D Total Loss - {D_total_loss.item():.4f} " +
                f"D Real Loss - {D_real_loss.item():.4f} " +
                f"D Fake Loss - {D_fake_loss.item():.4f} " +
                f"G Total Loss - {G_loss.item():.4f} " +
                f"G EPE Loss - {G_loss_data.loss_epe:.4f} " +
                f"G Warp Loss - {G_loss_data.loss_warp:.4f} " +
                f"G Fool Loss - {G_loss_data.loss_gan:.4f} "
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

    losses = np.asarray([np.arange(0, num_iterations), G_losses, epe_losses, warp_losses, gan_losses, D_losses, D_fake_losses, D_real_losses])
    losses.tofile(f"{outputpath}/losses.csv", sep=',')


def train(
    basepath: str,
    outputpath: str,
    no_crop: bool = False,
    batch_size: int = 2,
    num_workers: int = 8,
    num_epochs: int = 8,
    log_frequency: int = 50,
    learn_rate: float = 1e-4,
    # dataloader: Optional[str] = None,
    gen_checkpoint: Optional[str] = None,
    dis_checkpoint: Optional[str] = None,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    os.makedirs(outputpath, exist_ok=True)

    gen_config = GeneratorConfig()
    dis_config = DiscriminatorConfig()

    generator = Generator(gen_config).to(device)
    if gen_checkpoint:
        generator.load_state_dict(torch.load(gen_checkpoint, map_location="cpu"))

    discriminator = ConvGANDiscriminator(dis_config).to(device)
    if dis_checkpoint:
        discriminator.load_state_dict(torch.load(dis_checkpoint, map_location="cpu"))

    # if not dataloader:
    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=gan_collate,
    )
    #     torch.save(dataloader, f"{outputpath}/dataloader.pth")
    # else:
    #     dataloader = torch.load(dataloader, map_location="cpu")

    gen_loss_config = LossConfig()
    dis_loss_config = LossConfig()

    gen_loss = MixedGenLossFunction(gen_loss_config)
    dis_loss = BCELossFunction(dis_loss_config)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

    for epoch in range(num_epochs):
        G_losses, epe_losses, warp_losses, gan_losses, D_losses, D_fake_losses, D_real_losses = [], [], [], [], [], [], []
        for idx, (modified, original, modified_data, original_data) in tqdm(enumerate(dataloader)):
            original = Variable(original.to(device))
            # print(f"original: {original.shape}")

            # (2) Update D network
            discriminator.zero_grad()
            generator.eval()
            discriminator.train()

            # 2.1 Get noise from modified image
            noise = Variable(modified.to(device))

            # 2.2 Generate fake flow from the noise
            predicted_flow = generator(noise)  # [batch_size x D x H x W]

            predicted_flow_fake = original - warp(noise, predicted_flow)
            predicted_flow_D_fake = predicted_flow_fake

            # _, _, H, W = predicted_flow_fake.shape
            # sampled_point = np.random.choice(np.arange(min(H, W) - 64), 1).item()
            # predicted_flow_D_fake = predicted_flow_fake[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
            classify_fake = discriminator(predicted_flow_D_fake.detach())  # [batch_size x 1 x 1 x 1]
            fake = Variable(torch.zeros_like(classify_fake).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

            # 2.3 Compute the discriminator loss on the fake flow
            D_fake_loss, D_fake_loss_data = dis_loss(classify_fake, fake)
            D_fake_losses.append(D_fake_loss.item())

            # 2.4 Form the GT flow by PWC-Net
            with torch.no_grad():
                gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)

            gt_flow = Variable(gt_flow.to(device))  # [batch_size x D x H x W]

            gt_flow_real = original - warp(noise, gt_flow)
            gt_flow_D_real = gt_flow_real

            # gt_flow_D_real = gt_flow_real[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]
            classify_real = discriminator(gt_flow_D_real)  # [batch_size x 1 x 1 x 1]
            real = Variable(torch.ones_like(classify_real).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

            # 2.5 Compute the discriminator loss on the GT flow
            D_real_loss, D_real_loss_data = dis_loss(classify_real, real)
            D_real_losses.append(D_real_loss.item())

            # 2.6 Compute the total discriminator loss
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_losses.append(D_total_loss.item())

            # 2.7 Compute the gradients and run SGD on discriminator's parameters
            D_total_loss.backward()
            dis_optimizer.step()

            # (1) Update G network
            generator.zero_grad()
            discriminator.eval()
            generator.train()

            # 1.1 Get noise from modified image
            noise = Variable(modified.to(device))

            # 1.2 Generate fake flow from the noise
            predicted_flow = generator(noise)  # [batch_size x D x H x W]

            predicted_flow_warpped = warp(noise, predicted_flow)
            predicted_flow_fool = original - predicted_flow_warpped
            predicted_flow_G = predicted_flow_fool

            # _, _, H, W = predicted_flow_fool.shape
            # sampled_point = np.random.choice(np.arange(min(H, W) - 64), 1).item()
            # predicted_flow_G = predicted_flow_fool[:, :, sampled_point:sampled_point+64, sampled_point:sampled_point+64]  # [batch_size x D x 64 x 64]

            classify_fool = discriminator(predicted_flow_G)  # [batch_size x 1 x 1 x 1]
            fool = Variable(torch.ones_like(classify_fool).float().to(device), requires_grad=False)  # [batch_size x 1 x 1 x 1]

            # 1.3 Compute the generator loss on the fake flow
            G_loss, G_loss_data = gen_loss(predicted_flow, gt_flow, predicted_flow_warpped, original, classify_fool, fool)
            G_losses.append(G_loss.item())
            epe_losses.append(G_loss_data.loss_epe)
            warp_losses.append(G_loss_data.loss_warp)
            gan_losses.append(G_loss_data.loss_gan)

            # 1.4 Compute the gradients and run SGD on generator's parameters
            G_loss.backward()
            gen_optimizer.step()

            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(dataloader)}]: " +
                    f"D Total Loss - {D_total_loss.item():.4f} " +
                    f"D Real Loss - {D_real_loss.item():.4f} " +
                    f"D Fake Loss - {D_fake_loss.item():.4f} " +
                    f"G Total Loss - {G_loss.item():.4f} " +
                    f"G EPE Loss - {G_loss_data.loss_epe:.4f} " +
                    f"G Warp Loss - {G_loss_data.loss_warp:.4f} " +
                    f"G Fool Loss - {G_loss_data.loss_gan:.4f} "
                )

                predicted_flow = np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1))
                gt_flow = np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1))

                for batch_idx in range(batch_size):
                    predicted_flow_batch = np.expand_dims(predicted_flow[batch_idx], axis=0)
                    gt_flow_batch = np.expand_dims(gt_flow[batch_idx], axis=0)
                    modified_nps_batch = np.expand_dims(modified_nps[batch_idx], axis=0)

                    flow_output_path = f"{outputpath}/{epoch}_{idx}_{batch_idx}_flow.png"
                    visualize_flow_heatmap_batched(predicted_flow_batch, flow_output_path, max_flow_mag=7.0)

                    merge_output_path = f"{outputpath}/{epoch}_{idx}_{batch_idx}_merge.png"
                    visualize_merge_heatmap_batched(modified_nps_batch, predicted_flow_batch, merge_output_path, max_flow_mag=7.0)

                    flow_output_path = f"{outputpath}/{epoch}_{idx}_{batch_idx}_flow_gt.png"
                    visualize_flow_heatmap_batched(gt_flow_batch, flow_output_path)

                    merge_output_path = f"{outputpath}/{epoch}_{idx}_{batch_idx}_merge_gt.png"
                    visualize_merge_heatmap_batched(modified_nps_batch, gt_flow_batch, merge_output_path)

                    wrap_output_path = f"{outputpath}/{epoch}_{idx}_{batch_idx}_wrap.png"
                    visualize_warp_batched(modified_nps_batch, predicted_flow_batch, wrap_output_path)

        torch.save(generator.state_dict(), f"{outputpath}/generator_{epoch:03d}.pth")
        torch.save(discriminator.state_dict(), f"{outputpath}/discriminator_{epoch:03d}.pth")

        losses = np.asarray([np.arange(0, len(dataloader)), G_losses, epe_losses, warp_losses, gan_losses, D_losses, D_fake_losses, D_real_losses])
        losses.tofile(f"{outputpath}/losses_{epoch:03d}.csv", sep=',')


def evaluate(
    basepath: str,
    outputpath: str,
    no_crop: bool = False,
    num_workers: int = 8,
    checkpoint: Optional[str] = None
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    os.makedirs(outputpath, exist_ok=True)

    gen_config = GeneratorConfig()

    generator = Generator(gen_config).to(device)
    if checkpoint:
        generator.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        num_workers=num_workers,
        collate_fn=gan_collate,
    )

    metric = EvaluationMetric()

    for idx, (modified, original, modified_data, original_data) in tqdm(enumerate(dataloader)):
        generator.eval()

        with torch.no_grad():
            original = original.to(device)

            # Get noise from modified image
            noise = modified.to(device)

            # Generate fake flow from the noise
            predicted_flow = generator(noise)  # [batch_size x D x H x W]

            # Form the GT flow by PWC-Net
            gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)
            gt_flow = gt_flow.to(device) # [batch_size x D x H x W]

            metric(predicted_flow[0], gt_flow[0], noise[0], original[0])

            predicted_flow = np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1))
            gt_flow = np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1))

            flow_output_path = f"{outputpath}/{idx}_flow.png"
            visualize_flow_heatmap_batched(predicted_flow, flow_output_path, max_flow_mag=7.0)

            merge_output_path = f"{outputpath}/{idx}_merge.png"
            visualize_merge_heatmap_batched(modified_nps, predicted_flow, merge_output_path, max_flow_mag=7.0)

            flow_output_path = f"{outputpath}/{idx}_flow_gt.png"
            visualize_flow_heatmap_batched(gt_flow, flow_output_path)

            merge_output_path = f"{outputpath}/{idx}_merge_gt.png"
            visualize_merge_heatmap_batched(modified_nps, gt_flow, merge_output_path)

            wrap_output_path = f"{outputpath}/{idx}_wrap.png"
            visualize_warp_batched(modified_nps, predicted_flow, wrap_output_path)

    avg_epe, delta_psnr, avg_iou = metric.inference()
    print(
        f"There are {len(dataloader)} images in validation set: " +
        f"EPE: {avg_epe:.4f} " +
        f"PSNR increases: {delta_psnr:.4f} " +
        f"IOU: {avg_iou:.4f} "
    )

    np.array(metric.evaldata.epes).tofile(f"{outputpath}/result_epe.csv", sep=',')
    np.array(metric.evaldata.ious).tofile(f"{outputpath}/result_iou.csv", sep=',')
    np.array(metric.evaldata.before_psnrs).tofile(f"{outputpath}/result_psnr_before.csv", sep=',')
    np.array(metric.evaldata.after_psnrs).tofile(f"{outputpath}/result_psnr_after.csv", sep=',')


def pretrain(
    basepath: str,
    outputpath: str,
    no_crop: bool = False,
    num_workers: int = 8,
    checkpoint: str = "./drn/weights/local.pth"
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    os.makedirs(outputpath, exist_ok=True)

    gen_config = GeneratorConfig()
    gen_config.model_path = checkpoint

    generator = Generator(gen_config).to(device)

    gan_dataset = GANDataset(basepath, no_crop)
    dataloader = torch.utils.data.DataLoader(
        gan_dataset,
        num_workers=num_workers,
        collate_fn=gan_collate,
    )

    metric = EvaluationMetric()

    for idx, (modified, original, modified_data, original_data) in tqdm(enumerate(dataloader)):
        generator.eval()

        with torch.no_grad():
            original = original.to(device)

            # Get noise from modified image
            noise = modified.to(device)

            # Generate fake flow from the noise
            predicted_flow = generator(noise)  # [batch_size x D x H x W]

            # Form the GT flow by PWC-Net
            gt_flow, modified_nps = generator.inference(predicted_flow, modified_data, original_data, no_crop)
            gt_flow = gt_flow.to(device) # [batch_size x D x H x W]

            metric(predicted_flow[0], gt_flow[0], noise[0], original[0])

            predicted_flow = np.transpose(predicted_flow.detach().cpu().numpy(), (0, 2, 3, 1))
            gt_flow = np.transpose(gt_flow.detach().cpu().numpy(), (0, 2, 3, 1))

            flow_output_path = f"{outputpath}/{idx}_flow.png"
            visualize_flow_heatmap_batched(predicted_flow, flow_output_path, max_flow_mag=7.0)

            merge_output_path = f"{outputpath}/{idx}_merge.png"
            visualize_merge_heatmap_batched(modified_nps, predicted_flow, merge_output_path, max_flow_mag=7.0)

            flow_output_path = f"{outputpath}/{idx}_flow_gt.png"
            visualize_flow_heatmap_batched(gt_flow, flow_output_path)

            merge_output_path = f"{outputpath}/{idx}_merge_gt.png"
            visualize_merge_heatmap_batched(modified_nps, gt_flow, merge_output_path)

            wrap_output_path = f"{outputpath}/{idx}_wrap.png"
            visualize_warp_batched(modified_nps, predicted_flow, wrap_output_path)

    avg_epe, delta_psnr, avg_iou = metric.inference()
    print(
        f"There are {len(dataloader)} images in validation set: " +
        f"EPE: {avg_epe:.4f} " +
        f"PSNR increases: {delta_psnr:.4f} " +
        f"IOU: {avg_iou:.4f} "
    )

    np.array(metric.evaldata.epes).tofile(f"{outputpath}/result_epe.csv", sep=',')
    np.array(metric.evaldata.ious).tofile(f"{outputpath}/result_iou.csv", sep=',')
    np.array(metric.evaldata.before_psnrs).tofile(f"{outputpath}/result_psnr_before.csv", sep=',')
    np.array(metric.evaldata.after_psnrs).tofile(f"{outputpath}/result_psnr_after.csv", sep=',')


if __name__ == "__main__":
    import fire

    fire.Fire()
