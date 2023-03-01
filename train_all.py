import argparse
import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from utils import Lion, to_4d, to_diff4d, linear_warmup, cosine_anneal
from torch.utils.data import DataLoader
import wandb
from diffusion.diffusions.gaussian import GaussianDiffusion
from diffusion.models.unet import TemporalUnet
import torchvision.utils as vutils

from slate.shapes_2d import CswmStyleDataset as Shapes2Dtraj # look
from slate.slate import SLATE

def to_5d(batch):
    return torch.unflatten(batch, 0, (-1, args.traj_size))

if __name__ == '__main__':
    print('Running...')
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('--data_path', type=str,
                        default='C:/Users/lizav/PycharmProjects/sdm/c-swm/data/shapes_train.h5')
    parser.add_argument('--traj_size', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=1)  # care
    # slate args
    parser.add_argument('--attn_init_mode', type=str, default='classic')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=128)  # interesting
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_iterations', type=int, default=3)  # interesting
    parser.add_argument('--num_slots', type=int, default=6)  # interesting
    parser.add_argument('--num_slot_heads', type=int, default=1)  # interesting
    parser.add_argument('--slot_size', type=int, default=32)  # interesting
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)
    parser.add_argument('--hard', action='store_true')
    # q-vae params
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_final', type=float, default=0.1)
    parser.add_argument('--tau_steps', type=int, default=50_000_000)
    # diffusion params
    parser.add_argument('--n_timesteps', type=int, default=100)
    parser.add_argument('--action_emb', type=int, default=8)
    parser.add_argument('--diff_size', type=int, default=32)
    # optimizer params
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_dvae', type=float, default=3e-4)
    parser.add_argument('--lr_main', type=float, default=1e-4)
    parser.add_argument('--lr_diffusion', type=float, default=1e-3)
    parser.add_argument('--lr_action_emb', type=float, default=3e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=100_000_000)
    parser.add_argument('--lr_diffusion_warmup_steps', type=int, default=200_000_000)
    parser.add_argument('--max_steps', type=int, default=300_000_000)
    parser.add_argument('--start_decay', type=int, default=300_000_000)
    # visualisation params
    parser.add_argument('--vis_num', type=int, default=32)
    # extra
    parser.add_argument('--no_diff_loss', type=bool, default=False)

    args = parser.parse_args()

    # First, lets load batch of traj
    print('Started loading data')
    train_dataset = Shapes2Dtraj(root=args.data_path, phase='train', traj_size=args.traj_size)
    print('Train data loaded')
    val_dataset = Shapes2Dtraj(root=args.data_path, phase='val', traj_size=args.traj_size)
    print('Loaded data')
    print(len(train_dataset))
    # train_dataset is np array (#taj, #obs(96), 3, 64, 64)

    # Creating loaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
    }

    train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
    train_epoch_size = len(train_loader)

    # start params
    start_epoch = 0
    best_epoch = 0
    best_loss = math.inf
    lr_decay_factor = 1.0

    # creating slate
    slate = SLATE(args)
    slate = slate.cuda()

    # creating diffusion
    action_embedding = torch.nn.Embedding(20, args.action_emb, max_norm=1, device='cuda')
    model = TemporalUnet(args.traj_size, args.slot_size + args.action_emb, dim=args.diff_size)
    diffusion = GaussianDiffusion(model, n_timesteps=args.n_timesteps, device='cuda')

    # creating optimizer
    if args.optimizer == 'adam':
        optimizer_class = Adam
    elif args.optimizer == 'lion':
        optimizer_class = Lion
    else:
        raise NotImplementedError
    optimizer = optimizer_class([
        {'params': (x[1] for x in slate.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
        {'params': (x[1] for x in slate.named_parameters() if 'dvae' not in x[0]), 'lr': args.lr_main},
        {'params': diffusion.model.parameters(), 'lr': args.lr_diffusion},
        {'params': action_embedding.parameters(), 'lr': args.lr_action_emb}
    ])

    # starting wandb
    wandb.init(project='joint_training',
               # mode='offline'
               )
    wandb.config.update(args)
    print(args)

    epoch = -1
    global_slate_step = 0
    slate.train()
    model.train()
    while True:
        if global_slate_step > args.max_steps:
            break
        epoch += 1
        for traj_batch, (trajs, actions) in enumerate(train_loader):
            wandb_logs = {}
            global_slate_step = (epoch * train_epoch_size + traj_batch) * args.traj_size * args.batch_size
            # calculate training constants
            wandb_logs['STEPS/global_slate_step'] = global_slate_step
            wandb_logs['STEPS/epoch'] = epoch
            tau = cosine_anneal(
                global_slate_step,
                args.tau_start,
                args.tau_final,
                0,
                args.tau_steps)

            lr_warmup_factor = linear_warmup(
                global_slate_step,
                0.,
                1.0,
                0,
                args.lr_warmup_steps)

            lr_diffusion_warmup_factor = linear_warmup(
                global_slate_step,
                0.,
                1.0,
                0,
                args.lr_diffusion_warmup_steps)

            lr_global_decay = linear_warmup(
                global_slate_step,
                1.0,
                0,
                args.start_decay,
                args.max_steps)

            optimizer.param_groups[0]['lr'] = lr_decay_factor * args.lr_dvae
            optimizer.param_groups[3]['lr'] = lr_decay_factor * args.lr_action_emb * lr_global_decay
            optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor * args.lr_main * lr_global_decay
            optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_diffusion_warmup_factor * args.lr_diffusion * lr_global_decay

            wandb_logs['TRAIN/lr/tau'] = tau
            wandb_logs['TRAIN/lr/dvae'] = optimizer.param_groups[0]['lr']
            wandb_logs['TRAIN/lr/main'] = optimizer.param_groups[1]['lr']
            wandb_logs['TRAIN/lr/diffusion'] = optimizer.param_groups[2]['lr']
            wandb_logs['TRAIN/lr/action_emb'] = optimizer.param_groups[3]['lr']

            trajs = trajs.to(torch.float)
            trajs = trajs.cuda()

            optimizer.zero_grad()

            pic_batch = to_4d(trajs)

            # pass to slate
            recon, cross_entropy, mse, attns, slots = slate.diffusion_forward(pic_batch, tau, args.hard)
            wandb_logs['TRAIN/losses/q-vae_mse'] = mse
            wandb_logs['TRAIN/losses/slate_cross_entropy'] = cross_entropy
            # prepare slots for diffusion
            slots = to_5d(slots)
            actions = actions.to('cuda')
            actions = action_embedding(actions)
            actions = torch.unsqueeze(actions, -2)
            actions = actions.expand(-1, -1, args.num_slots, -1)
            slots = torch.cat([slots, actions], dim=-1)
            slots = to_diff4d(slots)
            # TODO: ADD NORM ?
            # pass to diffusion
            diffusion_mse = diffusion.loss(slots)
            wandb_logs['TRAIN/losses/diffusion_loss'] = diffusion_mse
            # TODO: weight losses actions

            # complete loss calculation here
            loss = mse + cross_entropy
            wandb_logs['TRAIN/losses/slate_loss'] = mse + cross_entropy
            if not args.no_diff_loss:
                loss += diffusion_mse
            wandb_logs['TRAIN/losses/loss'] = loss
            if loss < best_loss:
                best_loss = loss
            wandb_logs['TRAIN/losses/best_loss'] = best_loss

            # making step
            loss.backward()
            # TODO: grad accumulation
            clip_grad_norm_(slate.parameters(), args.clip, 'inf')
            optimizer.step()

            # calc req for visualise
            # now for every epoch
            visualise = (traj_batch == 0)

            # start visualisation
            if visualise:
                slate.eval()
                model.eval()
                traj, actions = val_dataset[torch.randint(high=len(val_dataset), size=(1,))]
                traj = torch.tensor(traj)
                traj = traj.to(torch.float)
                traj = traj.to('cuda')
                ground_vis = traj[-args.vis_num:]
                with torch.no_grad():
                    # slate visualisation
                    recon, cross_entropy, mse, attns, slots = slate.diffusion_forward(traj, tau, args.hard)
                    recon_vis = recon[-args.vis_num:]
                    gen_img_vis = slate.reconstruct_autoregressive(ground_vis)
                    ground_vis = ground_vis
                    # diff visualisation
                    history = slots[:-args.vis_num]
                    actions = torch.tensor(actions).to('cuda')
                    actions = action_embedding(actions)
                    slots = diffusion.wm_sample(history, actions)
                    slots = slots[-args.vis_num:, :, :-args.action_emb]
                    sample = slate.reconstruct_slots(slots)
                    sample = torch.clamp(sample, 0, 1)

                    #  logging pictures
                    vis = torch.cat((ground_vis, recon_vis, gen_img_vis, sample), dim=0)
                    vis = vis.view(-1, 3, args.image_size, args.image_size)
                    attns = torch.transpose(attns[-args.vis_num:], 0, 1)
                    attns = to_4d(attns)
                    grid = vutils.make_grid(vis, nrow=args.vis_num, pad_value=0.2)[:, 2:-2, 2:-2]
                    attns_grid = vutils.make_grid(attns, nrow=args.vis_num, pad_value=0.2)[:, 2:-2, 2:-2]
                    wandb_logs['VIS/results'] = wandb.Image(grid)
                    wandb_logs['VIS/attns'] = wandb.Image(attns_grid)

                slate.train()
                model.train()

            # logging
            wandb.log(wandb_logs)
            print(f'logged {wandb_logs}')