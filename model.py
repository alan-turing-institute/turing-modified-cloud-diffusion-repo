from cloudcasting.models import AbstractModel
import turing_modified_cloud_diffusion_repo

import torch
import numpy as np
import torchvision.transforms as T

import random, logging
from types import SimpleNamespace

import torch, wandb
from fastprogress import progress_bar

from cloud_diffusion.dataset import download_dataset, CloudDataset
from cloud_diffusion.ddpm import ddim_sampler
from cloud_diffusion.models import UNet2D, get_unet_params
from cloud_diffusion.utils import parse_args, set_seed

class TuringCloudDiffusionModel(AbstractModel):
    """TuringCloudDiffusionModel model class"""

    def __init__(self, history_steps: int, model_path: int) -> None:
        # All models must include `history_steps` as a parameter. This is the number of previous
        # frames that the model uses to makes its predictions. This should not be more than 25, i.e.
        # 6 hours (inclusive of end points) of 15 minutely data.
        # The history_steps parameter should be specified in `validate_config.yml`, along with
        # any other parameters (replace `example_parameter` with as many other parameters as you need to initialize your model, and also add them to `validate_config.yml` under `model: params`)
        super().__init__(history_steps)


        ###### YOUR CODE HERE ######
        # Here you can add any other parameters that you need to initialize your model
        # You might load your trained ML model or set up an optical flow method here.
        # You can also access any code from src/turing_modified_cloud_diffusion_repo, e.g.
        x = turing_modified_cloud_diffusion_repo.load_model()

        ############################


    def forward(self, X):
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)

        ###### YOUR CODE HERE ######
        ...
        samples_list = []
        
        for b in range (X.shape[0]): 
            channel_list = []

            for c in range (X.shape[1]):
                #STEP 1: Transform Input using tfms from CLoudcastingDataset Class
                img_size = 64
                tfms = [T.Resize((img_size, int(img_size*614/372)))]
                tfms += [T.RandomCrop(img_size)]
                concat_data = np.concatenate(X[b], axis=-3)[c] #concatenate input frames for one channel
                transformed_data = 0.5-T.Compose(tfms)(torch.from_numpy(concat_data))

                #model config
                config = SimpleNamespace(
                    model_name="unet_small", # model name to save [unet_small, unet_big]
                    sampler_steps=333, # number of sampler steps on the diffusion process
                    num_frames=4, # number of frames to use as input,
                    img_size=64, # image size to use
                    num_random_experiments = 1, # we will perform inference once
                    seed=42,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    # device="mps",
                    sampler="ddim",
                    future_frames=12,  # number of future frames
                    bs=8, # how many samples
                )
    
                #STEP 2: Run Model forward (based on cloud_diffusion inference.py)
                set_seed(config.seed)

                def prepare_data(data):
                    "Generates a batch of data from the validation dataset. Batches are of size [8, 4, 64, 64] representing batch, time, height, width "
            
                    data.valid_ds = valid_ds
                    idxs = random.choices(range(len(valid_ds) - config.future_frames), k=config.bs)  # select some samples
                    #data.idxs = random.choices(range(len(data.valid_ds) - config.future_frames), k=config.bs)  # select some samples
                    # fix the batch to the same samples for reproducibility
                    #self.batch = self.valid_ds[self.idxs[0]].to(config.device)
                    data.batch = torch.stack([data.valid_ds[idxs[0]].to(config.device), data.valid_ds[idxs[1]].to(config.device), 
                                 data.valid_ds[idxs[2]].to(config.device), data.valid_ds[idxs[3]].to(config.device), 
                                 data.valid_ds[idxs[4]].to(config.device), data.valid_ds[idxs[5]].to(config.device),
                                data.valid_ds[idxs[6]].to(config.device), data.valid_ds[idxs[7]].to(config.device)])
        
                # create a batch of data to use for inference
                prepared_data = prepare_data(transformed_data)
                
                # we default to ddim as it's faster and as good as ddpm
                sampler = ddim_sampler(config.sampler_steps)
        
                # create the Unet
                model_params = get_unet_params(config.model_name, config.num_frames)
        
                pretrained_model = UNet2D.from_artifact(model_params, self.model_path).to(config.device)
        
                model_eval = pretrained_model.eval()
            
                def sample_more(self, frames, future_frames=1):
                    "Autoregressive sampling, starting from `frames`. It is hardcoded to work with 3 frame inputs."
                    for _ in progress_bar(range(future_frames), total=future_frames, leave=True):
                        # compute new frame with previous 3 frames
                        new_frame = self.sampler(self.model, frames[:,-3:,...])
                        # add new frame to the sequence
                        frames = torch.cat([frames, new_frame.to(frames.device)], dim=1)
                    return frames.cpu()
            
                def forecast(self):
                    "Perform inference on the batch of data."
                    logger.info(f"Forecasting {self.batch.shape[0]} samples for {self.config.future_frames} future frames.")
                    sequences = []
                    for i in range(self.config.num_random_experiments):
                        logger.info(f"Generating {i+1}/{self.config.num_random_experiments} futures.")
                        frames = self.sample_more(self.batch, self.config.future_frames)
                        sequences.append(frames)
            
                    return sequences

                sequences = forecast(model_eval)

                #trim to predicted frames (last 12)
                predicted_frames = [tensor[:,-12:,:,:] for tensor in sequences]
                    
                #STEP 3: Resize images back to original size and shape
                resize_transform = T.Resize((372,372)) 
                resized_frames = [resize_transform(tensor) for tensor in predicted_frames]
                #STEP 4: Concatenate results into array of size [batch, channels, height, width]
                channel_list.append(resized_frames)
            channel_stack = np.stack(channel_list)
            samples_list.append(channel_stack)
        samples_stack = np.stack(samples_list)
        return samples_stack

    def hyperparameters_dict(self):

        # This function should return a dictionary of hyperparameters for the model
        # This is just for your own reference and will be saved with the model scores to wandb

        ###### YOUR CODE HERE ######

        params_dict = {
            "model_name":"unet_small", # model name to save [unet_small, unet_big]
            "sampler_steps":333, # number of sampler steps on the diffusion process
            "num_frames":4, 
            "img_size":64,
            "num_random_experiments" : 1,
            "seed":42,
            "device": "cuda",
            "sampler": "ddim",
            "future_frames": 10,
            "bs": 8
        }
        ...
      