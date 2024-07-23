from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
import einops

def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class VAEEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.sample_noise = False

    def __call__(self, x):
        mu, logvar, z = self.vae.encode(x)
        if self.sample_noise:
            return z
        else:
            return mu

    def encode(self, x):
        return self.__call__(x)


class VAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def __call__(self, z):
        return self.vae.decode(z)

    def decode(self, z):
        return self.__call__(z)



class DiffusionUnetLowdimPolicyEncoder(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            name = 'DiffusionUnetLowdimPolicyEncoder_v0',
            class_name = 'DiffusionUnetLowdimPolicyEncoder',
            cond_mode = 'none', # 'none', 'local', 'global
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            vision_model = None,
            z_diff = 1e-5 , 
            image_diff = 1.,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.name= name
        self.class_name = class_name
        self.cond_mode = cond_mode
        self.vision_model = vision_model
        self.encoder = VAEEncoder(vision_model)
        self.decoder = VAEDecoder(vision_model)
        self.z_diff = z_diff
        self.image_diff = image_diff

        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
        self.max_vals = None 
        self.min_vals = None 

        self.sqrt_recip_alphas_cumprod =  torch.sqrt(1. / self.noise_scheduler.alphas_cumprod).cuda()
        self.sqrt_recipm1_alphas_cumprod =  torch.sqrt(1. / self.noise_scheduler.alphas_cumprod - 1).cuda()


        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        # print("trajectory shape")
        # print(trajectory.shape)
        #lets assume traj is (batch )

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # assert 'obs' in obs_dict
        # assert 'past_action' not in obs_dict # not implemented yet
        # nobs = self.normalizer['obs'].normalize(obs_dict['obs'])


        _nobs = obs_dict['obs']
        nobs = self.encoder.encode(_nobs)
        n_seq = self.n_action_steps

        if self.obs_as_global_cond:
            nobs = einops.rearrange(nobs, "b z -> b 1 z")
        else:
            nobs = einops.repeat(nobs, "b z -> b t z", t=n_seq )

      

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # encode the observation

   

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        # action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action_pred = naction_pred

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            # obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            obs_pred = nobs_pred
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred

        # lets decode the action
        action = result['action_pred']

        # action has shape b t c
        action = einops.rearrange(action, "b t ... -> (b t) ...")
        action = self.decoder.decode(action)
        action = einops.rearrange(action, "(b t) ... -> b t ...", b=B)
        # action = einops.rearrange(action, "b c t -> b (c t)")
        result['action_pred'] = action # is this necessary?

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())



    def predict_start_from_noise( self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def loss(self,x,y):
        batch = {"obs" : y , "action": x}
        return self.compute_loss(batch)
        
        

    def compute_loss(self, batch):
        # normalize input
        # assert 'valid_mask' not in batch
        # nbatch = self.normalizer.normalize(batch)
        # obs = nbatch['obs']
        # action = nbatch['action']
        # I have to encode the observation 
        _obs = batch['obs']
        obs = self.encoder.encode(_obs)
        seq_length = batch['action'].shape[1]


        if not self.obs_as_global_cond:
            obs = einops.repeat(obs,'b z -> b t z',t=seq_length)
        else: 
            obs = einops.repeat(obs,'b z -> b t z',t=1)

        # obs = self.encoder.encode(_obs)

        _action = batch['action']
        action = einops.rearrange(_action, "b t ... -> (b t) ...")
        action = self.encoder.encode(action)
        action = einops.rearrange(action, "(b t) ... -> b t ...", b=_action.shape[0])

        

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual

        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

      
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss =  loss.mean()


        # compute the loss in image space.





       

        if pred_type == 'epsilon':
            # traj_start = 
            #raise ValueError("Not implemented")
            # modify here!!    
            # TODO: This expects the t to be the same for all the elements in the batch :(
            # TODO: Fix this!

            # note: by using this we see that it is equivalent!!
            # timesteps[:] = 20
            # self.noise_scheduler.config.thresholding = False
            # self.noise_scheduler.config.clip_sample = False
            # traj_start2 = self.noise_scheduler.step(
            #     pred, timesteps[0] , noisy_trajectory).pred_original_sample
            
          

            traj_start = self.predict_start_from_noise(noisy_trajectory, timesteps, pred)
          
            



            assert traj_start is not None
        elif pred_type ==    'sample':
            traj_start = pred

        # lets decode traj start
        traj_start = einops.rearrange(traj_start, "b t c -> (b t) c")
        if not self.obs_as_global_cond and not self.obs_as_local_cond:
            img_start = self.decoder.decode(traj_start[:,:self.action_dim])
        else:
            img_start = self.decoder.decode(traj_start)
        traj_img_fake = einops.rearrange(img_start, "(b t) c h w -> b t c h w", b=bsz)
        img_loss = F.mse_loss(traj_img_fake, batch['action'], reduction='mean')

        # loss += self.image_diff * img_loss

        return { 'z_loss' : loss , 'img_loss': img_loss}

