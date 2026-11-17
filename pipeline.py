from contextlib import nullcontext

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def predict_noise(pipe, latents, timestep, embeddings, scheduler, guidance_scale):
    """Use the same U-Net and CFG operations in the main and repair branches."""
    inputs = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
    noise = pipe.unet(inputs, timestep, encoder_hidden_states=embeddings,
                      timestep_cond=None, cross_attention_kwargs=None,
                      added_cond_kwargs=None, return_dict=False)[0]
    unconditional, conditional = noise.chunk(2)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.no_grad()
def latent_img2img(pipe, latents, schedule, i, j, embeddings, scheduler, generator, guidance_scale):
    """Jump z_t1 to t2, then denoise back to t1 with the repair conditions."""
    if not 0 <= j < i < len(schedule):
        raise ValueError("Require a rewind timestep strictly above the detection timestep")
    alpha = scheduler.alphas_cumprod.to(device=latents.device, dtype=torch.float32)
    ratio = alpha[int(schedule[j])] / alpha[int(schedule[i])]
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    current = (ratio.sqrt() * latents.float() + (1 - ratio).sqrt() * noise.float()).to(latents.dtype)
    for k in range(j, i):
        prediction = predict_noise(pipe, current, schedule[k], embeddings, scheduler, guidance_scale)
        current = scheduler.step(prediction, schedule[k], current, eta=0, return_dict=False)[0]
    return current


class TackPipeline(StableDiffusionPipeline):
    """Keep the original SD conditions outside each triggered repair window."""

    @torch.no_grad()
    def __call__(self, prompt, *, detector=None, generator=None, negative_prompt=None,
                 num_inference_steps=50, height=512, width=512, guidance_scale=7.5,
                 checkpoints=((601, 901), (501, 861), (401, 781)), rewind_seed=43,
                 repair_prompt="", repair_negative_prompt=None, repair_guidance_scale=7.5,
                 output_type="pil"):
        if detector is None:
            self.last_run = None
            return super().__call__(prompt, generator=generator, negative_prompt=negative_prompt,
                                    num_inference_steps=num_inference_steps, height=height, width=width,
                                    guidance_scale=guidance_scale, output_type=output_type)
        if type(self.scheduler) is not DDIMScheduler or self.scheduler.config.timestep_spacing != "leading":
            raise ValueError("Use DDIM with leading timestep spacing")
        if self.scheduler.config.prediction_type != "epsilon" or self.unet.config.in_channels != 4 or self.unet.config.time_cond_proj_dim is not None:
            raise ValueError("This release implements the SD1.4 epsilon-prediction path")
        if repair_negative_prompt is None:
            repair_negative_prompt = detector.concept.replace("_", " ")
        self._guidance_scale, self._guidance_rescale = guidance_scale, 0.0
        self._clip_skip, self._cross_attention_kwargs, self._interrupt = None, None, False
        device = self._execution_device
        positive, negative = self.encode_prompt(prompt, device, 1, True, negative_prompt=negative_prompt)
        embeddings = torch.cat([negative, positive])
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        schedule = self.scheduler.timesteps
        self._num_timesteps = len(schedule)
        times, plan = schedule.tolist(), {}
        for t1, t2 in checkpoints:
            plan[t1] = times.index(t2)
        stride = self.scheduler.config.num_train_timesteps // num_inference_steps
        latents = self.prepare_latents(1, 4, height, width, embeddings.dtype, device, generator, None)
        repair_scheduler = DDIMScheduler.from_config(self.scheduler.config)
        repair_scheduler.set_timesteps(num_inference_steps, device=device)
        repair_generator = torch.Generator(device=device).manual_seed(rewind_seed)
        repair_positive, repair_negative = self.encode_prompt(
            repair_prompt, device, 1, True, negative_prompt=repair_negative_prompt
        )
        repair_embeddings = torch.cat([repair_negative, repair_positive])
        records, main_calls, repair_calls = [], 0, 0
        with self.progress_bar(total=num_inference_steps) as progress:
            for i, t in enumerate(schedule):
                t1 = int(t)
                with detector.capture(self.unet) if t1 in plan else nullcontext() as features:
                    prediction = predict_noise(self, latents, t, embeddings, self.scheduler, guidance_scale)
                main_calls += 1
                if t1 in plan:
                    score, threshold = detector.score(features), detector.thresholds[t1]
                    triggered, j = score >= threshold, plan[t1]
                    if triggered:
                        latents = latent_img2img(self, latents, schedule, i, j, repair_embeddings,
                                                repair_scheduler, repair_generator, repair_guidance_scale)
                        repair_calls += i - j
                        # Resume the original prompt at the same diffusion time.
                        prediction = predict_noise(self, latents, t, embeddings, self.scheduler, guidance_scale)
                        main_calls += 1
                    records.append(dict(t1=t1, t2=int(schedule[j]), raw_score=score,
                                        threshold=threshold, triggered=triggered))
                latents = self.scheduler.step(prediction, t, latents, eta=0,
                                               generator=generator, return_dict=False)[0]
                progress.update()
        self.last_run = dict(steps=records, triggered=any(r["triggered"] for r in records),
                             main_unet_calls=main_calls, repair_unet_calls=repair_calls)
        self.last_latents = latents.detach().clone()
        if output_type == "latent":
            image, flags = latents, None
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor,
                                    return_dict=False, generator=generator)[0]
            image, flags = self.run_safety_checker(image, device, embeddings.dtype)
        image = self.image_processor.postprocess(
            image, output_type=output_type,
            do_denormalize=[True] * len(image) if flags is None else [not flag for flag in flags]
        )
        self.maybe_free_model_hooks()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=flags)
