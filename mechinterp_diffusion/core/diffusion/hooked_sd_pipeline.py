"""
Hooked Diffusion Pipeline for Stable Diffusion XL

Wrapper class for diffusion model pipelines enabling hooks and caching of
intermediate representations for further analysis.

Source/ copied from:
https://github.com/cywinski/SAeUron/blob/main/SAE/hooked_sd_noised_pipeline.py

Changes made to original code:
 - created HookedStableDiffusionXLPipeline class to work with SDXL Pipeline
 - comments to structure code
 - automatic formatting and ruff conformance
 - added option to early stop denoising loop in case only interested in
    few late timesteps
 - use contextlib.nullcontext for torch.no_grad() to
 allow for gradients in the denoising loop in circuit discovery and disable
 them during normal generation
 - added support for extracting both conditioanl and unconditional latents.


License of original code:
                                Apache License
                        Version 2.0, January 2004
                    http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

    "License" shall mean the terms and conditions for use, reproduction,
    and distribution as defined by Sections 1 through 9 of this document.

    "Licensor" shall mean the copyright owner or entity authorized by
    the copyright owner that is granting the License.

    "Legal Entity" shall mean the union of the acting entity and all
    other entities that control, are controlled by, or are under common
    control with that entity. For the purposes of this definition,
    "control" means (i) the power, direct or indirect, to cause the
    direction or management of such entity, whether by contract or
    otherwise, or (ii) ownership of fifty percent (50%) or more of the
    outstanding shares, or (iii) beneficial ownership of such entity.

    "You" (or "Your") shall mean an individual or Legal Entity
    exercising permissions granted by this License.

    "Source" form shall mean the preferred form for making modifications,
    including but not limited to software source code, documentation
    source, and configuration files.

    "Object" form shall mean any form resulting from mechanical
    transformation or translation of a Source form, including but
    not limited to compiled object code, generated documentation,
    and conversions to other media types.

    "Work" shall mean the work of authorship, whether in Source or
    Object form, made available under the License, as indicated by a
    copyright notice that is included in or attached to the work
    (an example is provided in the Appendix below).

    "Derivative Works" shall mean any work, whether in Source or Object
    form, that is based on (or derived from) the Work and for which the
    editorial revisions, annotations, elaborations, or other modifications
    represent, as a whole, an original work of authorship. For the purposes
    of this License, Derivative Works shall not include works that remain
    separable from, or merely link (or bind by name) to the interfaces of,
    the Work and Derivative Works thereof.

    "Contribution" shall mean any work of authorship, including
    the original version of the Work and any modifications or additions
    to that Work or Derivative Works thereof, that is intentionally
    submitted to Licensor for inclusion in the Work by the copyright owner
    or by an individual or Legal Entity authorized to submit on behalf of
    the copyright owner. For the purposes of this definition, "submitted"
    means any form of electronic, verbal, or written communication sent
    to the Licensor or its representatives, including but not limited to
    communication on electronic mailing lists, source code control systems,
    and issue tracking systems that are managed by, or on behalf of, the
    Licensor for the purpose of discussing and improving the Work, but
    excluding communication that is conspicuously marked or otherwise
    designated in writing by the copyright owner as "Not a Contribution."

    "Contributor" shall mean Licensor and any individual or Legal Entity
    on behalf of whom a Contribution has been received by Licensor and
    subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
    this License, each Contributor hereby grants to You a perpetual,
    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    copyright license to reproduce, prepare Derivative Works of,
    publicly display, publicly perform, sublicense, and distribute the
    Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
    this License, each Contributor hereby grants to You a perpetual,
    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    (except as stated in this section) patent license to make, have made,
    use, offer to sell, sell, import, and otherwise transfer the Work,
    where such license applies only to those patent claims licensable
    by such Contributor that are necessarily infringed by their
    Contribution(s) alone or by combination of their Contribution(s)
    with the Work to which such Contribution(s) was submitted. If You
    institute patent litigation against any entity (including a
    cross-claim or counterclaim in a lawsuit) alleging that the Work
    or a Contribution incorporated within the Work constitutes direct
    or contributory patent infringement, then any patent licenses
    granted to You under this License for that Work shall terminate
    as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
    Work or Derivative Works thereof in any medium, with or without
    modifications, and in Source or Object form, provided that You
    meet the following conditions:

    (a) You must give any other recipients of the Work or
        Derivative Works a copy of this License; and

    (b) You must cause any modified files to carry prominent notices
        stating that You changed the files; and

    (c) You must retain, in the Source form of any Derivative Works
        that You distribute, all copyright, patent, trademark, and
        attribution notices from the Source form of the Work,
        excluding those notices that do not pertain to any part of
        the Derivative Works; and

    (d) If the Work includes a "NOTICE" text file as part of its
        distribution, then any Derivative Works that You distribute must
        include a readable copy of the attribution notices contained
        within such NOTICE file, excluding those notices that do not
        pertain to any part of the Derivative Works, in at least one
        of the following places: within a NOTICE text file distributed
        as part of the Derivative Works; within the Source form or
        documentation, if provided along with the Derivative Works; or,
        within a display generated by the Derivative Works, if and
        wherever such third-party notices normally appear. The contents
        of the NOTICE file are for informational purposes only and
        do not modify the License. You may add Your own attribution
        notices within Derivative Works that You distribute, alongside
        or as an addendum to the NOTICE text from the Work, provided
        that such additional attribution notices cannot be construed
        as modifying the License.

    You may add Your own copyright statement to Your modifications and
    may provide additional or different license terms and conditions
    for use, reproduction, or distribution of Your modifications, or
    for any such Derivative Works as a whole, provided Your use,
    reproduction, and distribution of the Work otherwise complies with
    the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
    any Contribution intentionally submitted for inclusion in the Work
    by You to the Licensor shall be under the terms and conditions of
    this License, without any additional terms or conditions.
    Notwithstanding the above, nothing herein shall supersede or modify
    the terms of any separate license agreement you may have executed
    with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
    names, trademarks, service marks, or product names of the Licensor,
    except as required for reasonable and customary use in describing the
    origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
    agreed to in writing, Licensor provides the Work (and each
    Contributor provides its Contributions) on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    implied, including, without limitation, any warranties or conditions
    of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
    PARTICULAR PURPOSE. You are solely responsible for determining the
    appropriateness of using or redistributing the Work and assume any
    risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
    whether in tort (including negligence), contract, or otherwise,
    unless required by applicable law (such as deliberate and grossly
    negligent acts) or agreed to in writing, shall any Contributor be
    liable to You for damages, including any direct, indirect, special,
    incidental, or consequential damages of any character arising as a
    result of this License or out of the use or inability to use the
    Work (including but not limited to damages for loss of goodwill,
    work stoppage, computer failure or malfunction, or any and all
    other commercial damages or losses), even if such Contributor
    has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
    the Work or Derivative Works thereof, You may choose to offer,
    and charge a fee for, acceptance of support, warranty, indemnity,
    or other liability obligations and/or rights consistent with this
    License. However, in accepting such obligations, You may act only
    on Your own behalf and on Your sole responsibility, not on behalf
    of any other Contributor, and only if You agree to indemnify,
    defend, and hold each Contributor harmless for any liability
    incurred by, or claims asserted against, such Contributor by reason
    of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

    To apply the Apache License to your work, attach the following
    boilerplate notice, with the fields enclosed by brackets "[]"
    replaced with your own identifying information. (Don't include
    the brackets!)  The text should be enclosed in the appropriate
    comment syntax for the file format. We also recommend that a
    file or class name and description of purpose be included on the
    same "printed page" as the copyright notice for easier
    identification within third-party archives.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #


import contextlib
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (  # noqa: E501
    rescale_noise_cfg,
)

from .hooked_scheduler import HookedNoiseScheduler


# =========================================================================== #
#                          Helper Functions                                   #
# =========================================================================== #
def retrieve(
    io,
    unconditional: bool = False,
    guidance_scale: float = 7.5,
    return_both_cond_uncond: bool = False,
) -> torch.Tensor:
    if isinstance(io, tuple):
        if len(io) == 1:
            io = io[0].detach().cpu()
        else:
            raise ValueError("A tuple should have length of 1")
    elif isinstance(io, torch.Tensor):
        io = io.detach().cpu()
    else:
        raise ValueError("Input/Output must be a tensor, or 1-element tuple")
    if guidance_scale > 1.0:
        if return_both_cond_uncond:
            return io  # stacked tensor [uncond, cond]
        io_uncond, io_cond = io.chunk(2)
        if unconditional:
            return io_uncond
        return io_cond
    return io


# =========================================================================== #
#                    Hooked Diffusion Pipeline Class                     #
# =========================================================================== #
class HookedDiffusionAbstractPipeline:
    parent_cls = None

    def __init__(
        self, pipe: parent_cls, use_hooked_scheduler: bool = False
    ) -> None:
        if not isinstance(pipe, self.parent_cls):
            raise ValueError(
                f"Pipeline must be of type {self.parent_cls.__name__}."
            )

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if use_hooked_scheduler:
            pipe.scheduler = HookedNoiseScheduler(pipe.scheduler)

        print(f"{type(pipe.scheduler)=}")

        self.__dict__["pipe"] = pipe
        self.use_hooked_scheduler = use_hooked_scheduler

        # determine whether we are using SDXL:
        self.is_sdxl = isinstance(self.pipe, StableDiffusionXLPipeline)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(cls.parent_cls.from_pretrained(*args, **kwargs))

    def run_with_hooks(
        self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),  # noqa: B008
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        with_grad: bool = False,
        **kwargs,
    ):
        """
        Run the pipeline with hooks at specified positions.
        Returns the final output.

        Args:
            *args: Arguments to pass to the pipeline.
            position_hook_dict: A dictionary mapping positions to hooks.
                The keys are positions in the pipeline where the hooks should
                    be registered.
                The values are either a single hook or a list of hooks to be
                    registered at the specified position.
                Each hook should be a callable that takes three arguments:
                    (module, input, output).
            **kwargs: Keyword arguments to pass to the pipeline.
        """
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        # Define a context manager that is either torch.no_grad() or a dummy
        # null context
        context = (
            torch.no_grad() if not with_grad else contextlib.nullcontext()
        )

        try:
            (
                prompt_embeds,
                timesteps,
                latents,
                extra_step_kwargs,
                added_cond_kwargs,
            ) = self._prepare_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale,
                num_inference_steps,
                generator,
                latents,
                **kwargs,
            )

            with context:
                latents = self._denoise_loop(
                    timesteps,
                    latents,
                    guidance_scale,
                    extra_step_kwargs,
                    added_cond_kwargs,
                    prompt_embeds,
                    with_grad=with_grad,
                    **kwargs,
                )
                image = self._postprocess_latents(
                    latents, output_type, generator
                )
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []

        return image

    def run_with_cache(
        self,
        *args,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),  # noqa: B008
        return_both_cond_uncond: bool = False,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        latents: Optional[torch.Tensor] = None,
        positions_to_cache: List[str],
        output_type: Optional[str] = "pil",
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
        with_grad: bool = False,
        **kwargs,
    ):
        """
        Run the pipeline with caching at specified positions.

        This method allows you to cache the intermediate inputs and/or outputs
        of the pipeline at certain positions. The final output of the pipeline
        and a dictionary of cached values are returned.

        Args:
            *args: Arguments to pass to the pipeline.
            positions_to_cache (List[str]): A list of positions in the pipeline
                where intermediate inputs/outputs should be cached.
            save_input (bool, optional): If True, caches the input at each
                specified position. Defaults to False.
            save_output (bool, optional): If True, caches the output at each
                specified position. Defaults to True.
            **kwargs: Keyword arguments to pass to the pipeline.

        Returns:
            final_output: The final output of the pipeline after execution.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys are
                the specified positions
                and values are dictionaries containing the cached 'input'
                    and/or 'output' at each position,
                depending on the flags `save_input` and `save_output`.
        """
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        hooks = [
            self._register_cache_hook(
                position,
                cache_input,
                cache_output,
                unconditional,
                guidance_scale,
                return_both_cond_uncond,
            )
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]

        (
            prompt_embeds,
            timesteps,
            latents,
            extra_step_kwargs,
            added_cond_kwargs,
        ) = self._prepare_prompt(
            prompt,
            device,
            num_images_per_prompt,
            guidance_scale,
            num_inference_steps,
            generator,
            latents,
            **kwargs,
        )

        # Define a context manager that is either torch.no_grad() or a
        # dummy null context
        context = (
            torch.no_grad() if not with_grad else contextlib.nullcontext()
        )

        with context:
            latents = self._denoise_loop(
                timesteps,
                latents,
                guidance_scale,
                extra_step_kwargs,
                added_cond_kwargs,
                prompt_embeds,
                with_grad=with_grad,
                **kwargs,
            )

            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []

            cache_dict = {}
            if save_input:
                for position, block in cache_input.items():
                    cache_input[position] = torch.stack(block, dim=1)
                cache_dict["input"] = cache_input

            if save_output:
                for position, block in cache_output.items():
                    cache_output[position] = torch.stack(block, dim=1)
                cache_dict["output"] = cache_output

            image = self._postprocess_latents(latents, output_type, generator)

            return image, cache_dict

    def run_with_cache_intermediate(
        self,
        *args,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),  # noqa: B008
        unconditional: bool = False,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        latents: Optional[torch.Tensor] = None,
        positions_to_cache: List[str],
        output_type: Optional[str] = "pil",
        save_input: bool = False,
        save_output: bool = True,
        with_grad: bool = False,
        **kwargs,
    ):
        """
        Run the pipeline with caching at specified positions saving
            intermediate predictions of x_0 after every timestep.

        This method allows you to cache the intermediate inputs and/or outputs
            of the pipelinecat certain positions. The final output of the
            pipeline and a dictionary of cached values are returned.

        Args:
            *args: Arguments to pass to the pipeline.
            positions_to_cache (List[str]): A list of positions in the pipeline
                where intermediate inputs/outputs should be cached.
            save_input (bool, optional): If True, caches the input at each
                specified position. Defaults to False.
            save_output (bool, optional): If True, caches the output at each
                specified position.
                Defaults to True.
            **kwargs: Keyword arguments to pass to the pipeline.

        Returns:
            output: x_0 predicted after each timestep.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys
                are the specified positions and values are dictionaries
                containing the cached 'input' and/or 'output' at each position,
                depending on the flags `save_input` and `save_output`.
        """

        assert isinstance(
            self.pipe.scheduler, DDIMScheduler
        ), "Only DDIMScheduler is supported for intermediate caching"

        # Prepare prompt embeds and additional conditioning params
        (
            prompt_embeds,
            timesteps,
            latents,
            extra_step_kwargs,
            added_cond_kwargs,
        ) = self._prepare_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents=latents,
            **kwargs,
        )

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, 0.0)
        added_cond_kwargs = None
        ## END PREPARE ##

        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        all_intermediate_latents = []
        hooks = [
            self._register_cache_hook(
                position,
                cache_input,
                cache_output,
                unconditional,
                guidance_scale,
            )
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]

        # Define a context manager that is either torch.no_grad() or a
        # dummy null context
        context = (
            torch.no_grad() if not with_grad else contextlib.nullcontext()
        )

        with context:
            # Denoising loop
            self._num_timesteps = len(timesteps)

            # Get SDXL specifici denoising_end parameter:
            denoising_end = kwargs.get("denoising_end", None)
            # Apply denoising_end:
            if (
                denoising_end is not None
                and isinstance(denoising_end, float)
                and denoising_end > 0
                and denoising_end < 1
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (
                            denoising_end
                            * self.scheduler.config.num_train_timesteps
                        )
                    )
                )
                num_inference_steps = len(
                    list(
                        filter(
                            lambda ts: ts >= discrete_timestep_cutoff,
                            timesteps,
                        )
                    )
                )
                timesteps = timesteps[:num_inference_steps]
                self._num_timesteps = len(timesteps)

            timestep_cond = None
            if (
                self.is_sdxl
                and self.unet.config.time_cond_proj_dim is not None
            ):
                guidance_scale_tensor = torch.tensor(
                    self.pipe.guidance_scale - 1
                ).repeat(latents.shape[0])
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=self.unet.config.time_cond_proj_dim,
                ).to(device=latents.device, dtype=latents.dtype)

            guidance_rescale = kwargs.get("guidance_rescale", 0.0)

            # Early stopping:
            max_denoising_steps = kwargs.get("max_denoising_steps", None)

            for step_idx, t in enumerate(timesteps):
                # Early stopping condition:
                if (
                    max_denoising_steps is not None
                    and step_idx >= max_denoising_steps
                ):
                    break

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if guidance_scale > 1.0
                    else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=kwargs.get(
                        "cross_attention_kwargs", None
                    ),
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # Apply guidance rescale for SDXL:
                    if self.is_sdxl and guidance_rescale > 0.0:

                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_out = self.pipe.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=True,
                )
                latents = scheduler_out.prev_sample
                pred_original_sample = scheduler_out.pred_original_sample
                all_intermediate_latents.append(pred_original_sample)

            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []

            cache_dict = {}
            if save_input:
                for position, block in cache_input.items():
                    cache_input[position] = torch.stack(block, dim=1)
                cache_dict["input"] = cache_input

            if save_output:
                for position, block in cache_output.items():
                    cache_output[position] = torch.stack(block, dim=1)
                cache_dict["output"] = cache_output

            if not output_type == "latent":
                image = self.pipe.vae.decode(
                    latents / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                if len(all_intermediate_latents) > 0:
                    for i in range(len(all_intermediate_latents)):
                        all_intermediate_latents[i] = self.pipe.vae.decode(
                            all_intermediate_latents[i]
                            / self.pipe.vae.config.scaling_factor,
                            return_dict=False,
                            generator=generator,
                        )[0]
            else:
                image = latents
            do_denormalize = [True] * image.shape[0]

            image = self.pipe.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )
            if len(all_intermediate_latents) > 0:
                for i in range(len(all_intermediate_latents)):
                    all_intermediate_latents[i] = (
                        self.pipe.image_processor.postprocess(
                            all_intermediate_latents[i],
                            output_type=output_type,
                            do_denormalize=do_denormalize,
                        )
                    )

            if output_type == "latent":
                image = image.cpu().numpy()

            return image, all_intermediate_latents, cache_dict

    def run_with_hooks_and_cache(
        self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        positions_to_cache: Optional[List[str]] = None,
        unconditional: bool = False,
        guidance_scale: float = 7.5,
        save_input: bool = False,
        save_output: bool = True,
        **kwargs,
    ):
        """
        Run the pipeline with hooks and caching at specified positions.

        This method allows you to register hooks at certain positions in the
        pipeline and cache intermediate inputs and/or outputs at specified
        positions. Hooks can be used for inspecting or modifying the pipeline's
        execution, and caching stores intermediate values for later inspection
        or use.

        Args:
            *args: Arguments to pass to the pipeline.
            position_hook_dict Dict[str, Union[Callable, List[Callable]]]:
                A dictionary where the keys are the positions in the pipeline,
                and the values are hooks (either a single hook or a list of
                hooks) to be registered at those positions.
                Each hook should be a callable that accepts three arguments:
                (module, input, output).
            positions_to_cache (List[str], optional): A list of positions in
                the pipeline where intermediate inputs/outputs should be
                cached. Defaults to an empty list.
            save_input (bool, optional): If True, caches the input at each
                specified position. Defaults to False.
            save_output (bool, optional): If True, caches the output at each
                specified position. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the pipeline.

        Returns:
            final_output: The final output of the pipeline after execution.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys
                are the specified positions and values are dictionaries
                containing the cached 'input' and/or 'output' at each position,
                depending on the flags `save_input` and `save_output`.
        """
        if positions_to_cache is None:
            positions_to_cache = []
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        hooks = [
            self._register_cache_hook(
                position,
                cache_input,
                cache_output,
                unconditional,
                guidance_scale,
            )
            for position in positions_to_cache
        ]

        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict["input"] = cache_input

        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict["output"] = cache_output

        return output, cache_dict

    def _locate_block(self, position: str) -> nn.Module:
        """
        Locate the block at the specified position in the pipeline.

        Args:
            position (str): The position in the pipeline to locate.

        Returns:
            block (nn.Module): The block located at the specified position.
        """
        block = self.pipe

        try:
            for step in position.split("."):
                if step.isdigit():
                    step = int(step)
                    block = block[step]
                else:
                    block = getattr(block, step)
            return block
        except AttributeError as e:
            print(f"Error locating block at position '{position}': {e}")
            raise

    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        unconditional: bool = False,
        guidance_scale: float = 7.5,
        return_both_cond_uncond: bool = False,
    ):
        block = self._locate_block(position)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(
                    input,
                    unconditional,
                    guidance_scale,
                    return_both_cond_uncond,
                )
                if len(input_to_cache.shape) == 4:
                    input_to_cache = input_to_cache.view(
                        input_to_cache.shape[0], input_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_input[position].append(input_to_cache)

            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve(
                    output,
                    unconditional,
                    guidance_scale,
                    return_both_cond_uncond,
                )
                if len(output_to_cache.shape) == 4:
                    output_to_cache = output_to_cache.view(
                        output_to_cache.shape[0], output_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_output[position].append(output_to_cache)

        return block.register_forward_hook(hook, with_kwargs=True)

    def _register_general_hook(self, position, hook):
        if position == "scheduler_pre":
            if not self.use_hooked_scheduler:
                raise ValueError(
                    """
                    Cannot register hooks on scheduler without using hooked\
                    scheduler
                    """
                )
            self.pipe.scheduler.pre_hooks.append(hook)
            return
        elif position == "scheduler_post":
            if not self.use_hooked_scheduler:
                raise ValueError(
                    """
                    Cannot register hooks on scheduler without using hooked\
                    scheduler
                    """
                )
            self.pipe.scheduler.post_hooks.append(hook)
            return

        block = self._locate_block(position)
        return block.register_forward_hook(hook)

    def _prepare_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        guidance_scale,
        num_inference_steps,
        generator,
        latents,
        **kwargs,
    ):
        ## PREPARE PROMPT from StableDiffusionPipeline ##
        # 0. Default height and width to unet
        default_height = (
            self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        )
        default_width = (
            self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        )

        height = kwargs.get("height", default_height)
        width = kwargs.get("width", default_width)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        # Handle SDXL specifics:
        # based on __call__ method of StableDiffusionXLPipeline:
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L712 # noqa: E501
        if self.is_sdxl:
            # Get SDXL specific kwargs; defaults taken from __call__ method
            prompt_2 = kwargs.get("prompt_2", None)
            negative_prompt = kwargs.get("negative_prompt", None)
            negative_prompt_2 = kwargs.get("negative_prompt_2", None)
            pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds", None)
            negative_pooled_prompt_embeds = kwargs.get(
                "negative_pooled_prompt_embeds", None
            )
            lora_scale = kwargs.get("lora_scale", None)
            clip_skip = kwargs.get("clip_skip", None)
            original_size = kwargs.get("original_size", (height, width))
            crops_coords_top_left = kwargs.get("crops_coords_top_left", (0, 0))
            target_size = kwargs.get("target_size", (height, width))

            # encode prompt:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler, num_inference_steps, device, None, None
            )

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

            # 6. Prepare extra step kwargs
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(
                generator, 0.0
            )

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds

            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(
                    pooled_prompt_embeds.shape[-1]
                )
            else:
                text_encoder_projection_dim = int(
                    self.text_encoder_2.config.projection_dim
                )

            add_time_ids = self._get_add_time_ids(
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                target_size=target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )

            # defaults taken from __call__ method of StableDiffusionXLPipeline:
            negative_original_size = kwargs.get("negative_original_size", None)
            negative_crops_coords_top_left = kwargs.get(
                "negative_crops_coords_top_left", (0, 0)
            )
            negative_target_size = kwargs.get("negative_target_size", None)

            if (
                negative_original_size is not None
                and negative_target_size is not None
            ):
                negative_add_time_ids = self._get_add_time_ids(
                    original_size=negative_original_size,
                    crops_coords_top_left=negative_crops_coords_top_left,
                    target_size=negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            # Apply classifier free guidance
            if guidance_scale > 1.0:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                add_text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                )
                add_time_ids = torch.cat(
                    [negative_add_time_ids, add_time_ids], dim=0
                )

            # Move embeddings to device
            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(
                batch_size * num_images_per_prompt, 1
            )

            # Handle IP Adapter for SDXL if provided
            ip_adapter_image = kwargs.get("ip_adapter_image", None)
            ip_adapter_image_embeds = kwargs.get(
                "ip_adapter_image_embeds", None
            )

            if (
                ip_adapter_image is not None
                or ip_adapter_image_embeds is not None
            ):
                image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    guidance_scale > 1.0,
                )
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                    "image_embeds": image_embeds,
                }
            else:
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }

        else:
            # Standard SD pipeline:
            negative_prompt = kwargs.get("negative_prompt", None)
            lora_scale = kwargs.get("lora_scale", None)
            clip_skip = kwargs.get("clip_skip", None)

            # encode prompt:
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                negative_prompt=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )

            if guidance_scale > 1.0:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds]
                )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler, num_inference_steps, device, None, None
            )

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(
                generator, 0.0
            )
            added_cond_kwargs = None

        return (
            prompt_embeds,
            timesteps,
            latents,
            extra_step_kwargs,
            added_cond_kwargs,
        )

    def _denoise_loop(
        self,
        timesteps,
        latents,
        guidance_scale,
        extra_step_kwargs,
        added_cond_kwargs,
        prompt_embeds,
        with_grad: bool = False,
        **kwargs,
    ):
        # Define a context manager that is either torch.no_grad() or a
        # dummy null context
        context = (
            torch.no_grad() if not with_grad else contextlib.nullcontext()
        )

        with context:
            self._num_timesteps = len(timesteps)

            # Get SDXL specifici denoising_end parameter:
            denoising_end = kwargs.get("denoising_end", None)
            # Apply denoising_end:
            if (
                denoising_end is not None
                and isinstance(denoising_end, float)
                and denoising_end > 0
                and denoising_end < 1
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (
                            denoising_end
                            * self.scheduler.config.num_train_timesteps
                        )
                    )
                )
                num_inference_steps = len(
                    list(
                        filter(
                            lambda ts: ts >= discrete_timestep_cutoff,
                            timesteps,
                        )
                    )
                )
                timesteps = timesteps[:num_inference_steps]
                self._num_timesteps = len(timesteps)

            # Early stopping:
            max_denoising_steps = kwargs.get("max_denoising_steps", None)

            # Get SDXL specific guidance rescale parameter:
            guidance_rescale = kwargs.get("guidance_rescale", 0.0)
            # 9. Optionally get Guidance Scale Embedding
            timestep_cond = None
            if (
                self.is_sdxl
                and self.unet.config.time_cond_proj_dim is not None
            ):
                guidance_scale_tensor = torch.tensor(
                    self.pipe.guidance_scale - 1
                ).repeat(latents.shape[0])
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=self.unet.config.time_cond_proj_dim,
                ).to(device=latents.device, dtype=latents.dtype)

            for step_idx, t in enumerate(timesteps):
                # Early stopping condition:
                if (
                    max_denoising_steps is not None
                    and step_idx >= max_denoising_steps
                ):
                    break

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if guidance_scale > 1.0
                    else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=kwargs.get(
                        "cross_attention_kwargs", None
                    ),
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # Apply guidance rescale for SDXL:
                    if self.is_sdxl and guidance_rescale > 0.0:

                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

            return latents

    def _postprocess_latents(self, latents, output_type, generator):
        if not output_type == "latent":

            # for SDXL, special handling of VAE upcasting if needed
            if self.is_sdxl:
                needs_upcasting = (
                    self.pipe.vae.dtype == torch.float16
                    and self.pipe.vae.config.force_upcast
                )
                if needs_upcasting:
                    self.pipe.upcast_vae()
                    latents = latents.o(
                        next(
                            iter(self.pipe.vae.post_quant_conv.parameters())
                        ).dtype
                    )
                elif latents.dtype != self.pipe.vae.dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272 # noqa: E501
                        self.pipe.vae = self.pipe.vae.to(latents.dtype)
                has_latents_mean = (
                    hasattr(self.pipe.vae.config, "latents_mean")
                    and self.pipe.vae.config.latents_mean is not None
                )
                has_latents_std = (
                    hasattr(self.pipe.vae.config, "latents_std")
                    and self.pipe.vae.config.latents_std is not None
                )

                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(self.pipe.vae.config.latents_mean)
                        .view(1, 4, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.pipe.vae.config.latents_std)
                        .view(1, 4, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents = (
                        latents
                        * latents_std
                        / self.pipe.vae.config.scaling_factor
                        + latents_mean
                    )
                else:
                    latents = latents / self.pipe.vae.config.scaling_factor

                image = self.pipe.vae.decode(
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0]
                if needs_upcasting:
                    self.pipe.vae.to(dtype=torch.float16)

                if self.pipe.watermark is not None:
                    print("Applying watermark to SDXL generated image")
                    image = self.watermark.apply_watermark(image)
            else:
                # Standard SD scaling
                latents = latents / self.pipe.vae.config.scaling_factor

                image = self.pipe.vae.decode(
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0]
        else:
            image = latents
        do_denormalize = [True] * image.shape[0]

        image = self.pipe.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if output_type == "latent":
            # Only convert to numpy if the tensor doesn't require gradients
            if image.requires_grad:
                image = image
            else:
                image = image.cpu().numpy()
        return image

    def to(self, *args, **kwargs):
        self.pipe = self.pipe.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def __setattr__(self, name, value):
        return setattr(self.pipe, name, value)

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


class HookedStableDiffusionPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = DiffusionPipeline


class HookedStableDiffusionXLPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = StableDiffusionXLPipeline
