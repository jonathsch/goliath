import os

import slangtorch
import torch

#######################################
# Disney BRDF
#######################################

slang_disney_pbr = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "disney.slang"))


class _disney_pbr_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos, light_intensity)
        ctx.min_roughness = min_roughness
        return slang_disney_pbr.pbr_fwd(
            kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness, kd.dim()
        )

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos, light_intensity = ctx.saved_tensors
        min_roughness = ctx.min_roughness
        kd_grad, arm_grad, pos_grad, nrm_grad, lint_grad = slang_disney_pbr.pbr_bwd(
            kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness, kd.dim(), dout.contiguous()
        )
        # assert torch.all(torch.isfinite(kd_grad)), f"kd_grad is not finite: {kd_grad}"
        # assert torch.all(torch.isfinite(arm_grad)), f"arm_grad is not finite: {arm_grad}"
        # assert torch.all(torch.isfinite(pos_grad)), f"pos_grad is not finite: {pos_grad}"
        # assert torch.all(torch.isfinite(nrm_grad)), f"nrm_grad is not finite: {nrm_grad}"
        # assert torch.all(torch.isfinite(lint_grad)), f"lint_grad is not finite: {lint_grad}"
        return kd_grad, arm_grad, pos_grad, nrm_grad, None, None, None, None


def disney_pbr_slang(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness=0.08):
    # assert kd.dim() == 3 and kd.shape[-1] == 3, f"kd must be of shape (B, N, 3), got {kd.shape}"
    # assert arm.dim() == 3 and arm.shape[-1] == 3, f"arm must be of shape (B, N, 3), got {arm.shape}"
    # assert pos.dim() == 3 and pos.shape[-1] == 3, f"pos must be of shape (B, N, 3), got {pos.shape}"
    # assert nrm.dim() == 3 and nrm.shape[-1] == 3, f"nrm must be of shape (B, N, 3), got {nrm.shape}"
    assert view_pos.dim() == 2 and view_pos.shape[-1] == 3, f"view_pos must be of shape (B, 3), got {view_pos.shape}"
    assert light_pos.dim() == 3 and light_pos.shape[-1] == 3, (
        f"light_pos must be of shape (B, L, 3), got {light_pos.shape}"
    )
    assert (
        light_intensity.dim() == 3
        and light_intensity.shape[-1] == 3
        and light_intensity.shape[-2] == light_pos.shape[-2]
    ), f"light_intensity must be of shape (B, L, 1), got {light_intensity.shape}"

    kd = kd.contiguous()
    arm = arm.contiguous()
    pos = pos.contiguous()
    nrm = nrm.contiguous()
    view_pos = view_pos.contiguous()
    light_pos = light_pos.contiguous()
    light_intensity = light_intensity.contiguous()
    return _disney_pbr_func.apply(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness)
