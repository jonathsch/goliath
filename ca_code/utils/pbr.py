import torch

################################################################################
# PBR's implementation of GGX specular
################################################################################

specular_epsilon = 1e-4


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / length(x, eps)


def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0, 1), mode="constant", value=w)


def _bsdf_lambert(nrm, wi):
    return torch.clamp(dot(nrm, wi), min=0.0) / torch.pi


def bsdf_fresnel_shlick(f0, f90, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0


def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * torch.pi)


def ndf_blinn_phong(alpha, cos_theta):
    return 0.5 * (alpha + 2.0) * (1.0 / torch.pi) * torch.pow(cos_theta, alpha)


def bsdf_ndf_blinn_phong_mix(blend, cos_theta):
    return blend * ndf_blinn_phong(12.0, cos_theta) + (1.0 - blend) * ndf_blinn_phong(48.0, cos_theta)


def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res


def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)


def bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    _alpha = torch.clamp(alpha, min=min_roughness * min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = safe_normalize(wo + wi)
    woDotN = dot(wo, nrm)
    wiDotN = dot(wi, nrm)
    woDotH = dot(wo, h)
    nDotH = dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))


def pbr_python(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, spec_nrm=None):
    """
    Args:
        kd: (B, N, 3)
        ks: (B, N, 3)
        pos: (B, N, 3)
        nrm: (B, N, 3)
        view_pos: (B, 3)
        light_pos: (B, L, 3)
        light_intensity: (B, L, 3)
    """
    wi = safe_normalize(light_pos[:, None, :, :] - pos[:, :, None, :])  # [B, N, L, 3]
    wo = safe_normalize(view_pos[:, None, :] - pos)  # [B, N, 3]

    spec_str = arm[..., 0:1]  # x component
    roughness = arm[..., 1:2]  # y component
    metallic = arm[..., 2:3]  # z component
    ks = 0.04 * (1.0 - metallic) + kd * metallic
    kd = kd * (1.0 - metallic)
    alpha = roughness**2

    # relative_light_intensity = (
    #     light_intensity[:, None, :, :] / (light_pos[:, None, :, :] - pos[:, :, None, :]).norm(dim=-1, keepdim=True) ** 2
    # )

    diffuse = kd * (_bsdf_lambert(nrm[:, :, None, :], wi) * light_intensity[:, None, :, :]).sum(dim=-2)  # [B, N, 3]
    spec_nrm = nrm if spec_nrm is None else spec_nrm
    specular = (
        bsdf_pbr_specular(ks[:, :, None, :], spec_nrm[:, :, None, :], wo[:, :, None, :], wi, alpha[:, :, None, :])
        * light_intensity[:, None, :, :]
    ).sum(dim=-2) * spec_str  # [B, N, 3]

    return diffuse, specular