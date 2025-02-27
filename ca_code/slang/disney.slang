import utils;

[BackwardDifferentiable]
float lambert(const float3 nrm, const float3 wi)
{
    return max(dot(nrm, wi) / 3.14159265358979323846f, 0.0f);
}

[BackwardDifferentiable]
float3 fresnelSchlick(const float3 f0, const float3 f90, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = pow(1.0f - _cosTheta, 5.0f);
    return f0 * (1.0f - scale) + f90 * scale;
}

[BackwardDifferentiable]
float ndfGGX(const float alphaSqr, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * 3.14159265358979323846f);
}

[BackwardDifferentiable]
float lambdaGGX(const float alphaSqr, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    return 0.5f * (sqrt(1.0f + alphaSqr * tanThetaSqr) - 1.0f);
}

[BackwardDifferentiable]
float maskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO)
{
    float lambdaI = lambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = lambdaGGX(alphaSqr, cosThetaO);
    return 1.0f / (1.0f + lambdaI + lambdaO);
}

[BackwardDifferentiable]
float3 pbrSpecular(const float3 col, const float3 nrm, const float3 wo, const float3 wi, const float alpha, no_diff float min_roughness)
{
    float _alpha = clamp(alpha, min_roughness * min_roughness, 1.0f);
    float alphaSqr = _alpha * _alpha;
    const float SPECULAR_EPSILON = 1e-4f;
    float woDotN = dot(wo, nrm);
    float wiDotN = dot(wi, nrm);

    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);

    float3 res = float3(0.0f);
    if (frontfacing)
    {
        float3 h = safeNormalize(wo + wi);
        float woDotH = dot(wo, h);
        float nDotH = dot(nrm, h);
        float D = ndfGGX(alphaSqr, nDotH);
        float G = maskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN);
        float3 F = fresnelSchlick(col, 1.0f, woDotH);
        res = F * D * G * 0.25 / woDotN;
    }
    return res;
}

[BackwardDifferentiable]
float3 lambertBSDF(const float3 kd,
                   const float3 pos,
                   const float3 nrm,
                   const float3 light_intensity,
                   no_diff const float3 light_pos)
{
    float3 wi = normalize(light_pos - pos);
    float diff = lambert(nrm, wi);
    float3 diffuse = kd * diff;

    // let distLight = length(light_pos - pos);
    // let incidentLight = distLight > 1e-3 ? (light_intensity / distLight) : 0.0f;

    return diffuse * light_intensity;
}

[BackwardDifferentiable]
float3 specularBSDF(const float3 kd,
                    const float3 arm,
                    const float3 pos,
                    const float3 nrm,
                    const float3 light_intensity,
                    no_diff const float3 view_pos,
                    no_diff const float3 light_pos,
                    no_diff float min_roughness)
{
    float3 wo = safeNormalize(view_pos - pos);
    float3 wi = safeNormalize(light_pos - pos);

    float alpha = arm.y * arm.y;
    float3 spec_col = (0.04f * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x);
    float3 specular = pbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness);

    let distLight = length(light_pos - pos);
    let incidentLight = distLight > 1e-3 ? (light_intensity / distLight) : 0.0f;
    return specular * incidentLight;
}

//////////////////////////////////////////////////////////////////////////
// Full PBR BSDF
//////////////////////////////////////////////////////////////////////////
[BackwardDifferentiable]
float3 pbrBSDF(const float3 kd,
               const float3 arm,
               const float3 pos,
               const float3 nrm,
               const float3 light_intensity,
               no_diff const float3 view_pos,
               no_diff const float3 light_pos,
               no_diff float min_roughness)
{
    float3 wo = safeNormalize(view_pos - pos);
    float3 wi = safeNormalize(light_pos - pos);

    float alpha = arm.y * arm.y;
    float spec_str = arm.x;
    float metallic = arm.z;
    float3 spec_col = 0.04f * (1.0f - metallic) + kd * metallic;
    float3 diff_col = kd * (1.0f - metallic);

    float3 diffuse = diff_col * lambert(nrm, wi);
    float3 specular = spec_str * pbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness);

    // let distLight = length(light_pos - pos);
    // let incidentLight = distLight > 1e-3 ? (light_intensity / (distLight * distLight)) : 0.0f;

    return (diffuse + specular) * light_intensity;
}

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////
[CudaKernel]
void pbr_bn_fwd_kernel(TensorView<float3> kd,              // B x N x 3
                       TensorView<float3> arm,             // B x N x 3
                       TensorView<float3> pos,             // B x N x 3
                       TensorView<float3> nrm,             // B x N x 3
                       TensorView<float3> view_pos,        // B x 3
                       TensorView<float3> light_pos,       // B x L x 3
                       TensorView<float3> light_intensity, // B x L x 3
                       const float min_roughness,
                       TensorView<float3> output)
{
    let idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(1) || idx.y > output.size(0))
    {
        return;
    }

    let idxr = uint2(idx.y, idx.x);
    float3 v_pos = view_pos[idx.y];

    let n_lights = light_pos.size(1);
    output[idxr] = float3(0.0f);

    for (uint i = 0; i < n_lights; i++)
    {
        float3 l_pos = light_pos[idx.y, i];
        float3 l_int = light_intensity[idx.y, i];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        let res = pbrBSDF(kd[idxr], arm[idxr], pos[idxr], nrm[idxr], l_int, v_pos, l_pos, min_roughness);
        output[idxr] += res;
    }
}

[CudaKernel]
void pbr_bn_bwd_kernel(TensorView<float3> kd,
                       TensorView<float3> arm,
                       TensorView<float3> pos,
                       TensorView<float3> nrm,
                       TensorView<float3> view_pos,
                       TensorView<float3> light_pos,
                       TensorView<float3> light_intensity,
                       const float min_roughness,
                       TensorView<float3> kd_grad,
                       TensorView<float3> arm_grad,
                       TensorView<float3> pos_grad,
                       TensorView<float3> nrm_grad,
                       TensorView<float3> light_intensity_grad,
                       TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(1) || idx.y > grad_out.size(0))
    {
        return;
    }

    uint2 idxr = uint2(idx.y, idx.x);

    float3 v_pos = view_pos[idx.y];

    let n_lights = light_pos.size(1);
    float3 kd_accum = float3(0.0f);
    float3 arm_accum = float3(0.0f);
    float3 pos_accum = float3(0.0f);
    float3 nrm_accum = float3(0.0f);

    let kd = kd[idxr];
    let arm = arm[idxr];
    let pos = pos[idxr];
    let nrm = nrm[idxr];
    let dout = grad_out[idxr];

    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 light_idx = uint2(idx.y, i);
        float3 l_pos = light_pos[light_idx];
        float3 l_int = light_intensity[light_idx];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        var dp_kd = diffPair(kd);
        var dp_arm = diffPair(arm);
        var dp_pos = diffPair(pos);
        var dp_nrm = diffPair(nrm);
        var dp_lint = diffPair(l_int);

        __bwd_diff(pbrBSDF)(dp_kd, dp_arm, dp_pos, dp_nrm, dp_lint, v_pos, l_pos,
                            min_roughness, dout);

        kd_accum += dp_kd.d;
        arm_accum += dp_arm.d;
        pos_accum += dp_pos.d;
        nrm_accum += dp_nrm.d;

        // float3 old_light_grad;
        // light_intensity_grad.InterlockedAdd(light_idx, dp_lint.d, old_light_grad);
    }

    kd_grad[idxr] = clamp(select(isfinite(kd_accum), kd_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    arm_grad[idxr] = clamp(select(isfinite(arm_accum), arm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    pos_grad[idxr] = clamp(select(isfinite(pos_accum), pos_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    nrm_grad[idxr] = clamp(select(isfinite(nrm_accum), nrm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
}

[CudaKernel]
void pbr_nhwc_fwd_kernel(TensorView<float3> kd,            // B x H x W x 3
                       TensorView<float3> arm,             // B x H x W x 3
                       TensorView<float3> pos,             // B x H x W x 3
                       TensorView<float3> nrm,             // B x H x W x 3
                       TensorView<float3> view_pos,        // B x 3
                       TensorView<float3> light_pos,       // B x L x 3
                       TensorView<float3> light_intensity, // B x L x 3
                       const float min_roughness,
                       TensorView<float3> output)
{
    let idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(2) || idx.y > output.size(1) || idx.z > output.size(0))
    {
        return;
    }

    let idxr = uint3(idx.z, idx.y, idx.x);
    float3 v_pos = view_pos[idx.z];

    let n_lights = light_pos.size(1);
    float3 res = float3(0.0f);

    for (uint i = 0; i < n_lights; i++)
    {
        float3 l_pos = light_pos[idx.z, i];
        float3 l_int = light_intensity[idx.z, i];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        res += pbrBSDF(kd[idxr], arm[idxr], pos[idxr], nrm[idxr], l_int, v_pos, l_pos, min_roughness);
    }
    output[idxr] = res;
}

[CudaKernel]
void pbr_nhwc_bwd_kernel(TensorView<float3> kd,
                       TensorView<float3> arm,
                       TensorView<float3> pos,
                       TensorView<float3> nrm,
                       TensorView<float3> view_pos,
                       TensorView<float3> light_pos,
                       TensorView<float3> light_intensity,
                       const float min_roughness,
                       TensorView<float3> kd_grad,
                       TensorView<float3> arm_grad,
                       TensorView<float3> pos_grad,
                       TensorView<float3> nrm_grad,
                       TensorView<float3> light_intensity_grad,
                       TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(2) || idx.y > grad_out.size(1) || idx.z > grad_out.size(0))
    {
        return;
    }

    let idxr = uint3(idx.z, idx.y, idx.x);

    float3 v_pos = view_pos[idx.z];

    let n_lights = light_pos.size(1);
    float3 kd_accum = float3(0.0f);
    float3 arm_accum = float3(0.0f);
    float3 pos_accum = float3(0.0f);
    float3 nrm_accum = float3(0.0f);

    let kd = kd[idxr];
    let arm = arm[idxr];
    let pos = pos[idxr];
    let nrm = nrm[idxr];
    let dout = grad_out[idxr];

    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 light_idx = uint2(idx.z, i);
        float3 l_pos = light_pos[light_idx];
        float3 l_int = light_intensity[light_idx];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        var dp_kd = diffPair(kd);
        var dp_arm = diffPair(arm);
        var dp_pos = diffPair(pos);
        var dp_nrm = diffPair(nrm);
        var dp_lint = diffPair(l_int);

        __bwd_diff(pbrBSDF)(dp_kd, dp_arm, dp_pos, dp_nrm, dp_lint, v_pos, l_pos,
                            min_roughness, dout);

        kd_accum += dp_kd.d;
        arm_accum += dp_arm.d;
        pos_accum += dp_pos.d;
        nrm_accum += dp_nrm.d;

        // float3 old_light_grad;
        // light_intensity_grad.InterlockedAdd(light_idx, dp_lint.d, old_light_grad);
    }

    kd_grad[idxr] = clamp(select(isfinite(kd_accum), kd_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    arm_grad[idxr] = clamp(select(isfinite(arm_accum), arm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    pos_grad[idxr] = clamp(select(isfinite(pos_accum), pos_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    nrm_grad[idxr] = clamp(select(isfinite(nrm_accum), nrm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
}

[CudaKernel]
void lambert_bn_fwd_kernel(TensorView<float3> kd,              // B x N x 3
                           TensorView<float3> pos,             // B x N x 3
                           TensorView<float3> nrm,             // B x N x 3
                           TensorView<float3> light_pos,       // B x L x 3
                           TensorView<float3> light_intensity, // B x L x 1
                           TensorView<float3> output)
{
    let idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(1) || idx.y > output.size(0))
    {
        return;
    }

    let idxr = uint2(idx.y, idx.x);

    let n_lights = light_pos.size(1);
    float3 res = float3(0.0f);

    for (uint i = 0; i < n_lights; i++)
    {
        float3 l_pos = light_pos[idx.y, i];
        float3 l_int = light_intensity[idx.y, i];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        res += lambertBSDF(kd[idxr], pos[idxr], nrm[idxr], l_int, l_pos);
    }

    output[idxr] = res;
}

[CudaKernel]
void lambert_bn_bwd_kernel(TensorView<float3> kd,
                           TensorView<float3> pos,
                           TensorView<float3> nrm,
                           TensorView<float3> light_pos,
                           TensorView<float3> light_intensity,
                           TensorView<float3> kd_grad,
                           TensorView<float3> pos_grad,
                           TensorView<float3> nrm_grad,
                           TensorView<float3> lint_grad,
                           TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(1) || idx.y > grad_out.size(0))
    {
        return;
    }

    uint2 idxr = uint2(idx.y, idx.x);

    let n_lights = light_pos.size(1);
    float3 kd_accum = float3(0.0f);
    float3 pos_accum = float3(0.0f);
    float3 nrm_accum = float3(0.0f);

    let kd = kd[idxr];
    let pos = pos[idxr];
    let nrm = nrm[idxr];
    let dout = grad_out[idxr];

    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 light_idx = uint2(idx.y, i);
        float3 l_pos = light_pos[light_idx];
        float3 l_int = light_intensity[light_idx];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        var dp_kd = diffPair(kd);
        var dp_pos = diffPair(pos);
        var dp_nrm = diffPair(nrm);
        var dp_lint = diffPair(l_int);

        __bwd_diff(lambertBSDF)(dp_kd, dp_pos, dp_nrm, dp_lint, l_pos, dout);

        kd_accum += dp_kd.d;
        pos_accum += dp_pos.d;
        nrm_accum += dp_nrm.d;

        // float3 old_lint_grad;
        // lint_grad.InterlockedAdd(idx.y, dp_lint.d, old_lint_grad);
    }

    kd_grad[idxr] = clamp(select(isfinite(kd_accum), kd_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    pos_grad[idxr] = clamp(select(isfinite(pos_accum), pos_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    nrm_grad[idxr] = clamp(select(isfinite(nrm_accum), nrm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
}

[CudaKernel]
void lambert_nhwc_fwd_kernel(TensorView<float3> kd,            // B x H x W x 3
                           TensorView<float3> pos,             // B x H x W x 3
                           TensorView<float3> nrm,             // B x H x W x 3
                           TensorView<float3> light_pos,       // B x L x 3
                           TensorView<float3> light_intensity, // B x L x 3
                           TensorView<float3> output)
{
    let idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(2) || idx.y > output.size(1) || idx.z > output.size(0))
    {
        return;
    }

    let idxr = uint3(idx.z, idx.y, idx.x);

    let n_lights = light_pos.size(1);
    float3 res = float3(0.0f);

    for (uint i = 0; i < n_lights; i++)
    {
        float3 l_pos = light_pos[idx.z, i];
        float3 l_int = light_intensity[idx.z, i];

        res += lambertBSDF(kd[idxr], pos[idxr], nrm[idxr], l_int, l_pos);
    }

    output[idxr] = res;
}

[CudaKernel]
void lambert_nhwc_bwd_kernel(TensorView<float3> kd,
                           TensorView<float3> pos,
                           TensorView<float3> nrm,
                           TensorView<float3> light_pos,
                           TensorView<float3> light_intensity,
                           TensorView<float3> kd_grad,
                           TensorView<float3> pos_grad,
                           TensorView<float3> nrm_grad,
                           TensorView<float3> lint_grad,
                           TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(2) || idx.y > grad_out.size(1) || idx.z > grad_out.size(0))
    {
        return;
    }

    let idxr = uint3(idx.z, idx.y, idx.x);

    let n_lights = light_pos.size(1);
    float3 kd_accum = float3(0.0f);
    float3 pos_accum = float3(0.0f);
    float3 nrm_accum = float3(0.0f);

    let kd = kd[idxr];
    let pos = pos[idxr];
    let nrm = nrm[idxr];
    let dout = grad_out[idxr];

    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 light_idx = uint2(idx.z, i);
        float3 l_pos = light_pos[light_idx];
        float3 l_int = light_intensity[light_idx];

        var dp_kd = diffPair(kd);
        var dp_pos = diffPair(pos);
        var dp_nrm = diffPair(nrm);
        var dp_lint = diffPair(l_int);

        __bwd_diff(lambertBSDF)(dp_kd, dp_pos, dp_nrm, dp_lint, l_pos, dout);

        kd_accum += dp_kd.d;
        pos_accum += dp_pos.d;
        nrm_accum += dp_nrm.d;

        // float3 old_lint_grad;
        // lint_grad.InterlockedAdd(idx.y, dp_lint.d, old_lint_grad);
    }

    kd_grad[idxr] = clamp(select(isfinite(kd_accum), kd_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    pos_grad[idxr] = clamp(select(isfinite(pos_accum), pos_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    nrm_grad[idxr] = clamp(select(isfinite(nrm_accum), nrm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
}

[CudaKernel]
void specular_bn_fwd_kernel(TensorView<float3> kd,              // B x N x 3
                            TensorView<float3> arm,             // B x N x 3
                            TensorView<float3> pos,             // B x N x 3
                            TensorView<float3> nrm,             // B x N x 3
                            TensorView<float3> view_pos,        // B x 3
                            TensorView<float3> light_pos,       // B x L x 3
                            TensorView<float3> light_intensity, // B x L x 1
                            const float min_roughness,
                            TensorView<float3> output)
{
    let idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(1) || idx.y > output.size(0))
    {
        return;
    }

    let idxr = uint2(idx.y, idx.x);
    float3 v_pos = view_pos[idx.y];

    let n_lights = light_pos.size(1);
    output[idxr] = float3(0.0f);

    for (uint i = 0; i < n_lights; i++)
    {
        float3 l_pos = light_pos[idx.y, i];
        float3 l_int = light_intensity[idx.y, i];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        let res = specularBSDF(kd[idxr], arm[idxr], pos[idxr], nrm[idxr], l_int, v_pos, l_pos, min_roughness);
        output[idxr] += res;
    }
}

[CudaKernel]
void specular_bn_bwd_kernel(TensorView<float3> kd,
                            TensorView<float3> arm,
                            TensorView<float3> pos,
                            TensorView<float3> nrm,
                            TensorView<float3> view_pos,
                            TensorView<float3> light_pos,
                            TensorView<float3> light_intensity,
                            const float min_roughness,
                            TensorView<float3> kd_grad,
                            TensorView<float3> arm_grad,
                            TensorView<float3> pos_grad,
                            TensorView<float3> nrm_grad,
                            TensorView<float3> lint_grad,
                            TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(1) || idx.y > grad_out.size(0))
    {
        return;
    }

    uint2 idxr = uint2(idx.y, idx.x);

    float3 v_pos = view_pos[idx.y];

    let n_lights = light_pos.size(1);
    float3 kd_accum = float3(0.0f);
    float3 arm_accum = float3(0.0f);
    float3 pos_accum = float3(0.0f);
    float3 nrm_accum = float3(0.0f);
    // float lint_accum = 0.0f;
    float lint_accum = kd.dims();

    let kd = kd[idxr];
    let arm = arm[idxr];
    let pos = pos[idxr];
    let nrm = nrm[idxr];
    let dout = grad_out[idxr];

    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 light_idx = uint2(idx.y, i);
        float3 l_pos = light_pos[idx.y, i];
        float3 l_int = light_intensity[idx.y, i];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        var dp_kd = diffPair(kd);
        var dp_arm = diffPair(arm);
        var dp_pos = diffPair(pos);
        var dp_nrm = diffPair(nrm);
        var dp_lint = diffPair(l_int);

        __bwd_diff(specularBSDF)(dp_kd, dp_arm, dp_pos, dp_nrm, dp_lint, v_pos, l_pos, min_roughness, dout);

        kd_accum += dp_kd.d;
        arm_accum += dp_arm.d;
        pos_accum += dp_pos.d;
        nrm_accum += dp_nrm.d;

        // float3 old_lint_grad;
        // lint_grad.InterlockedAdd(light_idx, dp_lint.d, old_lint_grad);
    }

    kd_grad[idxr] = clamp(select(isfinite(kd_accum), kd_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    arm_grad[idxr] = clamp(select(isfinite(arm_accum), arm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    pos_grad[idxr] = clamp(select(isfinite(pos_accum), pos_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    nrm_grad[idxr] = clamp(select(isfinite(nrm_accum), nrm_accum, float3(0.0f)), float3(-1.0f), float3(1.0f));
    lint_grad[idx.y] = clamp(select(isfinite(lint_accum), lint_accum, 0.0f), -1.0f, 1.0f);
}

/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////
[TorchEntryPoint]
TorchTensor<float3> pbr_fwd(TorchTensor<float3> kd,
                               TorchTensor<float3> arm,
                               TorchTensor<float3> pos,
                               TorchTensor<float3> nrm,
                               TorchTensor<float3> view_pos,
                               TorchTensor<float3> light_pos,
                               TorchTensor<float3> light_intensity,
                               const float min_roughness,
                               const uint dims)
{  
    var result = TorchTensor<float3>.emptyLike(kd);

    if (dims == 3)
    {
        // B x N x 3
        uint3 dims = uint3(result.size(1), result.size(0), 1);
        let blockSize = uint3(64, 1, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);
        __dispatch_kernel(pbr_bn_fwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness, result);
    } 
    else if (dims == 4)
    {
        // B x H x W x 3
        uint3 dims = uint3(result.size(2), result.size(1), result.size(0));
        let blockSize = uint3(8, 8, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);
        __dispatch_kernel(pbr_nhwc_fwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness, result);
    }

    return result;
}

struct PbrBwdReturnType
{
    TorchTensor<float3> kd_grad;
    TorchTensor<float3> arm_grad;
    TorchTensor<float3> pos_grad;
    TorchTensor<float3> nrm_grad;
    TorchTensor<float3> light_int_grad;
}

[TorchEntryPoint]
PbrBwdReturnType pbr_bwd(TorchTensor<float3> kd,
                            TorchTensor<float3> arm,
                            TorchTensor<float3> pos,
                            TorchTensor<float3> nrm,
                            TorchTensor<float3> view_pos,
                            TorchTensor<float3> light_pos,
                            TorchTensor<float3> light_intensity,
                            const float min_roughness,
                            const int dims,
                            TorchTensor<float3> grad_out)
{
    PbrBwdReturnType rt;

    rt.kd_grad = TorchTensor<float3>.zerosLike(kd);
    rt.arm_grad = TorchTensor<float3>.zerosLike(arm);
    rt.pos_grad = TorchTensor<float3>.zerosLike(pos);
    rt.nrm_grad = TorchTensor<float3>.zerosLike(nrm);
    rt.light_int_grad = TorchTensor<float3>.zerosLike(light_intensity);

    if (dims== 3)
    {
        uint3 dims = uint3(grad_out.size(1), grad_out.size(0), 1);
        let blockSize = uint3(64, 1, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);
        __dispatch_kernel(pbr_bn_bwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness,
                                                                rt.kd_grad, rt.arm_grad, rt.pos_grad, rt.nrm_grad, rt.light_int_grad, grad_out);
    }
    else if (dims == 4)
    {
        uint3 dims = uint3(grad_out.size(2), grad_out.size(1), grad_out.size(0));
        let blockSize = uint3(8, 8, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);
        __dispatch_kernel(pbr_nhwc_bwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness,
                                                                    rt.kd_grad, rt.arm_grad, rt.pos_grad, rt.nrm_grad, rt.light_int_grad, grad_out);
    }
    
    return rt;
}

[TorchEntryPoint]
TorchTensor<float3> lambert_fwd(TorchTensor<float3> kd,
                                   TorchTensor<float3> pos,
                                   TorchTensor<float3> nrm,
                                   TorchTensor<float3> light_pos,
                                   TorchTensor<float3> light_intensity,
                                   const int dims)
{
    var result = TorchTensor<float3>.emptyLike(kd);
    
    if (dims == 3)
    {
        uint3 dims = uint3(result.size(1), result.size(0), 1);
        let blockSize = uint3(64, 1, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);
        __dispatch_kernel(lambert_bn_fwd_kernel, blockCount, blockSize)(kd, pos, nrm, light_pos, light_intensity, result);
    }
    else if (dims == 4)
    {
        uint3 dims = uint3(result.size(2), result.size(1), result.size(0));
        let blockSize = uint3(8, 8, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);
        __dispatch_kernel(lambert_nhwc_fwd_kernel, blockCount, blockSize)(kd, pos, nrm, light_pos, light_intensity, result);
    }

    return result;
}

struct LambertBwdReturnType
{
    TorchTensor<float3> kd_grad;
    TorchTensor<float3> pos_grad;
    TorchTensor<float3> nrm_grad;
    TorchTensor<float3> light_int_grad;
}

[TorchEntryPoint]
LambertBwdReturnType lambert_bwd(TorchTensor<float3> kd,
                                    TorchTensor<float3> pos,
                                    TorchTensor<float3> nrm,
                                    TorchTensor<float3> light_pos,
                                    TorchTensor<float3> light_intensity,
                                    const int dims,
                                    TorchTensor<float3> grad_out)
{
    LambertBwdReturnType rt;

    rt.kd_grad = TorchTensor<float3>.zerosLike(kd);
    rt.pos_grad = TorchTensor<float3>.zerosLike(pos);
    rt.nrm_grad = TorchTensor<float3>.zerosLike(nrm);
    rt.light_int_grad = TorchTensor<float3>.zerosLike(light_intensity);

    if (dims == 3)
    {
        uint3 dims = uint3(grad_out.size(1), grad_out.size(0), 1);
        let blockSize = uint3(64, 1, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);

        __dispatch_kernel(lambert_bn_bwd_kernel, blockCount, blockSize)(kd, pos, nrm, light_pos, light_intensity,
                                                                        rt.kd_grad, rt.pos_grad, rt.nrm_grad, rt.light_int_grad, grad_out);
    }
    else if (dims == 4)
    {
        uint3 dims = uint3(grad_out.size(2), grad_out.size(1), grad_out.size(0));
        let blockSize = uint3(8, 8, 1);
        let blockCount = getLaunchGridSize(blockSize, dims);

        __dispatch_kernel(lambert_nhwc_bwd_kernel, blockCount, blockSize)(kd, pos, nrm, light_pos, light_intensity,
                                                                        rt.kd_grad, rt.pos_grad, rt.nrm_grad, rt.light_int_grad, grad_out);
    }

    return rt;
}

[TorchEntryPoint]
TorchTensor<float3> specular_bn_fwd(TorchTensor<float3> kd,
                                    TorchTensor<float3> arm,
                                    TorchTensor<float3> pos,
                                    TorchTensor<float3> nrm,
                                    TorchTensor<float3> view_pos,
                                    TorchTensor<float3> light_pos,
                                    TorchTensor<float3> light_intensity,
                                    const float min_roughness)
{
    // B x N x 3
    var result = TorchTensor<float3>.emptyLike(kd);
    uint3 dims = uint3(result.size(1), result.size(0), 1);
    let blockSize = uint3(64, 1, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(specular_bn_fwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness, result);

    return result;
}

struct SpecularBwdReturnType
{
    TorchTensor<float3> kd_grad;
    TorchTensor<float3> arm_grad;
    TorchTensor<float3> pos_grad;
    TorchTensor<float3> nrm_grad;
    TorchTensor<float3> light_int_grad;
}

[TorchEntryPoint]
SpecularBwdReturnType specular_bn_bwd(TorchTensor<float3> kd,
                                      TorchTensor<float3> arm,
                                      TorchTensor<float3> pos,
                                      TorchTensor<float3> nrm,
                                      TorchTensor<float3> view_pos,
                                      TorchTensor<float3> light_pos,
                                      TorchTensor<float3> light_intensity,
                                      const float min_roughness,
                                      TorchTensor<float3> grad_out)
{

    // B x N x 3
    SpecularBwdReturnType rt;

    rt.kd_grad = TorchTensor<float3>.zerosLike(kd);
    rt.arm_grad = TorchTensor<float3>.zerosLike(arm);
    rt.pos_grad = TorchTensor<float3>.zerosLike(pos);
    rt.nrm_grad = TorchTensor<float3>.zerosLike(nrm);
    rt.light_int_grad = TorchTensor<float3>.zerosLike(light_intensity);

    uint3 dims = uint3(grad_out.size(1), grad_out.size(0), 1);
    let blockSize = uint3(64, 1, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(specular_bn_bwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness,
                                                                     rt.kd_grad, rt.arm_grad, rt.pos_grad, rt.nrm_grad, rt.light_int_grad, grad_out);

    return rt;
}
