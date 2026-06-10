from .PFESA import TokenPFESA as OriginalTokenPFESA
from .PFESA_residual import TokenPFESA as ResidualTokenPFESA


PFESA_VARIANTS = {
    "original": OriginalTokenPFESA,
    "residual": ResidualTokenPFESA,
}


def available_pfesa_variants():
    return tuple(PFESA_VARIANTS.keys())


def build_token_pfesa(variant="original", base_ratio=0.1, **kwargs):
    if variant not in PFESA_VARIANTS:
        choices = ", ".join(available_pfesa_variants())
        raise ValueError(f"Unsupported pfesa_variant: {variant}. Available variants: {choices}")
    return PFESA_VARIANTS[variant](base_ratio=base_ratio, **kwargs)
