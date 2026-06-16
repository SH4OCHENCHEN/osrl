import torch
import torch.nn as nn


DEFAULT_COMPILE_MODULES = (
    "actor",
    "actor_low",
    "critic",
    "critic_target",
    "cost_critic",
    "cost_critic_target",
    "value",
    "cost_value",
    "ema_model",
)


def resolve_compile_enabled(setting, device):
    if setting == "on":
        return True
    if setting == "off":
        return False
    return getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"


def parse_compile_dynamic(setting):
    if setting == "auto":
        return None
    return setting == "true"


def configure_matmul_precision(precision):
    if not precision:
        return None
    if not hasattr(torch, "set_float32_matmul_precision"):
        print("torch.set_float32_matmul_precision is unavailable in this PyTorch version; skipping.")
        return None
    torch.set_float32_matmul_precision(precision)
    return precision


def compile_policy_modules(
    policy,
    enabled=False,
    mode="default",
    backend="inductor",
    fullgraph=False,
    dynamic=None,
    module_names=DEFAULT_COMPILE_MODULES,
):
    if not enabled:
        return []
    if not hasattr(torch, "compile"):
        print("torch.compile is unavailable in this PyTorch version; skipping compilation.")
        return []
    if not hasattr(nn.Module, "compile"):
        print("torch.nn.Module.compile is unavailable in this PyTorch version; skipping to preserve checkpoint compatibility.")
        return []

    compiled = []
    for name in module_names:
        module = getattr(policy, name, None)
        if not isinstance(module, nn.Module):
            continue

        try:
            module.compile(
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                mode=mode,
            )
            compiled.append(name)
        except Exception as exc:
            print(f"Skipping torch.compile for policy.{name}: {exc}")

    if compiled:
        print(
            "torch.compile enabled for "
            f"{', '.join(compiled)} (backend={backend}, mode={mode}, fullgraph={fullgraph}, dynamic={dynamic})"
        )
    else:
        print("torch.compile was requested, but no policy modules were compiled.")
    return compiled
