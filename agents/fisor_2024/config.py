FISOR_config = {
    "algo": "ql",
    "eval_freq": int(1e5),
    "max_timestep": int(1e6),
    "checkpoint_start": int(9e5),
    "checkpoint_every": int(1e5),
    "gamma": 0.99,
    "tau": 0.005,
    "eta": 1.0,
    "lr_decay": True,
    "max_q_backup": False,
    "step_start_ema": 1000,
    "ema_decay": 0.995,
    "update_ema_every": 1,
    "flow_steps": 10,
    "beta_schedule": "vp",
    "ms": "offline",
    "gn": 5.0,
    "expectile_temp": 0.9,
    "cost_expectile_temp": 0.9,
    "cost_adv_ub": 100.0,
    "cost_temperature": 6.0,
    "reward_temperature": 6.0,
    "reward_scale": 1,
    "cost_scale": 1,
    "num_q": 2,
    "num_v": 2,
    "num_qc": 2,
    "num_vc": 2,
    "episode_length": 200,
    "chunking_length": 5,
    "guided_step": 10,
    "ood_noise": 3e-2,
    "unsafe_penalty": -1.0,
    "safety_threshold": 1.0,
    "safe_portion": 0.7,
    "cost_limit": 10.0,
    "guidance_scale": 3.0,
    "lr": 3e-4,
    "alpha": 0.2,
    "batch_size": 1024,
    "hidden_dim": 512,
    "reward_tune": "no",
    "actor_num_stack": 3,
    "critic_net": "mlp",
    "print_more_info": False,
    "normalize": False,
}


ENV_CONFIG_OVERRIDES = (
    ("goal", {
        "num_q": 2, "num_v": 2, "num_qc": 2, "num_vc": 2,
        "reward_scale": 200, "actor_num_stack": 3, "episode_length": 1000,
        "safe_portion": 0.9, "unsafe_penalty": -10.0, "chunking_length": 5,
    }),
    ("push", {
        "num_q": 2, "num_v": 1, "num_qc": 2, "num_vc": 1,
        "reward_scale": 200, "actor_num_stack": 3, "episode_length": 1000,
        "chunking_length": 5, "safe_portion": 0.95, "unsafe_penalty": -1.0,
    }),
    ("button", {
        "num_q": 2, "num_v": 1, "num_qc": 2, "num_vc": 1,
        "reward_scale": 200, "cost_scale": 1, "actor_num_stack": 3,
        "episode_length": 1000, "safe_portion": 0.5, "unsafe_penalty": -10.0,
    }),
    ("swimmervel", {"safe_portion": 0.7}),
    ("antvel", {"safe_portion": 0.99, "chunking_length": 1}),
    ("halfcheetah", {"safe_portion": 0.99, "chunking_length": 1}),
    ("ballrun", {
        "guided_step": 10, "chunking_length": 5, "reward_temperature": 1.0,
        "cost_temperature": 2.0, "cost_limit": 5.0, "safe_portion": 0.7,
        "unsafe_penalty": -1.0,
    }),
    ("ballcircle", {"guided_step": 10, "cost_limit": 5.0}),
    ("dronecircle", {
        "episode_length": 300, "guided_step": 10, "chunking_length": 3,
        "cost_limit": 5.0, "safe_portion": 0.6, "guidance_scale": 2.0,
    }),
    ("dronerun", {
        "episode_length": 300, "guided_step": 10, "chunking_length": 1,
        "cost_limit": 5.0, "safe_portion": 0.6,
    }),
    ("antcircle", {
        "episode_length": 500, "cost_limit": 5.0, "chunking_length": 5,
        "safe_portion": 0.7,
    }),
    ("antrun", {"episode_length": 500, "cost_limit": 5.0, "safe_portion": 0.7}),
    ("carcircle", {"episode_length": 500, "cost_limit": 5.0}),
    ("carrun", {"episode_length": 500, "cost_limit": 5.0, "safe_portion": 0.9}),
    ("easy", {"reward_scale": 100, "cost_limit": 5, "safe_portion": 0.6}),
    ("medium", {"reward_scale": 100, "cost_limit": 5, "safe_portion": 0.7}),
    ("hard", {"reward_scale": 100, "cost_limit": 5, "safe_portion": 0.5}),
)


def update_config(env_name, config):
    env_name_lower = env_name.lower()
    for pattern, overrides in ENV_CONFIG_OVERRIDES:
        if pattern in env_name_lower:
            config.update(overrides)
            break
    return config
