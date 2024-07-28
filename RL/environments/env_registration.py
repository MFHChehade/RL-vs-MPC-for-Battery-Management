from gym.envs.registration import register

def register_env(env_id, entry_point, kwargs):
    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=kwargs,
    )

def environment_creator(environment_class, **kwargs):
    return environment_class(**kwargs)
