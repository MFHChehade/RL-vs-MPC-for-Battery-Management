from energy_management_env import EnergyManagementEnv

def energy_management_env_creator(SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC=None):
    return EnergyManagementEnv(SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC)
