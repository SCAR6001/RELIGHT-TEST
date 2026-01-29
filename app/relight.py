import torch

def relight(normals, diffuse, specular, light_dir, intensity):
    light_dir = torch.tensor(light_dir).view(1, 3, 1, 1).to(normals.device)
    light_dir = torch.nn.functional.normalize(light_dir, dim=1)

    ndotl = torch.clamp((normals * light_dir).sum(1, keepdim=True), 0, 1)

    diffuse_term = diffuse * ndotl
    specular_term = specular * (ndotl ** 25)

    return torch.clamp((diffuse_term + specular_term) * intensity, 0, 1)
