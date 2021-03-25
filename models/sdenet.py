import math
import torch
from .model_configs import SDEnet_configs

drift_depth = SDEnet_configs.drift_depth
latent_nodes = SDEnet_configs.latent_nodes
diffusion_depth = SDEnet_configs.diffusion_depth
diffusion_nodes = SDEnet_configs.diffusion_nodes
in_nodes = SDEnet_configs.in_nodes

class Drift(torch.nn.Module):
    global latent_nodes
    def __init__(self):
        super().__init__()        
        self.fc = torch.nn.Linear(latent_nodes, latent_nodes)
        self.relu = torch.nn.ReLU(inplace=True)
        

    def forward(self, x):
        out = self.relu(self.fc(x))
        return out

class Diffusion(torch.nn.Module):
    global diffusion_depth, latent_nodes, diffusion_nodes
    def __init__(self):
        super().__init__()
        # self.relu = torch.nn.ReLU(inplace=True)
        self.diffusion_transform = torch.nn.Linear(latent_nodes, diffusion_nodes)
        self.fc = torch.nn.Linear(diffusion_nodes,diffusion_nodes)
        self.diffusion_node = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.Linear(diffusion_nodes, 1))
        self.diffusion_depth = diffusion_depth
        self.deltat = 1./self.diffusion_depth
    
    def forward(self, x):
        out = self.diffusion_transform(x)
        for i in range(self.diffusion_depth):
            t = 1.*float(i)/self.diffusion_depth
            out = out + self.fc(out)*self.deltat

        diffusion_out = torch.sigmoid(self.diffusion_node(out))
        return diffusion_out

class SDENet(torch.nn.Module):
    global drift_depth, latent_nodes, in_nodes
    def __init__(self):
        super().__init__()
        self.drift_depth = drift_depth
        self.pre_transform = torch.nn.Linear(in_nodes, latent_nodes)
        self.drift = Drift()
        self.diffusion = Diffusion()
        self.output_layer = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.Linear(latent_nodes, 2))
        self.deltat = 1./self.drift_depth
        self.sigma = 0.5

    def forward(self, x, training_diffusion=False):
        out = self.pre_transform(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma*self.diffusion(out)
            for i in range(self.drift_depth):
                t = 1.*(float(i))/self.drift_depth
                out = out+self.drift(out)*self.deltat + diffusion_term*math.sqrt(self.deltat)*torch.randn_like(out).to(x)
            
            final_out = self.output_layer(out)
            mean = final_out[:,0]
            sigma = torch.nn.functional.softplus(final_out[:,1])
            return mean, sigma

        else:
            final_out = self.diffusion(out.detach())
            return final_out

    def custom_compile(self):
        pass

    def train_step(self):
        pass

    def evaluation_step(self):
        pass
    





