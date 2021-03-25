import math
import torch
from .model_configs import SDEnet_configs
import utils

drift_depth = SDEnet_configs.drift_depth
latent_nodes = SDEnet_configs.latent_nodes
diffusion_depth = SDEnet_configs.diffusion_depth
diffusion_nodes = SDEnet_configs.diffusion_nodes
in_nodes = SDEnet_configs.in_nodes

lr_1 = SDEnet_configs.lr_1
lr_2 = SDEnet_configs.lr_2
momentum_1 = SDEnet_configs.momentum_1
momentum_2 = SDEnet_configs.momentum_2
weight_decay = SDEnet_configs.weight_decay

noise_scale = SDEnet_configs.noise_scale

eval_iters = SDEnet_configs.eval_iters
pred_iters = SDEnet_configs.pred_iters

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
    global drift_depth, latent_nodes, in_nodes, lr_1, lr_2, momentum_1, momentum_2, weight_decay, noise_scale, eval_iters, predict_iters
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
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.momentum_1 = momentum_1
        self.momentum_2 = momentum_2
        self.real_label = 0
        self.fake_label = 1
        self.diffusion_loss = torch.nn.BCELoss()
        self.drift_loss = lambda y, mean, sigma: torch.mean(torch.log(sigma**2)+(y-mean)**2/(sigma**2)) 
        self.optim_drift = torch.optim.SGD([{'params': self.pre_transform.parameters()}, {'params': self.drift.parameters()}, {'params': self.output_layer.parameters()}], lr=self.lr_1, momentum=self.momentum_1, weight_decay=self.weight_decay)
        self.optim_diffusion = torch.optim.SGD([{'params': net.diffusion.parameters()}], lr=self.lr_2, momentum=self.momentum_2, weight_decay=self.weight_decay)
        self.noise_scale = noise_scale
        self.test_iters = test_iters


    def train_step(self, **kwargs):
        self.train()
        inputs, targets = kwargs['batch_tuple']        
        if kwargs['epoch'] == 0:
            self.sigma = 0.1
        if kwargs['epoch'] == 30:
            self.sigma = 0.5
        
        self.optim_drift.zero_grad()
        mean, sigma = self.forward(inputs)
        loss = self.drift_loss(targets, mean, sigma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.)
        self.optim_drift.step()

        label = torch.full((len(inputs), 1), real_label)
        optim_diffusion.zero_grad()
        predict_in = self.forward(inputs, training_diffusion=True)
        loss_in = self.diffusion_loss(predict_in, label)
        loss_in.backward()

        label.fill_(fake_label)
        inputs_out = self.noise_scale*torch.randn(len(inputs), in_nodes)+inputs
        predict_out = self.forward(inputs_out, training_diffusion=True)
        loss_out = self.diffusion_loss(predict_out, label)
        loss_out.backward()
        self.optim_diffusion.step()
        
        return loss.item(), loss_in.item(), loss_out.item()


    def evaluation_step(self, **kwargs):
        self.eval()
        inputs, targets = kwargs['batch_tuple']
        current_mean = 0
        with torch.no_grad():
            for i in range(self.eval_iters):
                mean, sigma = self.forward(inputs)
                current_mean += mean
            current_mean = current_mean/self.eval_iters
            current_mean = current_mean*kwargs['target_scale']
            targets = targets*kwargs['target_scale']
            test_loss = kwargs['eval_func'](targets, current_mean)
        
        return test_loss.item()

    def prediction(self, **kwargs):
        self.eval()
        inputs, targets = kwargs['batch_tuple']
        current_mean = 0
        current_sigma = 0
        with torch.no_grad():
            for i in range(self.pred_iters):
                mean, sigma = self.forward(inputs)
                mean = mean*kwargs['target_scale']
                current_mean += mean
                current_sigma += sigma
                if i == 0:
                    sigmas = torch.unsqueeze(sigma, 1)
                    means = torch.unsqueeze(mean, 1)
                else:
                    means = torch.cat((means, torch.unsqueeze(mean, 1)), dim=1)
                    sigmas = torch.cat((sigmas, torch.unsqueeze(sigma, 1)), dim=1)
            current_mean = current_mean/self.pred_iters
            current_sigma = current_sigma/self.pred_iters
            var_means = means.std(dim=1)
            var_sigmas = sigmas.std(dim=1)
        
        return current_mean.item(), var_means.item(), current_sigma.item(), var_sigmas.item()

    def save_model(self, **kwargs):
        pass

    def load_model(self, **kwargs):
        pass





    





