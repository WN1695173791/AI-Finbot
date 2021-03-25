import models
import utils


def test():
    model = models.SDENet()
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":    
    model = test()
    num_params = count_parameters(model)
    print(f"SDENet parameters: {num_params}")
    X = utils.pipelines.random(shape=(3,20))
    target_scale = utils.pipelines.random(shape=(1,))
    print(model.prediction(data=X, target_scale=target_scale))
    
    # print(X)
    # print(model(X, training_diffusion = True))