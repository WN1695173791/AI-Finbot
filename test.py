import models
import utils

if __name__ == "__main__":    
    model = utils.misc.test(models.SDENet)
    model.custom_compile()
    num_params = utils.count_parameters(model)
    print(f"Model: {model.custom_name}; #parameters: {num_params}")
    X = utils.pipelines.get_random(shape=(3,20))
    print(model.prediction(data=X))
