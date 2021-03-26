from models import SDENet
import utils

LATEST_TRAINED_MODEL_PATH = ""

class Bot():
    global LATEST_TRAINED_MODEL_PATH
    def __init__(self, name):
        self.name = name
        self.model = SDENet()   
        self.model.custom_compile()
        self.model.load_model(path=LATEST_TRAINED_MODEL_PATH)


    def hello(self):
        print(f"{self.name}: ~(^_^)~")
        print(f"Function approximator: {self.model.custom_name}")
        print(f"Trainable parameters: {utils.misc.count_parameters(self.model)}")
        


    
