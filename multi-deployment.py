import os
import ray
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ray import serve
from typing import Dict
from transformers import pipeline
from torchvision import transforms
from starlette.requests import Request
from ray.serve.drivers import DAGDriver
from ray.serve.deployment_graph import InputNode

transform = transforms.Compose([
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize the image with mean and std of 0.5 each
])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

@serve.deployment(name="mnist", 
                  autoscaling_config={"min_replicas": 1, "max_replicas": 2},
                  ray_actor_options={"runtime_env": {"pip": ["torch", "torchvision"]}})
class MnistDeployment:
    def __init__(self, msg: str, model_path: str):
        # Initialize model state: could be very large neural net weights.
        self._msg = msg
        self.model = Net()
        """
        if not os.path.exists(model_path):
            download_file_from_s3(model_path)
        """
        current_folder = os.getcwd()
        model_path = os.path.join(current_folder, model_path)
        self.model.load_state_dict(torch.load(model_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    async def __call__(self, request: Request) -> Dict:
        images = (await request.json())["images"]
        # print(f"jin debugging {type(images)}")
        image_to_tensor = torch.tensor(images)
        if image_to_tensor.shape == (28, 28):
            image_to_tensor = transform(np.asarray(images, dtype=np.float32))
            image_to_tensor = image_to_tensor.unsqueeze(0)
      
        image_to_tensor= image_to_tensor.to(self.device)
        predictions = self.model(image_to_tensor)
        predicted_indices = torch.argmax(predictions, dim=1)

        # Converting index to corresponding string label for Fashion MNIST
        class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        predicted_labels = [class_labels[i] for i in predicted_indices]
        
        return {"result": predicted_labels}

@serve.deployment(name="gpt2",
                  autoscaling_config={"min_replicas": 1, "max_replicas": 4},
                  ray_actor_options={"runtime_env": {"pip": ["transformers"]}})
class Gpt2Deployment:
    def __init__(self):
        # Initialize model state: could be very large neural net weights.
        self.pipeline = pipeline(model="gpt2")

    async def __call__(self, request: Request) -> Dict:
        print(request)
        message = (await request.json())["message"]
        result = self.pipeline(message)[0]["generated_text"][len(message):]
        return {"result": result}
    
with InputNode() as name:
    mnist_app = MnistDeployment.bind(msg="mnist_deployment", model_path="model.pth")
    gpt2_app = Gpt2Deployment.bind()

mnist = DAGDriver.options(route_prefix="/mnist").bind(mnist_app)
gpt2 =  DAGDriver.options(route_prefix="/gpt2").bind(gpt2_app)
