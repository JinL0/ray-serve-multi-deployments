# ray-serve-multi-deployments
Experiments of deploying multiple ray serve endpoints 
```
mnist - image classification 
gpt2 - nlp 
```
# Local ray cluster deployment 

# Kuberay 
Fellow the kuberay repo to install `kuberay-operator`

## AWS Kuberay Set Up
### Bring up eks cluster
update the `publicKeyName` with your key. 
Then run the following command `eksctl create cluster -f ray_cluster.yaml`. <br/>
This may take a while to bring up the eks cluster. 

### Bring up the ray cluster 
Run the following command to bring up the ray cluster and port forwarding the dashboard. 

```
kubectl apply -f ray_serve.yaml
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl port-forward $HEAD_POD --address 0.0.0.0 8265:8265
```

To test the endpoint deployments, port forwarding the 8000 by `kubectl port-forward $HEAD_POD --address 0.0.0.0 8265:8265`. Then open your terminal and type `python3`, run:
```
>>> import numpy as np
>>> import requests
>>> result = requests.post("http://localhost:8000/mnist", json= {"images": np.random.rand(16, 1, 28, 28).tolist()}).json()
>>> result
{'result': ['Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag', 'Bag']}
>>> result = requests.post("http://localhost:8000/gpt2", json= {"message": "how are you"}).json()
>>> result
{'result': ' feeling that feeling now? That he did nothing wrong?"\n\n"Well, I had some things coming up on the morning. He had something to do with the rest of the battle."\n\n"Hmm, good news, you'}
``` 