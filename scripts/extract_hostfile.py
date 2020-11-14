import os
import json
import torch

if os.path.exists("/home/hostfile.json"):
    with open("/home/hostfile.json") as file:
        hosts = json.load(file)
    with open("/workspace/hostfile", "w") as output:
        for host in hosts:
            if host["role"] == "master":
                output.write(f"root@{host['ip']} slots=1\n")
else:
    gpu_count = torch.cuda.device_count()
    with open("/workspace/hostfile", "w") as output:
        output.write("root@127.0.0.1 slots=8\n")
