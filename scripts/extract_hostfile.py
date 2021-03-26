import os
import json
import torch

if os.path.exists("/home/hostfile.json"):
    with open("/home/hostfile.json") as file:
        hosts = json.load(file)
    master_hosts, slave_hosts = [], []
    for host_info in hosts:
        if host_info["role"] == "master":
            master_hosts.append(host_info["ip"])
        else:
            slave_hosts.append(host_info["ip"])
    with open("/workspace/hostfile", "w") as output:
        for host in master_hosts:
            output.write(f"root@{host} slots=8\n")
        for host in slave_hosts:
            output.write(f"root@{host} slots=8\n")
    with open("/workspace/pssh_hosts", "w") as output:
        for host in master_hosts:
            output.write(f"root@{host}\n")
        for host in slave_hosts:
            output.write(f"root@{host}\n")
else:
    gpu_count = torch.cuda.device_count()
    with open("/workspace/hostfile", "w") as output:
        output.write("root@127.0.0.1 slots=8\n")
