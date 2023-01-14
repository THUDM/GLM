import os
import sys
from typing import Dict
from omegaconf import OmegaConf,DictConfig
import wandb
import loguru

class Logger:
    def __init__(self, cfg:DictConfig):
        self.cfg = cfg
        self.logger = loguru.logger
        self.logger.remove()
        self.logger.add(f"{self.cfg.jobname}.log")

    def log_rank(self,data:Dict):
        log_str = "\t".join(f"{key}  {data[key]}" for key in data)
        self.logger.info(log_str)

    def log_master(self,data:Dict,if_wandb:bool=True):
        """
        only called in master process
        """
        log_str = "\t".join(f"{key}  {data[key]}" for key in data)
        self.logger.info(log_str)
        if self.cfg.debug != True and if_wandb:
            wandb.log(data)

    def login(self):
        """
        only login once in distributed training
        """
        self.logger.add(sys.stderr)
        # if you do not want to use wandb, you can set self.cfg.debug = True
        if self.cfg.debug != True:
            # you can use export VAR=VALUE to set environment variables before running the script
            wandb_var_l = ["WANDB_API_KEY","WANDB_ENTITY","WANDB_PROJECT"]
            for wandb_var in wandb_var_l:
                if os.environ.get(wandb_var) is None:
                    os.environ[wandb_var] = input(f"Please input your {wandb_var}:")
            wandb.login(key=os.environ['WANDB_API_KEY'])
            wandb.init(project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_ENTITY'], 
                name=self.cfg.jobname,config=OmegaConf.to_container(self.cfg,resolve=True))
            # wandb.config.update(self.cfg)

        self.logger.info("\nConfigs are:\n" + OmegaConf.to_yaml(self.cfg))

if __name__ == "__main__":
    mylogger = Logger()
    mylogger.log({"loss":0.1,"acc":0.9})
