import warnings
import hydra
from omegaconf import DictConfig

from logger import Logger
from data import PGDataset,PGDataCollator,PCDataset,PCDataCollator
from model import PGModel, PCModel

import torch
from torch import optim

from typing import Tuple,List
from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine,Events
from ignite import metrics
from lrscheduler import build_lrscheduler
from ignite.handlers import Checkpoint, global_step_from_engine

def main_engine(local_rank: int, cfg: DictConfig,**kwargs):
    # ignore warnings
    if not cfg.debug:
        warnings.filterwarnings("ignore")
    # Setup logger
    logger = Logger(cfg)
    if idist.get_rank() == 0:
        logger.login()
    logger.log_rank({"rank":idist.get_rank(),"local_rank":local_rank,"world_size":idist.get_world_size()})

    # Setup model
    if cfg.task == "pg":
        model = PGModel(cfg.model)
    elif cfg.task == "pc":
        model = PCModel(cfg.model)
    else:
        raise NotImplementedError(f"Task {cfg.task} is not supported")
    model = idist.auto_model(model)
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(),
        lr=cfg.optimizer.lr,betas=(cfg.optimizer.beta1,cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.wd,
    )
    optimizer = idist.auto_optim(optimizer)

    def train_step(engine:Engine, batch:Tensor) -> Tensor:
        model.train()
        batch.to(idist.device())
        loss = model(batch).loss
        loss = loss / cfg.trainer.accumulate_steps # accumulate gradients
        loss.backward()
        if engine.state.iteration % cfg.trainer.accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def pg_test_evaluate_step(engine:Engine, batch:Tensor) -> Tuple[List[str],List[str]]:
        model.eval()
        prompts = batch.pop("prompts")
        labels = batch.pop("labels")
        with torch.no_grad():
            batch.to(idist.device())
            if idist.get_world_size() > 1:
                res = model.module.generate(batch)
            else:
                res = model.generate(batch)
            return res, labels

    def pc_test_evaluate_step(engine:Engine, batch:Tensor) -> Tuple[List[str],List[str]]:
        model.eval()
        with torch.no_grad():
            batch.to(idist.device())
            labels = batch.pop("labels")
            res = model(batch).logits
            return res, labels

    # setup engines
    trainer = Engine(train_step)
    if cfg.task == "pg":
        test_evaluate_step = pg_test_evaluate_step
    elif cfg.task == "pc":
        test_evaluate_step = pc_test_evaluate_step
    else:
        raise NotImplementedError(f"Task {cfg.task} is not supported")
    test_evaluator = Engine(test_evaluate_step)

    # metrics
    train_loss = metrics.Average()
    train_loss.attach(trainer,"train_loss")
    if cfg.task == "pg":
        def rouge_output_transform(output:Tuple[List[str],List[str]]) -> Tuple[List[List[str]],List[List[List[str]]]]:
            res,labels = output
            res = [item.split() for item in res]
            labels = [[item.split()] for item in labels]
            return res,labels
        test_rouge = metrics.Rouge(variants=['L',1,2],output_transform=rouge_output_transform)
        test_rouge.attach(test_evaluator,"test_rouge")
    elif cfg.task == "pc":
        test_acc = metrics.Accuracy()
        test_acc.attach(test_evaluator,"test_acc")
    else:
        raise NotImplementedError(f"Task {cfg.task} is not supported")
    
    @trainer.on(Events.COMPLETED)
    def log_final_results(engine:Engine):
        log_data = { "epoch":engine.state.epoch, }
        # test evaluation of rouge is too slow, so we only evaluate it at the end
        test_evaluator.run(test_dataloader)
        test_metrics = test_evaluator.state.metrics
        if cfg.task == "pg":
            log_data["test_rouge"] = test_metrics["test_rouge"]
        elif cfg.task == "pc":
            log_data["test_acc"] = test_metrics["test_acc"]
        else:
            raise NotImplementedError(f"Task {cfg.task} is not supported")
        # log test evaluation
        logger.log_rank(log_data)
        if idist.get_rank() == 0:
            logger.log_master(log_data)
        return log_data
    
    # @trainer.on(Events.EPOCH_STARTED) # for debug
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_results(engine:Engine):
        log_data = { "epoch":engine.state.epoch, }
        # train evaluation
        train_metrics = engine.state.metrics
        log_data["train_loss"] = train_metrics["train_loss"]

        model.eval()
        if cfg.task == "pg":
            # qualitative study first
            qualitative_num = cfg.trainer.qualitative_num
            qualitative_log_data = {}
            qualitative_rouge = metrics.Rouge() # calculate rouge for qualitative study
            with torch.no_grad():
                for batch in test_dataloader:
                    prompts = batch.pop("prompts")
                    labels = batch.pop("labels")
                    batch.to(idist.device())
                    if idist.get_world_size() > 1:
                        res = model.module.generate(batch)
                    else:
                        res = model.generate(batch)
                    res = res[:qualitative_num]
                    labels = labels[:qualitative_num]
                    # calculcate rouge for qulitative study examples
                    qualitative_rouge.update(([item.split() for item in res],[[item.split()] for item in labels]))
                    qualitative_log_data["qualitative_rouge"] = qualitative_rouge.compute()
                    # log qualitative study examples
                    for idx,(prompt,model_res,label) in enumerate(zip(prompts,res,labels)):
                        qualitative_log_data[f"qualitative_{idx}"] = {
                            "prompt":prompt,
                            "model_res":model_res,
                            "label":label,
                        }
                    break
            # log qualitative study
            if idist.get_rank() == 0:
                logger.log_master(qualitative_log_data,if_wandb=False)
        elif cfg.task == "pc":
            # test accuracy
            test_evaluator.run(test_dataloader)
            acc_metrics = test_evaluator.state.metrics
            log_data['test_acc'] = acc_metrics['test_acc']

        # log train evaluation
        logger.log_rank(log_data)
        if idist.get_rank() == 0:
            logger.log_master(log_data)
        return log_data

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.trainer.log_interval))
    def log_step_results(engine:Engine):
        log_data = {
            "iteration":engine.state.iteration,
            "loss_per_step":engine.state.output,
            "lr":optimizer.param_groups[0]["lr"],
        }
        logger.log_rank(log_data)
        if idist.get_rank() == 0:
            logger.log_master(log_data)

    if cfg.task=="pg":
        train_data_collator = PGDataCollator(cfg.data,"train")
        train_dataset = PGDataset(cfg.data,split="train")
    elif cfg.task=="pc":
        train_data_collator = PCDataCollator(cfg.data)
        train_dataset = PCDataset(cfg.data,split="train")
    else:
        raise NotImplementedError(f"Task {cfg.task} is not supported")
    if idist.get_rank() == 0:
        logger.log_master({
            "train dataset prompt_key":f"{train_dataset.prompt_key}"
        },if_wandb=False)

    train_dataloader = idist.auto_dataloader(
        train_dataset,batch_size=cfg.trainer.batch, 
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
        collate_fn=train_data_collator,
        shuffle=True,drop_last=True,
    )
    if cfg.task=="pg":
        test_data_collator = PGDataCollator(cfg.data,"test")
        test_dataset = PGDataset(cfg.data,split="test")
    elif cfg.task=="pc":
        test_data_collator = PCDataCollator(cfg.data)
        test_dataset = PCDataset(cfg.data,split="validation")
    test_dataloader = idist.auto_dataloader(
        test_dataset,batch_size=cfg.trainer.batch,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
        collate_fn=test_data_collator,
        shuffle=False,drop_last=False,
    )

    lrscheduler = build_lrscheduler(optimizer,cfg.trainer,len(train_dataset)//cfg.trainer.batch)
    trainer.add_event_handler(Events.ITERATION_STARTED,lrscheduler)

    # checkpointing; distributed is automatically handled
    to_save = {"model":model,"optimizer":optimizer,"trainer":trainer}
    checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.jobname}"
    checkpoint = Checkpoint(to_save,checkpoint_dir)#,global_step_from_engine(trainer))
    test_evaluator.add_event_handler(Events.STARTED,checkpoint)

    trainer.run(train_dataloader,max_epochs=cfg.trainer.epochs)

@hydra.main(version_base=None,config_path="config", config_name="basic")
def main(cfg: DictConfig) -> None:
    backend = cfg.distributed.backend
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(main_engine, cfg)
    return

if __name__ == '__main__':
    main()
