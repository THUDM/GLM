# Machine Learning HW3 Extra Credit (Generation) #

- **Dataset:** KQA Pro
- **Code Author:** Jiuding Sun
- **Contact:** jiuding.sun@gmail.com
- **Task Description:** KQA Pro is a large-scale, multi-hop KBQA dataset with logical forms and programs. Dataset is evaluated with the execution accuracy of the answer
- **Running Commands:** Before training, preprocess the dataset with bash preprocess.sh and copy the kb.json to the target directory. Run bash DDP_train.sh/tran.sh_ for training; _bash inference.sh_ for inference on given checkpoint.
- **Results:** See result folder for printed training and inference log.
- **Reference**: 
```
@article{shi2020kqa,
  title={Kqa pro: A large-scale dataset with interpretable programs and accurate sparqls for complex question answering over knowledge base},
  author={Shi, Jiaxin and Cao, Shulin and Pan, Liangming and Xiang, Yutong and Hou, Lei and Li, Juanzi and Zhang, Hanwang and He, Bin},
  journal={arXiv preprint arXiv:2007.03875},
  year={2020}
}
```



