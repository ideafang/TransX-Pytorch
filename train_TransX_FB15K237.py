import numpy as np
from openke.config import Trainer, Tester
from openke.module.model import TransX
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

transx = TransX(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True,
    n_head=4,
    d_key=200
)

params = [value.numel() for value in transx.parameters()]
print('model parameters: ', np.sum(params))

# define the loss function
model = NegativeSampling(
    model=transx,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

trainer = Trainer(model=model, data_loader=train_dataloader, train_times=3000, alpha=1.0, use_gpu=True,
                  save_steps=500, checkpoint_dir='./checkpoint/FB15K237')
trainer.run()
transx.save_checkpoint('./checkpoint/FB15K237-TransX.ckpt')

# test the model
transx.load_checkpoint('./checkpoint/FB15K237-TransX.ckpt')
tester = Tester(model=transx, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=True)
