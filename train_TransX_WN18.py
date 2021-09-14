import numpy as np
from openke.config import Trainer, Tester
from openke.module.model import TransX
from openke.module.loss import MarginLoss, SoftplusLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/WN18/",
    batch_size = 2000,
    # nbatches=100,
    threads=8,
    sampling_mode="cross",
    bern_flag=1,
    filter_flag=1,
    neg_ent=10,
    neg_rel=0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18/", "link")

# define the model
transx = TransX(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True,
    n_head=16,
    d_key=50
)

params = [value.numel() for value in transx.parameters()]
print('model parameters: ', np.sum(params))

# define the loss function
model = NegativeSampling(
    model=transx,
    loss=MarginLoss(margin=10.0),
    batch_size=train_dataloader.get_batch_size()
)
# train the model
trainer = Trainer(model=model, data_loader=train_dataloader, train_times=2000, alpha=0.5, use_gpu=True,
                  save_steps=100, checkpoint_dir='./checkpoint/TransX/WN18')
trainer.run()
transx.save_checkpoint('./checkpoint/TransX-WN18.ckpt')

# test the model
transx.load_checkpoint('./checkpoint/TransX-WN18.ckpt')
tester = Tester(model=transx, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=True)