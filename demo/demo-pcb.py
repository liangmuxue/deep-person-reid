import torchreid
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    train_sampler='RandomIdentitySampler'
)

model = torchreid.models.build_model(
    name='pcb_p6',
    num_classes=datamanager.num_train_pids,
    loss='triplet',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageTripletEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

# weight_path = 'log/resnet50/model.pth.tar-60'
# torchreid.utils.load_pretrained_weights(model, weight_path)

start_epoch = torchreid.utils.resume_from_checkpoint(
    'log/pcb-triplet/model.pth.tar-120',
    model,
    optimizer
)

engine.run(
    save_dir='../log/pcb-triplet',
    max_epoch=600,
    eval_freq=10,
    print_freq=10,
    start_epoch=start_epoch,
    test_only=False
)