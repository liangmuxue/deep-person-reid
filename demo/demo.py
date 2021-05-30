import torchreid
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name='pcb_p6',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
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

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

weight_path = 'log/resnet50/model.pth.tar-60'
torchreid.utils.load_pretrained_weights(model, weight_path)

engine.run(
    save_dir='../log/resnet50',
    max_epoch=1,
    eval_freq=10,
    print_freq=10,
    test_only=True
)