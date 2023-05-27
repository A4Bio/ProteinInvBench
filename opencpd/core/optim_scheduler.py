from torch import optim

def get_optim_scheduler(lr, epoch, model, steps_per_epoch):
    optimizer = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epoch)
    return optimizer, scheduler