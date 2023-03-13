from models import GCA_Model
from .structgnn import StructGNN_Design


class GCA_Design(StructGNN_Design):
    def __init__(self, args, device, steps_per_epoch):
        StructGNN_Design.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        
    def _build_model(self):
        return GCA_Model(self.args).to(self.device)