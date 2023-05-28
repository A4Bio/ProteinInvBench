from .structgnn import StructGNN
from opencpd.models import GraphTrans_Model


class GraphTrans(StructGNN):
    def __init__(self, args, device, steps_per_epoch):
        StructGNN.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self):
        return GraphTrans_Model(self.args).to(self.device)