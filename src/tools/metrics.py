class CalMetric:
    def __init__(self):
        pass
     
    def diversity(self,  S_pred, S_ref):
        pass

    def recovery(self, S_pred, S_ref):
        pass

    def MSE(self, S_pred, S_ref):
        '''
        输入: S_pred是算法预测的蛋白序列, S_ref是ground truth蛋白序列
        -->通过ESMFold预测S_pred和S_ref的蛋白结构, 并进行对齐, 计算MSE
        返回: MSE
        '''
        X_pred = ESMFold(S_pred)
        S_ref = ESMFold(S_ref)
        X_pred, X_ref = align(X_pred, X_ref)
        result = ComputeMSE(X_pred, X_ref)
        return result

    def Robustness(self, model, X, S):
        pass

    def Efficiency(self, model, X, S):
        pass