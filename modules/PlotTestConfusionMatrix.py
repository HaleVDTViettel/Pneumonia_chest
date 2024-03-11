import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class PlotTestConfusionMatrixCallback(pl.Callback):
    def on_test_end(seld, trainer, pl_module):
        cm = confusion_matrix(pl_module.test_true_labels, pl_module.test_predicted_labels)
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
        disp.plot()
        plt.show()   
        
        